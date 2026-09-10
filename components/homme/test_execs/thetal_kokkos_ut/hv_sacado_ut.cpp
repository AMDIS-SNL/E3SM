#include <catch2/catch.hpp>

#include <random>
#include <iomanip>

#include "Types.hpp"
#include "Context.hpp"
#include "Elements.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HyperviscosityFunctorImpl.hpp"
#include "HybridVCoord.hpp"
#include "ReferenceElement.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "KernelVariables.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_string_utils.hpp>

using namespace Homme;

extern "C" {
// Builds a real (cube-sphere) mesh/connectivity via the F90 side, since
// HV's boundary exchange assumes every element has 4 real edge connections;
// a hand-rolled, connection-less Connectivity segfaults in unpack().
void init_hv_f90 (const int& ne,
                   const Real* hyai_ptr, const Real* hybi_ptr,
                   const Real* hyam_ptr, const Real* hybm_ptr,
                   Real* dvv, Real* mp,
                   const Real& ps0,
                   const int& hv_subcycle,
                   const Real& hv_nu, const Real& hv_nu_div, const Real& hv_nu_top,
                   const Real& hv_nu_p, const Real& hv_nu_s);
void cleanup_f90();
}

namespace {
constexpr int last_interface_lev_idx = NUM_INTERFACE_LEV - 1;

template<typename ST>
void init_ref_and_derived (ElementsST<ST>& e) {
  Kokkos::deep_copy(e.m_state.m_ref_states.dp_ref, Real(0));
  Kokkos::deep_copy(e.m_state.m_ref_states.theta_ref, Real(0));
  Kokkos::deep_copy(e.m_state.m_ref_states.phi_i_ref, Real(0));
  Kokkos::deep_copy(e.m_derived.m_dpdiss_ave, ST(0));
  Kokkos::deep_copy(e.m_derived.m_dpdiss_biharmonic, ST(0));
}

// Builds a real cube-sphere mesh/connectivity (ne x ne x 6 elements) via the
// F90 side, and inits ref_FE's mass/deriv matrices from it. Geometry itself
// is still randomized by the caller: only the *topology* (each element having
// 4 real edge neighbors) needs to be genuine for BoundaryExchange to work.
void init_mesh_and_ref_elem (Context& c, const int ne, const SimulationParams& params,
                              HybridVCoord& hvcoord, ReferenceElement& ref_FE,
                              const unsigned int seed) {
  hvcoord.random_init(seed);

  auto hyai = Kokkos::create_mirror_view(hvcoord.hybrid_ai);
  auto hybi = Kokkos::create_mirror_view(hvcoord.hybrid_bi);
  auto hyam = Kokkos::create_mirror_view(hvcoord.hybrid_am);
  auto hybm = Kokkos::create_mirror_view(hvcoord.hybrid_bm);
  Kokkos::deep_copy(hyai,hvcoord.hybrid_ai);
  Kokkos::deep_copy(hybi,hvcoord.hybrid_bi);
  Kokkos::deep_copy(hyam,hvcoord.hybrid_am);
  Kokkos::deep_copy(hybm,hvcoord.hybrid_bm);
  HostViewManaged<Real[NUM_PHYSICAL_LEV]> hyam_r(""), hybm_r("");
  for (int i=0; i<NUM_PHYSICAL_LEV; ++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyam_r(i) = ADValue(hyam(ilev)[ivec]);
    hybm_r(i) = ADValue(hybm(ilev)[ivec]);
  }

  std::vector<Real> dvv(NP*NP), mp(NP*NP);
  init_hv_f90(ne, hyai.data(), hybi.data(), hyam_r.data(), hybm_r.data(),
              dvv.data(), mp.data(), hvcoord.ps0,
              params.hypervis_subcycle, params.nu, params.nu_div,
              params.nu_top, params.nu_p, params.nu_s);

  ref_FE.init_mass(mp.data());
  ref_FE.init_deriv(dvv.data());

  auto& bmm = c.create<MpiBuffersManagerMap>();
  if (!bmm.is_connectivity_set()) {
    bmm.set_connectivity(c.get_ptr<Connectivity>());
  }
}

// A failed REQUIRE throws past the cleanup_f90()/Context::finalize_singleton()
// calls at the end of a TEST_CASE, leaving the F90-side mesh state allocated;
// the next TEST_CASE then aborts trying to re-allocate it. Run cleanup from a
// destructor instead so it always runs, pass or fail.
struct F90Cleanup {
  ~F90Cleanup() {
    cleanup_f90();
    Context::finalize_singleton();
  }
};

SimulationParams init_params () {
  SimulationParams params;
  params.params_set = true;
  params.theta_hydrostatic_mode = false;
  params.hypervis_subcycle = 1;
  params.hypervis_subcycle_tom = 0;
  params.hypervis_scaling = 0.0;
  params.nu = 1e-3;
  params.nu_div = 1e-3;
  params.nu_top = 0.0;
  params.nu_p = 0.0;
  params.nu_s = 1e-3;
  params.nu_ratio1 = 1.0;
  params.nu_ratio2 = 1.0;
  return params;
}

} // namespace

TEST_CASE ("hyperviscosity_dp_and_jv_testing")
{
  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");

  using rpdf = std::uniform_real_distribution<Real>;
  using rngAlg = std::mt19937_64;
  rngAlg engine(seed);

  Context::finalize_singleton();
  F90Cleanup f90_cleanup_guard;
  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);

  constexpr int ne = 2;
  const int np1 = 1;
  const Real dt = rpdf(1e-4,1e-2)(engine);
  const Real eta_ave_w = 1.0;
  const Real atol = 1e-6;
  const Real rtol = 1e-4;

  auto params = init_params();
  c.create<SimulationParams>() = params;

  auto& hvcoord = c.create<HybridVCoord>();
  auto& ref_FE  = c.create<ReferenceElement>();
  init_mesh_and_ref_elem(c, ne, params, hvcoord, ref_FE, seed);

  const int num_elems = c.get<Connectivity>().get_num_local_elements();

  ElementsST<Real> elems_ref, elems_0, elems_h;
  ElementsST<DpFadType> elems_dp;
  elems_ref.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_0.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_h.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);

  auto& geo = elems_ref.m_geometry = elems_0.m_geometry = elems_h.m_geometry
            = elems_dp.m_geometry;
  geo.randomize(seed, c.get_ptr<Connectivity>(), c.get<MpiBuffersManagerMap>()[MPI_EXCHANGE]);

  init_ref_and_derived(elems_ref);
  init_ref_and_derived(elems_0);
  init_ref_and_derived(elems_h);
  init_ref_and_derived(elems_dp);

  c.create<SphereOperatorsST<Real>>().setup(geo, ref_FE);
  c.create<SphereOperatorsST<DpFadType>>().setup(geo, ref_FE);

  const auto max_pressure = 1000.0 + hvcoord.ps0;
  elems_ref.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,geo.m_phis);
  elems_0.m_state.import_values(elems_ref.m_state,np1);
  elems_h.m_state.import_values(elems_ref.m_state,np1);
  elems_dp.m_state.import_values(elems_ref.m_state,np1);

  // Note: these hold one Real per *physical* level (or interface level), not
  // per pack, since they are indexed directly by physical/interface level
  // below (in set_derivs/set_derivs_int). Sizing them by NUM_LEV/NUM_LEV_P
  // (the pack counts) instead would leave them far too short and make every
  // access at k>=NUM_LEV (or NUM_LEV_P) an out-of-bounds read.
  ExecViewManaged<Real*[2][NP][NP][NUM_PHYSICAL_LEV]> Vv("", num_elems);
  ExecViewManaged<Real*[NP][NP][NUM_PHYSICAL_LEV]> Vdp("", num_elems);
  ExecViewManaged<Real*[NP][NP][NUM_PHYSICAL_LEV]> Vvth("", num_elems);
  ExecViewManaged<Real*[NP][NP][NUM_INTERFACE_LEV]> Vw("", num_elems);
  ExecViewManaged<Real*[NP][NP][NUM_INTERFACE_LEV]> Vphi("", num_elems);
  genRandArray(Vv, engine, rpdf(-1.0,1.0));
  genRandArray(Vdp, engine, rpdf(-1.0,1.0));
  genRandArray(Vvth, engine, rpdf(-1.0,1.0));
  genRandArray(Vw, engine, rpdf(-1.0,1.0));
  genRandArray(Vphi, engine, rpdf(-1.0,1.0));

  auto v_fad = ekat::scalarize(elems_dp.m_state.m_v);
  auto dp_fad = ekat::scalarize(elems_dp.m_state.m_dp3d);
  auto vth_fad = ekat::scalarize(elems_dp.m_state.m_vtheta_dp);
  auto w_fad = ekat::scalarize(elems_dp.m_state.m_w_i);
  auto phi_fad = ekat::scalarize(elems_dp.m_state.m_phinh_i);
  auto set_derivs = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    v_fad  (ie,np1,0,ip,jp,k).fastAccessDx(0) = Vv  (ie,0,ip,jp,k);
    v_fad  (ie,np1,1,ip,jp,k).fastAccessDx(0) = Vv  (ie,1,ip,jp,k);
    dp_fad (ie,np1,ip,jp,k).fastAccessDx(0) = Vdp (ie,ip,jp,k);
    vth_fad(ie,np1,ip,jp,k).fastAccessDx(0) = Vvth(ie,ip,jp,k);
    w_fad  (ie,np1,ip,jp,k).fastAccessDx(0) = Vw  (ie,ip,jp,k);
    phi_fad(ie,np1,ip,jp,k).fastAccessDx(0) = Vphi(ie,ip,jp,k);
  };
  auto set_derivs_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp) {
    w_fad  (ie,np1,ip,jp,last_interface_lev_idx).fastAccessDx(0) = Vw  (ie,ip,jp,last_interface_lev_idx);
    phi_fad(ie,np1,ip,jp,last_interface_lev_idx).fastAccessDx(0) = Vphi(ie,ip,jp,last_interface_lev_idx);
  };
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> p4({0,0,0,0},{num_elems,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>> p3({0,0,0},{num_elems,NP,NP});
  Kokkos::parallel_for(p4, set_derivs);
  Kokkos::parallel_for(p3, set_derivs_int);

  HyperviscosityFunctorImplST<Real> hv_0(params,geo,elems_0.m_state,elems_0.m_derived);
  HyperviscosityFunctorImplST<Real> hv_h(params,geo,elems_h.m_state,elems_h.m_derived);
  HyperviscosityFunctorImplST<DpFadType> hv_dp(params,geo,elems_dp.m_state,elems_dp.m_derived);

  // Sacado::Fad::SFad's default ctor (StaticFixedStorage() = default, unless
  // SACADO_SFAD_INIT_DEFAULT_CONSTRUCTOR is defined) leaves both the value and
  // the derivative array as raw uninitialized memory. SphereOperatorsST's
  // internal scratch (scalar_buf_ml/vector_buf_ml/vector_buf_sl) is a
  // Kokkos::View of this Fad-typed Pack, constructed via the normal labeled
  // ctor, which default-constructs (and hence leaves garbage in) every
  // element. Zero it explicitly for the Fad-typed instance, since the sphere
  // operators otherwise start from garbage instead of a clean scratch buffer.
  {
    auto& sop_dp = c.get<SphereOperatorsST<DpFadType>>();
    Kokkos::deep_copy(sop_dp.vector_buf_sl, DpFadType(0));
    Kokkos::deep_copy(sop_dp.scalar_buf_ml, DpFadType(0));
    Kokkos::deep_copy(sop_dp.vector_buf_ml, DpFadType(0));
  }

  FunctorsBuffersManager fbm;
  fbm.request_size(hv_0.requested_buffer_size());
  fbm.request_size(hv_h.requested_buffer_size());
  fbm.request_size(hv_dp.requested_buffer_size());
  fbm.allocate();
  hv_0.init_buffers(fbm);
  hv_h.init_buffers(fbm);
  hv_dp.init_buffers(fbm);
  hv_0.init_boundary_exchanges();
  hv_h.init_boundary_exchanges();
  hv_dp.init_boundary_exchanges();

  hv_dp.run(np1,dt,eta_ave_w);
  hv_0.run(np1,dt,eta_ave_w);

  auto v_0 = ekat::scalarize(elems_0.m_state.m_v);
  auto dp_0 = ekat::scalarize(elems_0.m_state.m_dp3d);
  auto vth_0 = ekat::scalarize(elems_0.m_state.m_vtheta_dp);
  auto w_0 = ekat::scalarize(elems_0.m_state.m_w_i);
  auto phi_0 = ekat::scalarize(elems_0.m_state.m_phinh_i);
  auto v_dp = ekat::scalarize(elems_dp.m_state.m_v);
  auto dp_dp = ekat::scalarize(elems_dp.m_state.m_dp3d);
  auto vth_dp = ekat::scalarize(elems_dp.m_state.m_vtheta_dp);
  auto w_dp = ekat::scalarize(elems_dp.m_state.m_w_i);
  auto phi_dp = ekat::scalarize(elems_dp.m_state.m_phinh_i);

  const std::vector<Real> h_vals = {1e-1,1e-2,1e-3,1e-4,1e-5,1e-6};
  std::vector<Real> err_fd;
  for (const auto h : h_vals) {
    elems_h.m_state.import_values(elems_ref.m_state,np1);
    auto v_h = ekat::scalarize(elems_h.m_state.m_v);
    auto dp_h = ekat::scalarize(elems_h.m_state.m_dp3d);
    auto vth_h = ekat::scalarize(elems_h.m_state.m_vtheta_dp);
    auto w_h = ekat::scalarize(elems_h.m_state.m_w_i);
    auto phi_h = ekat::scalarize(elems_h.m_state.m_phinh_i);

    auto perturb = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      v_h   (ie,np1,0,ip,jp,k) += h*Vv  (ie,0,ip,jp,k);
      v_h   (ie,np1,1,ip,jp,k) += h*Vv  (ie,1,ip,jp,k);
      dp_h (ie,np1,ip,jp,k) += h*Vdp (ie,ip,jp,k);
      vth_h(ie,np1,ip,jp,k) += h*Vvth(ie,ip,jp,k);
      w_h  (ie,np1,ip,jp,k) += h*Vw  (ie,ip,jp,k);
      phi_h(ie,np1,ip,jp,k) += h*Vphi(ie,ip,jp,k);
    };
    auto perturb_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp) {
      w_h  (ie,np1,ip,jp,last_interface_lev_idx) += h*Vw  (ie,ip,jp,last_interface_lev_idx);
      phi_h(ie,np1,ip,jp,last_interface_lev_idx) += h*Vphi(ie,ip,jp,last_interface_lev_idx);
    };
    Kokkos::parallel_for(p4, perturb);
    Kokkos::parallel_for(p3, perturb_int);

    hv_h.run(np1,dt,eta_ave_w);

    auto v_out_h = ekat::scalarize(elems_h.m_state.m_v);
    auto dp_out_h = ekat::scalarize(elems_h.m_state.m_dp3d);
    auto vth_out_h = ekat::scalarize(elems_h.m_state.m_vtheta_dp);
    auto w_out_h = ekat::scalarize(elems_h.m_state.m_w_i);
    auto phi_out_h = ekat::scalarize(elems_h.m_state.m_phinh_i);
    auto linf = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k, Real& accum) {
      const Real du_fd   = (v_out_h(ie,np1,0,ip,jp,k) - v_0(ie,np1,0,ip,jp,k))/h;
      const Real dv_fd   = (v_out_h(ie,np1,1,ip,jp,k) - v_0(ie,np1,1,ip,jp,k))/h;
      const Real ddp_fd  = (dp_out_h(ie,np1,ip,jp,k)  - dp_0 (ie,np1,ip,jp,k))/h;
      const Real dvth_fd = (vth_out_h(ie,np1,ip,jp,k) - vth_0(ie,np1,ip,jp,k))/h;
      const Real dw_fd   = (w_out_h(ie,np1,ip,jp,k)   - w_0(ie,np1,ip,jp,k))/h;
      const Real dphi_fd = (phi_out_h(ie,np1,ip,jp,k) - phi_0(ie,np1,ip,jp,k))/h;
      const Real du_ex   = v_dp(ie,np1,0,ip,jp,k).fastAccessDx(0);
      const Real dv_ex   = v_dp(ie,np1,1,ip,jp,k).fastAccessDx(0);
      const Real ddp_ex  = dp_dp (ie,np1,ip,jp,k).fastAccessDx(0);
      const Real dvth_ex = vth_dp(ie,np1,ip,jp,k).fastAccessDx(0);
      const Real dw_ex   = w_dp(ie,np1,ip,jp,k).fastAccessDx(0);
      const Real dphi_ex = phi_dp(ie,np1,ip,jp,k).fastAccessDx(0);
      Real lcl = Kokkos::abs(du_fd-du_ex);
      lcl = Kokkos::max(lcl, Kokkos::abs(dv_fd-dv_ex));
      lcl = Kokkos::max(lcl, Kokkos::abs(ddp_fd-ddp_ex));
      lcl = Kokkos::max(lcl, Kokkos::abs(dvth_fd-dvth_ex));
      lcl = Kokkos::max(lcl, Kokkos::abs(dw_fd-dw_ex));
      lcl = Kokkos::max(lcl, Kokkos::abs(dphi_fd-dphi_ex));
      if (lcl > accum) accum = lcl;
    };
    auto linf_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, Real& accum) {
      const Real dwl_fd   = (w_out_h(ie,np1,ip,jp,last_interface_lev_idx)   - w_0(ie,np1,ip,jp,last_interface_lev_idx))/h;
      const Real dphil_fd = (phi_out_h(ie,np1,ip,jp,last_interface_lev_idx) - phi_0(ie,np1,ip,jp,last_interface_lev_idx))/h;
      const Real dwl_ex   = w_dp(ie,np1,ip,jp,last_interface_lev_idx).fastAccessDx(0);
      const Real dphil_ex = phi_dp(ie,np1,ip,jp,last_interface_lev_idx).fastAccessDx(0);
      Real lcl = Kokkos::abs(dwl_fd-dwl_ex);
      lcl = Kokkos::max(lcl, Kokkos::abs(dphil_fd-dphil_ex));
      if (lcl > accum) accum = lcl;
    };
    Real err_phys = 0;
    Real err_int  = 0;
    Kokkos::parallel_reduce(p4, linf, Kokkos::Max<Real>(err_phys));
    Kokkos::parallel_reduce(p3, linf_int, Kokkos::Max<Real>(err_int));
    err_fd.emplace_back(Kokkos::max(err_phys,err_int));
  }

  std::cout << std::setprecision(16)
            << "  h = [" << ekat::join(h_vals,",") << "]\n"
            << "  |dF/dp - FD|_inf = [" << ekat::join(err_fd,",") << "]\n";
  const Real min_err = *std::min_element(err_fd.begin(),err_fd.end());
  REQUIRE((err_fd[0] < atol or min_err < err_fd[0]*rtol));

  // Reset elems_0 back to the pre-HV base state (hv_0.run() above already
  // advanced it), then linearize about that same base state: init_J snapshots
  // it, then run() (the real, nonlinear forward pass) must run again before
  // run_JV can be called.
  elems_0.m_state.import_values(elems_ref.m_state,np1);

  // x = the same direction V used for the FD/DpFadType check above (extracted
  // as a StateSnapshot from elems_dp's stored derivative), y = J*x.
  StateSnapshot x = elems_dp.m_state.take_deriv_snapshot(np1,0);
  StateSnapshot y = x.clone();

  hv_0.init_J(np1);
  hv_0.run(np1,dt,eta_ave_w);
  hv_0.run_JV(np1,x,y);

  auto v_jv = ekat::scalarize(y.v);
  auto dp_jv = ekat::scalarize(y.dp3d);
  auto vth_jv = ekat::scalarize(y.vtheta_dp);
  auto w_jv = ekat::scalarize(y.w_i);
  auto phi_jv = ekat::scalarize(y.phinh_i);

  auto v_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_dp);
  auto dp_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dp_dp);
  auto vth_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vth_dp);
  auto w_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w_dp);
  auto phi_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi_dp);
  auto v_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_jv);
  auto dp_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dp_jv);
  auto vth_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vth_jv);
  auto w_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w_jv);
  auto phi_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi_jv);

  for (int ie = 0; ie < num_elems; ++ie) {
    for (int ip = 0; ip < NP; ++ip) {
      for (int jp = 0; jp < NP; ++jp) {
        for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
          const auto du_src = v_jv_h(ie,0,ip,jp,k);
          const auto du_tgt = v_dp_h(ie,np1,0,ip,jp,k).dx(0);
          CHECK_THAT(du_src, Catch::WithinRel(du_tgt, rtol) || Catch::WithinAbs(du_tgt, atol));

          const auto dv_src = v_jv_h(ie,1,ip,jp,k);
          const auto dv_tgt = v_dp_h(ie,np1,1,ip,jp,k).dx(0);
          CHECK_THAT(dv_src, Catch::WithinRel(dv_tgt, rtol) || Catch::WithinAbs(dv_tgt, atol));

          const auto ddp_src = dp_jv_h(ie,ip,jp,k);
          const auto ddp_tgt = dp_dp_h(ie,np1,ip,jp,k).dx(0);
          CHECK_THAT(ddp_src, Catch::WithinRel(ddp_tgt, rtol) || Catch::WithinAbs(ddp_tgt, atol));

          const auto dvth_src = vth_jv_h(ie,ip,jp,k);
          const auto dvth_tgt = vth_dp_h(ie,np1,ip,jp,k).dx(0);
          CHECK_THAT(dvth_src, Catch::WithinRel(dvth_tgt, rtol) || Catch::WithinAbs(dvth_tgt, atol));

          const auto dw_src = w_jv_h(ie,ip,jp,k);
          const auto dw_tgt = w_dp_h(ie,np1,ip,jp,k).dx(0);
          CHECK_THAT(dw_src, Catch::WithinRel(dw_tgt, rtol) || Catch::WithinAbs(dw_tgt, atol));

          const auto dphi_src = phi_jv_h(ie,ip,jp,k);
          const auto dphi_tgt = phi_dp_h(ie,np1,ip,jp,k).dx(0);
          CHECK_THAT(dphi_src, Catch::WithinRel(dphi_tgt, rtol) || Catch::WithinAbs(dphi_tgt, atol));
        }
        const auto dwl_src = w_jv_h(ie,ip,jp,last_interface_lev_idx);
        const auto dwl_tgt = w_dp_h(ie,np1,ip,jp,last_interface_lev_idx).dx(0);
        CHECK_THAT(dwl_src, Catch::WithinRel(dwl_tgt, rtol) || Catch::WithinAbs(dwl_tgt, atol));
        const auto dphil_src = phi_jv_h(ie,ip,jp,last_interface_lev_idx);
        const auto dphil_tgt = phi_dp_h(ie,np1,ip,jp,last_interface_lev_idx).dx(0);
        CHECK_THAT(dphil_src, Catch::WithinRel(dphil_tgt, rtol) || Catch::WithinAbs(dphil_tgt, atol));
      }
    }
  }
}

TEST_CASE ("hyperviscosity_jtv_testing") {
  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");

  using rpdf = std::uniform_real_distribution<Real>;
  using rngAlg = std::mt19937_64;
  rngAlg engine(seed);

  Context::finalize_singleton();
  F90Cleanup f90_cleanup_guard;
  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);

  constexpr int ne = 2;
  const int np1 = 1;
  const Real dt = rpdf(1e-4,1e-2)(engine);
  const Real eta_ave_w = 1.0;
  const Real atol = 1e-8;
  const Real rtol = 1e-8;

  auto params = init_params();
  c.create<SimulationParams>() = params;

  auto& hvcoord = c.create<HybridVCoord>();
  auto& ref_FE  = c.create<ReferenceElement>();
  init_mesh_and_ref_elem(c, ne, params, hvcoord, ref_FE, seed);

  const int num_elems = c.get<Connectivity>().get_num_local_elements();

  ElementsST<Real> elems;
  elems.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems.m_geometry.randomize(seed, c.get_ptr<Connectivity>(), c.get<MpiBuffersManagerMap>()[MPI_EXCHANGE]);
  init_ref_and_derived(elems);

  c.create<SphereOperatorsST<Real>>().setup(elems.m_geometry, ref_FE);

  const auto max_pressure = 1000.0 + hvcoord.ps0;
  elems.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,elems.m_geometry.m_phis);

  // xa, xb: two independent random directions. ya = J*xa is not needed; we
  // only need yb = J*xb and ya = J^T*xa to check <xa,J*xb> = <J^T*xa,xb>.
  StateSnapshot xa(num_elems), xb(num_elems), ya(num_elems), yb(num_elems);
  xa.randomize(seed,  max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, elems.m_geometry.m_phis);
  xb.randomize(seed+1,max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, elems.m_geometry.m_phis);

  HyperviscosityFunctorImplST<Real> hv(params,elems.m_geometry,elems.m_state,elems.m_derived);

  FunctorsBuffersManager fbm;
  fbm.request_size(hv.requested_buffer_size());
  fbm.allocate();
  hv.init_buffers(fbm);
  hv.init_boundary_exchanges();

  // State at which J is evaluated.
  hv.init_J(np1);
  hv.run(np1,dt,eta_ave_w);

  // J*xb -> yb
  hv.run_JV(np1,xb,yb);
  // J^T*xa -> ya
  hv.run_JtV(np1,xa,ya);

  auto xa_v = ekat::scalarize(xa.v);
  auto xa_dp = ekat::scalarize(xa.dp3d);
  auto xa_vth = ekat::scalarize(xa.vtheta_dp);
  auto xa_w = ekat::scalarize(xa.w_i);
  auto xa_phi = ekat::scalarize(xa.phinh_i);
  auto xb_v = ekat::scalarize(xb.v);
  auto xb_dp = ekat::scalarize(xb.dp3d);
  auto xb_vth = ekat::scalarize(xb.vtheta_dp);
  auto xb_w = ekat::scalarize(xb.w_i);
  auto xb_phi = ekat::scalarize(xb.phinh_i);
  auto ya_v = ekat::scalarize(ya.v);
  auto ya_dp = ekat::scalarize(ya.dp3d);
  auto ya_vth = ekat::scalarize(ya.vtheta_dp);
  auto ya_w = ekat::scalarize(ya.w_i);
  auto ya_phi = ekat::scalarize(ya.phinh_i);
  auto yb_v = ekat::scalarize(yb.v);
  auto yb_dp = ekat::scalarize(yb.dp3d);
  auto yb_vth = ekat::scalarize(yb.vtheta_dp);
  auto yb_w = ekat::scalarize(yb.w_i);
  auto yb_phi = ekat::scalarize(yb.phinh_i);

  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> p4({0,0,0,0},{num_elems,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>> p3({0,0,0},{num_elems,NP,NP});
  Real2 dot;
  Real2 dot_int;
  // (xa,yb) = (xa,J*xb) = (J^T*xa,xb) = (ya,xb)
  Kokkos::parallel_reduce(p4, KOKKOS_LAMBDA(const int ie, const int ip, const int jp, const int k, Real2& acc) {
    acc.v[0] += xa_v(ie,0,ip,jp,k) * yb_v(ie,0,ip,jp,k)
             +  xa_v(ie,1,ip,jp,k) * yb_v(ie,1,ip,jp,k)
             +  xa_dp (ie,ip,jp,k) * yb_dp (ie,ip,jp,k)
             +  xa_vth(ie,ip,jp,k) * yb_vth(ie,ip,jp,k)
             +  xa_w  (ie,ip,jp,k) * yb_w  (ie,ip,jp,k)
             +  xa_phi(ie,ip,jp,k) * yb_phi(ie,ip,jp,k);
    acc.v[1] += ya_v(ie,0,ip,jp,k) * xb_v(ie,0,ip,jp,k)
             +  ya_v(ie,1,ip,jp,k) * xb_v(ie,1,ip,jp,k)
             +  ya_dp (ie,ip,jp,k) * xb_dp (ie,ip,jp,k)
             +  ya_vth(ie,ip,jp,k) * xb_vth(ie,ip,jp,k)
             +  ya_w  (ie,ip,jp,k) * xb_w  (ie,ip,jp,k)
             +  ya_phi(ie,ip,jp,k) * xb_phi(ie,ip,jp,k);
  }, dot);
  Kokkos::parallel_reduce(p3, KOKKOS_LAMBDA(const int ie, const int ip, const int jp, Real2& acc) {
    acc.v[0] += xa_w  (ie,ip,jp,last_interface_lev_idx) * yb_w  (ie,ip,jp,last_interface_lev_idx)
             +  xa_phi(ie,ip,jp,last_interface_lev_idx) * yb_phi(ie,ip,jp,last_interface_lev_idx);
    acc.v[1] += ya_w  (ie,ip,jp,last_interface_lev_idx) * xb_w  (ie,ip,jp,last_interface_lev_idx)
             +  ya_phi(ie,ip,jp,last_interface_lev_idx) * xb_phi(ie,ip,jp,last_interface_lev_idx);
  }, dot_int);
  dot += dot_int;

  std::cout << std::setprecision(15)
            << "  <xa, J*xb> = " << dot.v[0] << ",  <J^T*xa, xb> = " << dot.v[1] << "\n";
  CHECK_THAT(dot.v[0], Catch::WithinRel(dot.v[1], rtol) || Catch::WithinAbs(dot.v[1], atol));
}

// Discrete self-adjointness check for the two building blocks a from-scratch
// (discretize-then-optimize) HV adjoint would reuse verbatim, in reverse order,
// instead of re-deriving separate transpose kernels:
//  - laplace_simple / vlaplace_sphere_wk_contra (element-local, pre-exchange):
//    both are literally divergence_sphere_wk(gradient_sphere(.)) (or the
//    vector analogue built from div/grad/curl in the same "weak-form"
//    fashion), i.e. algebraically of the form D^T*W*D for the actual discrete
//    D (derivative matrix) and diagonal quadrature-weight W used by the code
//    -- symmetric by construction, not because the *continuous* Laplacian is
//    self-adjoint. Checked here per-element (no exchange involved).
//  - BoundaryExchangeST::exchange()/exchange(rspheremp) (cross-element DSS):
//    at a shared DOF, rspheremp is the same value for every owning element
//    (it comes from the already-assembled mass matrix), so the map is
//    "sum then scale by a shared weight", whose matrix is symmetric. Checked
//    here globally (summed over the whole mesh), since that's inherently a
//    cross-element property.
// If either check fails, run_JV/run_JtV cannot simply reuse the same kernels
// in reverse: a genuine discrete transpose would need to be written instead.
TEST_CASE ("hv_sphere_ops_discrete_self_adjoint") {
  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");

  using rpdf = std::uniform_real_distribution<Real>;
  using rngAlg = std::mt19937_64;
  rngAlg engine(seed);

  Context::finalize_singleton();
  F90Cleanup f90_cleanup_guard;
  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);

  constexpr int ne = 2;
  auto params = init_params();
  c.create<SimulationParams>() = params;

  auto& hvcoord = c.create<HybridVCoord>();
  auto& ref_FE  = c.create<ReferenceElement>();
  init_mesh_and_ref_elem(c, ne, params, hvcoord, ref_FE, seed);

  const int num_elems = c.get<Connectivity>().get_num_local_elements();

  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  geo.randomize(seed, c.get_ptr<Connectivity>(), c.get<MpiBuffersManagerMap>()[MPI_EXCHANGE]);

  auto& sphop = c.create<SphereOperatorsST<Real>>();
  sphop.setup(geo, ref_FE);
  const auto policy = Homme::get_default_team_policy<ExecSpace>(num_elems);
  sphop.allocate_buffers(policy);

  auto& bmm = c.get<MpiBuffersManagerMap>(); // already created+connected by init_mesh_and_ref_elem

  using PT = PackType<Real>;
  const Real tol = 1e-10;

  const Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> p4_mid({0,0,0,0},{num_elems,NP,NP,NUM_PHYSICAL_LEV});
  const Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>> p5_mid({0,0,0,0,0},{num_elems,2,NP,NP,NUM_PHYSICAL_LEV});

  // ---- 1a. Scalar Laplacian, element-local: <a, L(b)> == <L(a), b> per element ----
  {
    ExecViewManaged<Real*[NP][NP][NUM_PHYSICAL_LEV]> a_phys("",num_elems), b_phys("",num_elems);
    genRandArray(a_phys, engine, rpdf(-1.0,1.0));
    genRandArray(b_phys, engine, rpdf(-1.0,1.0));

    ExecViewManaged<PT*[NP][NP][NUM_LEV]> a("",num_elems), b("",num_elems), La("",num_elems), Lb("",num_elems);
    Kokkos::deep_copy(a, PT(0));
    Kokkos::deep_copy(b, PT(0));
    auto a_s = ekat::scalarize(a);
    auto b_s = ekat::scalarize(b);
    Kokkos::parallel_for(p4_mid, KOKKOS_LAMBDA(int ie,int ip,int jp,int k) {
      a_s(ie,ip,jp,k) = a_phys(ie,ip,jp,k);
      b_s(ie,ip,jp,k) = b_phys(ie,ip,jp,k);
    });

    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team) {
      KernelVariables kv(team);
      sphop.laplace_simple(kv, Homme::subview(a,kv.ie), Homme::subview(La,kv.ie));
      sphop.laplace_simple(kv, Homme::subview(b,kv.ie), Homme::subview(Lb,kv.ie));
    });
    Kokkos::fence();

    auto La_s_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ekat::scalarize(La));
    auto Lb_s_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ekat::scalarize(Lb));
    auto a_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_phys);
    auto b_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_phys);

    for (int ie=0; ie<num_elems; ++ie) {
      Real dot1=0, dot2=0;
      for (int ip=0; ip<NP; ++ip)
      for (int jp=0; jp<NP; ++jp)
      for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
        dot1 += a_h(ie,ip,jp,k) * Lb_s_h(ie,ip,jp,k);
        dot2 += La_s_h(ie,ip,jp,k) * b_h(ie,ip,jp,k);
      }
      CHECK_THAT(dot1, Catch::WithinRel(dot2, tol) || Catch::WithinAbs(dot2, tol));
    }
    std::cout << "  laplace_simple self-adjoint: checked " << num_elems << " elements\n";
  }

  // ---- 1b. Vector Laplacian, element-local: <a, L(b)> == <L(a), b> per element ----
  // Use a nu_ratio far from 0/1 to exercise the div-scaling branch too.
  {
    const Real nu_ratio = 1.7;
    ExecViewManaged<Real*[2][NP][NP][NUM_PHYSICAL_LEV]> a_phys("",num_elems), b_phys("",num_elems);
    genRandArray(a_phys, engine, rpdf(-1.0,1.0));
    genRandArray(b_phys, engine, rpdf(-1.0,1.0));

    ExecViewManaged<PT*[2][NP][NP][NUM_LEV]> a("",num_elems), b("",num_elems), La("",num_elems), Lb("",num_elems);
    Kokkos::deep_copy(a, PT(0));
    Kokkos::deep_copy(b, PT(0));
    auto a_s = ekat::scalarize(a);
    auto b_s = ekat::scalarize(b);
    Kokkos::parallel_for(p5_mid, KOKKOS_LAMBDA(int ie,int ic,int ip,int jp,int k) {
      a_s(ie,ic,ip,jp,k) = a_phys(ie,ic,ip,jp,k);
      b_s(ie,ic,ip,jp,k) = b_phys(ie,ic,ip,jp,k);
    });

    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team) {
      KernelVariables kv(team);
      sphop.vlaplace_sphere_wk_contra(kv, nu_ratio, Homme::subview(a,kv.ie), Homme::subview(La,kv.ie));
      sphop.vlaplace_sphere_wk_contra(kv, nu_ratio, Homme::subview(b,kv.ie), Homme::subview(Lb,kv.ie));
    });
    Kokkos::fence();

    auto La_s_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ekat::scalarize(La));
    auto Lb_s_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ekat::scalarize(Lb));
    auto a_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_phys);
    auto b_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_phys);

    for (int ie=0; ie<num_elems; ++ie) {
      Real dot1=0, dot2=0;
      for (int ic=0; ic<2; ++ic)
      for (int ip=0; ip<NP; ++ip)
      for (int jp=0; jp<NP; ++jp)
      for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
        dot1 += a_h(ie,ic,ip,jp,k) * Lb_s_h(ie,ic,ip,jp,k);
        dot2 += La_s_h(ie,ic,ip,jp,k) * b_h(ie,ic,ip,jp,k);
      }
      CHECK_THAT(dot1, Catch::WithinRel(dot2, tol) || Catch::WithinAbs(dot2, tol));
    }
    std::cout << "  vlaplace_sphere_wk_contra self-adjoint: checked " << num_elems << " elements\n";
  }

  // ---- 2. Cross-element exchange, global: <a, E(b)> == <E(a), b>, weighted and unweighted ----
  {
    ExecViewManaged<Real*[NP][NP][NUM_PHYSICAL_LEV]> a_phys("",num_elems), b_phys("",num_elems);
    genRandArray(a_phys, engine, rpdf(-1.0,1.0));
    genRandArray(b_phys, engine, rpdf(-1.0,1.0));

    ExecViewManaged<PT*[NP][NP][NUM_LEV]> a("",num_elems), b("",num_elems), a0("",num_elems), b0("",num_elems);
    Kokkos::deep_copy(a, PT(0));
    Kokkos::deep_copy(b, PT(0));
    auto a_s = ekat::scalarize(a);
    auto b_s = ekat::scalarize(b);
    Kokkos::parallel_for(p4_mid, KOKKOS_LAMBDA(int ie,int ip,int jp,int k) {
      a_s(ie,ip,jp,k) = a_phys(ie,ip,jp,k);
      b_s(ie,ip,jp,k) = b_phys(ie,ip,jp,k);
    });
    Kokkos::deep_copy(a0, a);
    Kokkos::deep_copy(b0, b);

    auto be = std::make_shared<BoundaryExchangeST<Real>>();
    be->set_buffers_manager(bmm[MPI_EXCHANGE]);
    be->set_num_fields(0,0,2);
    be->register_field(a);
    be->register_field(b);
    be->registration_completed();

    auto global_dots = [&] (const ExecViewManaged<PT*[NP][NP][NUM_LEV]>& a_after,
                             const ExecViewManaged<PT*[NP][NP][NUM_LEV]>& b_after) {
      auto a_after_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ekat::scalarize(a_after));
      auto b_after_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ekat::scalarize(b_after));
      auto a0_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ekat::scalarize(a0));
      auto b0_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ekat::scalarize(b0));
      Real dot1=0, dot2=0;
      for (int ie=0; ie<num_elems; ++ie)
      for (int ip=0; ip<NP; ++ip)
      for (int jp=0; jp<NP; ++jp)
      for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
        dot1 += a0_h(ie,ip,jp,k) * b_after_h(ie,ip,jp,k); // <a0, E(b0)>
        dot2 += a_after_h(ie,ip,jp,k) * b0_h(ie,ip,jp,k); // <E(a0), b0>
      }
      return std::make_pair(dot1,dot2);
    };

    // Weighted exchange (the flavor used between HV's first and second Laplacian).
    be->exchange(geo.m_rspheremp);
    auto dots_w = global_dots(a,b);
    CHECK_THAT(dots_w.first, Catch::WithinRel(dots_w.second, tol) || Catch::WithinAbs(dots_w.second, tol));
    std::cout << "  exchange(rspheremp) self-adjoint: <a0,Eb0>=" << dots_w.first
              << ", <Ea0,b0>=" << dots_w.second << "\n";

    // Unweighted exchange (the flavor used between HyperPreExchange and UpdateStates).
    Kokkos::deep_copy(a, a0);
    Kokkos::deep_copy(b, b0);
    be->exchange();
    auto dots_u = global_dots(a,b);
    CHECK_THAT(dots_u.first, Catch::WithinRel(dots_u.second, tol) || Catch::WithinAbs(dots_u.second, tol));
    std::cout << "  exchange() self-adjoint: <a0,Eb0>=" << dots_u.first
              << ", <Ea0,b0>=" << dots_u.second << "\n";
  }
}
