#include <catch2/catch.hpp>

#include "Types.hpp"
#include "Tape.hpp"
#include "Context.hpp"
#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"
#include "DirkFunctor.hpp"
#include "DirkFunctorImpl.hpp"
#include "LimiterFunctor.hpp"
#include "HyperviscosityFunctor.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "PhysicalConstants.hpp"
#include "TimeLevel.hpp"
#include "prim_advance_exp.hpp"
#include "prim_advance_adj.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_string_utils.hpp>
#include <ekat_comm.hpp>

#include <iomanip>

namespace Homme
{

extern "C" {
// Even if we don't run the f90 code in this unit test, it is easier to
// init from f90, which takes care of creating the grid and decomposing it
void init_f90 (const int& ne,
               const Real* hyai_ptr, const Real* hybi_ptr,
               const Real* hyam_ptr, const Real* hybm_ptr,
               Real* dvv, Real* mp,
               const Real& ps0);
void init_geo_views_f90 (Real*& d_ptr, Real*& dinv_ptr,
               const Real*& phis_ptr, const Real*& gradphis_ptr,
               Real*& fcor_ptr,
               Real*& sphmp_ptr, Real*& rspmp_ptr,
               Real*& tVisc_ptr, Real*& sph2c_ptr,
               Real*& metdet_ptr, Real*& metinv_ptr);
void cleanup_f90();
}

namespace {
// A failed CHECK/REQUIRE throws past the cleanup_f90()/Context::finalize_singleton()
// calls at the end of a TEST_CASE, leaving the F90-side mesh state allocated;
// the next TEST_CASE then aborts trying to re-allocate it. Run cleanup from a
// destructor instead, so it always runs, pass or fail.
struct F90Cleanup {
  ~F90Cleanup() {
    cleanup_f90();
    Context::finalize_singleton();
  }
};
} // anonymous namespace

// Full adjoint test for prim_advance_exp (ttype10_imex): unlike
// imex_adjoint_ut.cpp's ttype10_imex_adjoint test (which only covers the
// CAAR/DIRK stages), this also exercises the hyperviscosity step and the
// w_i(n0) surface fix, i.e. prim_advance_adj as a whole.
//
// As in imex_adjoint_ut.cpp, the ground truth tangent is obtained by running
// the real (nonlinear) forward code with Sacado (DpFadType) elements: the
// CAAR/DIRK stages already do this via ttype10_imex_timestep<DpFadType>, and
// here we do the same for HV, by running HyperviscosityFunctorST<DpFadType>
// on the same elements. This gives the true Jacobian-vector product through
// the whole step (CAAR/DIRK + HV) "for free", via automatic differentiation,
// without needing to hand-chain HV's own (Real-only) run_JV into the Fad
// tangent. prim_advance_adj itself, however, always uses HV<Real>'s
// hand-coded run_JtV internally (not a DxFad-like type), since HV's
// diffusion operator is linear/self-adjoint and doesn't need Sacado.
TEST_CASE("prim_advance_adj")
{
  constexpr int ne = 2;
  constexpr int nlevs = NUM_PHYSICAL_LEV;

  // The random numbers generator
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");

  // Use stuff from Context, to increase similarity with actual runs
  auto& c = Context::singleton();
  F90Cleanup f90_cleanup_guard;
  auto& comm = c.create<ekat::Comm>(MPI_COMM_WORLD);

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.dp3d_thresh = 0; // don't let the limiter do anything, for now
  params.vtheta_thresh = 0; // don't let the limiter do anything, for now
  params.params_set = true;
  params.qsplit = 1;
  params.rsplit = 1;
  params.store_fwd_state = true;
  params.theta_hydrostatic_mode = false; // exercise the w_i(n0) surface fix too
  params.scale_factor = PhysicalConstants::rearth0;
  params.laplacian_rigid_factor = PhysicalConstants::rrearth0;

  // prim_advance_adj's assumptions
  params.time_step_type = TimeStepType::ttype10_imex;
  params.prescribed_wind = false;

  // HV params: enable HV, with the const-viscosity, no-sponge-layer config
  // that HV's run_JtV/run_JV currently support (see HyperviscosityFunctorImpl.hpp).
  params.hypervis_order = 2;
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
  params.dt_tracer_factor = 4;

  // Create and init hvcoord and ref_elem, needed to init the fortran interface
  auto& hvcoord = c.create<HybridVCoord>();
  auto& ref_FE  = c.create<ReferenceElement>();
  hvcoord.random_init(seed);

  auto hyai = Kokkos::create_mirror_view(hvcoord.hybrid_ai);
  auto hybi = Kokkos::create_mirror_view(hvcoord.hybrid_bi);
  auto hyam = Kokkos::create_mirror_view(hvcoord.hybrid_am);
  auto hybm = Kokkos::create_mirror_view(hvcoord.hybrid_bm);
  Kokkos::deep_copy(hyai,hvcoord.hybrid_ai);
  Kokkos::deep_copy(hybi,hvcoord.hybrid_bi);
  Kokkos::deep_copy(hyam,hvcoord.hybrid_am);
  Kokkos::deep_copy(hybm,hvcoord.hybrid_bm);
  HostViewManaged<Real[NUM_PHYSICAL_LEV]> hyam_r(""),hybm_r("");
  for (int i=0;i<NUM_PHYSICAL_LEV;++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyam_r(i) = ADValue(hyam(ilev)[ivec]);
    hybm_r(i) = ADValue(hybm(ilev)[ivec]);
  }

  std::vector<Real> dvv(NP*NP);
  std::vector<Real> mp(NP*NP);
  init_f90(ne,hyai.data(),hybi.data(),hyam_r.data(),hybm_r.data(),dvv.data(),mp.data(),hvcoord.ps0);

  ref_FE.init_mass(mp.data());
  ref_FE.init_deriv(dvv.data());

  const int num_elems = c.get<Connectivity>().get_num_local_elements();

  // Init geometry views once (same for all elements structs)
  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);

  // Pull physical geometry from f90 (gives realistic Dinv, spheremp, fcor, etc.)
  {
    auto d        = Kokkos::create_mirror_view(geo.m_d);
    auto dinv     = Kokkos::create_mirror_view(geo.m_dinv);
    auto phis     = Kokkos::create_mirror_view(geo.m_phis);
    auto gradphis = Kokkos::create_mirror_view(geo.m_gradphis);
    auto fcor     = Kokkos::create_mirror_view(geo.m_fcor);
    auto spmp     = Kokkos::create_mirror_view(geo.m_spheremp);
    auto rspmp    = Kokkos::create_mirror_view(geo.m_rspheremp);
    auto tVisc    = Kokkos::create_mirror_view(geo.m_tensorvisc);
    auto sph2c    = Kokkos::create_mirror_view(geo.m_vec_sph2cart);
    auto mdet     = Kokkos::create_mirror_view(geo.m_metdet);
    auto minv     = Kokkos::create_mirror_view(geo.m_metinv);

    // Aquaplanet: zero phis/gradphis before passing to f90, as in
    // imex_adjoint_ut.cpp, to avoid combining topography with a fully random
    // atmospheric state (risk of negative dp3d / unstable DIRK Newton
    // iterations). With gradphis==0, the w_i(n0) surface fix (mirrored below
    // in the fwd sweep, and undone by prim_advance_adj) degenerates to
    // w_i(n0,surface):=0, so the fix's own coefficients aren't numerically
    // exercised here; what *is* exercised is that prim_advance_adj discards
    // the corresponding adjoint component (since the fwd fix fully
    // overwrites, rather than accumulates into, w_i(n0,surface)).
    Kokkos::deep_copy(phis,    Real(0));
    Kokkos::deep_copy(gradphis,Real(0));

    Real*        d_ptr        = d.data();
    Real*        dinv_ptr     = dinv.data();
    const Real*  phis_ptr     = phis.data();
    const Real*  gradphis_ptr = gradphis.data();
    Real*        fcor_ptr     = fcor.data();
    Real*        spmp_ptr     = spmp.data();
    Real*        rspmp_ptr    = rspmp.data();
    Real*        tVisc_ptr    = tVisc.data();
    Real*        sph2c_ptr    = sph2c.data();
    Real*        mdet_ptr     = mdet.data();
    Real*        minv_ptr     = minv.data();

    init_geo_views_f90(d_ptr, dinv_ptr, phis_ptr, gradphis_ptr, fcor_ptr,
                       spmp_ptr, rspmp_ptr, tVisc_ptr,
                       sph2c_ptr, mdet_ptr, minv_ptr);

    Kokkos::deep_copy(geo.m_d,           d);
    Kokkos::deep_copy(geo.m_dinv,        dinv);
    Kokkos::deep_copy(geo.m_spheremp,    spmp);
    Kokkos::deep_copy(geo.m_rspheremp,   rspmp);
    Kokkos::deep_copy(geo.m_tensorvisc,  tVisc);
    Kokkos::deep_copy(geo.m_vec_sph2cart,sph2c);
    Kokkos::deep_copy(geo.m_metdet,      mdet);
    Kokkos::deep_copy(geo.m_metinv,      minv);
    Kokkos::deep_copy(geo.m_fcor,        fcor);
    Kokkos::deep_copy(geo.m_phis,        phis);
    Kokkos::deep_copy(geo.m_gradphis,    gradphis);
  }

  // Create elements for FWD/BWD integration
  auto& elems_dp = c.create<ElementsST<DpFadType>>();
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dp.m_geometry = geo;

  auto& elems_dx_caar = c.create<ElementsST<DxFadTypeCaar>>();
  elems_dx_caar.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx_caar.m_geometry = geo; // Use same views for geometry

  auto& elems_dx_dirk = c.create<ElementsST<DxFadTypeDirk>>();
  elems_dx_dirk.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx_dirk.m_geometry = geo; // Use same views for geometry

  // Create auxiliary structures
  auto& bmm = c.create<MpiBuffersManagerMap>();
  bmm[MPI_EXCHANGE]->set_connectivity(c.get_ptr<Connectivity>());

  auto& sphop_dp = c.create<SphereOperatorsST<DpFadType>>();
  auto& sphop_dx_caar = c.create<SphereOperatorsST<DxFadTypeCaar>>();
  // Needed by prim_advance_adj's internal (Real-valued) HV adjoint scratch,
  // via HyperviscosityFunctorImplST<Real>::setup(); in a real run this is the
  // same SphereOperatorsST<Real> the production CaarFunctor/HyperviscosityFunctor use.
  auto& sphop_real = c.create<SphereOperatorsST<Real>>();

  sphop_dp.setup(geo,ref_FE);
  sphop_dx_caar.setup(geo,ref_FE);
  sphop_real.setup(geo,ref_FE);

  int nm1 = 0;
  int n0  = 1;
  int np1 = 2;
  // Create and init Caar/Limiter/Dirk functors
  auto& caar_dp = c.create<CaarFunctorST<DpFadType>>();
  auto& caar_dx = c.create<CaarFunctorST<DxFadTypeCaar>>();
  caar_dp.setup(elems_dp,ref_FE,hvcoord,sphop_dp);
  caar_dx.setup(elems_dx_caar,ref_FE,hvcoord,sphop_dx_caar);
  caar_dp.init_boundary_exchanges(bmm[MPI_EXCHANGE]);
  caar_dx.init_boundary_exchanges(bmm[MPI_EXCHANGE]);
  std::any_cast<CaarFunctorImplST<DpFadType>&>(caar_dp.impl()).m_run_limiter = false;
  std::any_cast<CaarFunctorImplST<DxFadTypeCaar>&>(caar_dx.impl()).m_run_limiter = false;

  auto& limiter_dp = c.create<LimiterFunctorST<DpFadType>>(elems_dp,hvcoord,params);
  auto& limiter_dx = c.create<LimiterFunctorST<DxFadTypeCaar>>(elems_dx_caar,hvcoord,params);
  limiter_dp.m_verbose = false;
  limiter_dx.m_verbose = false;

  auto& dirk_dp = c.create<DirkFunctorST<DpFadType>>(elems_dp.num_elems());
  auto& dirk_dx = c.create<DirkFunctorST<DxFadTypeDirk>>(elems_dx_dirk.num_elems());

  // HV<DpFadType>: used only in this test, to get the true (AD) tangent
  // through HV "for free". prim_advance_adj itself always uses its own,
  // internal HV<Real> for the actual adjoint (see prim_advance_adj.cpp).
  auto& hv_dp = c.create<HyperviscosityFunctorST<DpFadType>>(num_elems,params);
  hv_dp.setup(geo,elems_dp.m_state,elems_dp.m_derived);

  // Setup scratch buffers
  FunctorsBuffersManager fbm;
  fbm.request_size(caar_dp.requested_buffer_size());
  fbm.request_size(caar_dx.requested_buffer_size());
  fbm.request_size(dirk_dp.requested_buffer_size());
  fbm.request_size(dirk_dx.requested_buffer_size());
  fbm.request_size(limiter_dp.requested_buffer_size());
  fbm.request_size(limiter_dx.requested_buffer_size());
  fbm.request_size(hv_dp.requested_buffer_size());

  fbm.allocate();

  caar_dp.init_buffers(fbm);
  caar_dx.init_buffers(fbm);
  dirk_dp.init_buffers(fbm);
  dirk_dx.init_buffers(fbm);
  limiter_dp.init_buffers(fbm);
  limiter_dx.init_buffers(fbm);
  hv_dp.init_buffers(fbm);
  hv_dp.init_boundary_exchanges();

  // Scalar params
  const double dt = 10;
  const double eta_ave_w = 1.0/params.dt_tracer_factor;

  // Create the Imex tape. Capacity is 12: the usual 11 ttype10 CAAR/DIRK
  // checkpoints (y0,u1,y1,...,u5,y5), plus one more for the state right
  // after HV runs (see prim_advance_exp.cpp's store_fwd_state block).
  c.any_map().try_emplace("imex_tape",std::in_place_type<Tape<StateSnapshot>>,12,num_elems);

  // Initial state
  StateSnapshot state_t0(num_elems);
  state_t0.randomize(seed,1e5,1e3,hvcoord.hybrid_ai0,geo.m_phis);

  // Run FWD sweep
  elems_dp.m_state.import_snapshot(state_t0,n0);
  elems_dp.m_state.randomize_derivs(seed,n0);
  auto du0 = elems_dp.m_state.take_deriv_snapshot(n0,0);

  // Mirror the w_i(n0) surface fix at the very top of prim_advance_exp (see
  // prim_advance_exp.cpp), applied here directly to elems_dp (DpFadType), so
  // its Fad-tracked tangent flows through the rest of the computation
  // exactly like a real run. This makes du0.w_i(n0,surface) (whatever random
  // value randomize_derivs gave it) irrelevant to the true composite (it is
  // fully superseded by this fix before ttype10 ever reads it) -- which is
  // exactly what prim_advance_adj assumes when its own copy of this fix's
  // adjoint discards the corresponding component. Skipping this step here
  // (i.e. only running it in the adjoint, never in the fwd sweep) would make
  // du0.w_i(n0,surface) a genuinely free input on the fwd side only, and the
  // dot-product identity below would fail.
  {
    auto w_i = elems_dp.m_state.m_w_i;
    auto v   = elems_dp.m_state.m_v;
    auto gradphis = geo.m_gradphis;
    constexpr auto LAST_LEV_P = ColInfo<NUM_INTERFACE_LEV>::LastPack;
    constexpr auto LAST_LEV   = ColInfo<NUM_PHYSICAL_LEV>::LastPack;
    constexpr auto LAST_INTERFACE_VEC_IDX = ColInfo<NUM_INTERFACE_LEV>::LastPackEnd;
    constexpr auto LAST_MIDPOINT_VEC_IDX  = ColInfo<NUM_PHYSICAL_LEV>::LastPackEnd;
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,NP*NP*num_elems),
                         KOKKOS_LAMBDA(const int idx) {
      const int ie  = idx / (NP*NP);
      const int igp = (idx / NP) % NP;
      const int jgp = idx % NP;
      w_i(ie,n0,igp,jgp,LAST_LEV_P)[LAST_INTERFACE_VEC_IDX] =
                      (v(ie,n0,0,igp,jgp,LAST_LEV)[LAST_MIDPOINT_VEC_IDX]*gradphis(ie,0,igp,jgp) +
                       v(ie,n0,1,igp,jgp,LAST_LEV)[LAST_MIDPOINT_VEC_IDX]*gradphis(ie,1,igp,jgp))/PhysicalConstants::g;
    });
    Kokkos::fence();
  }

  printf(" -> Run forward problem (ttype10 CAAR/DIRK stages)...\n");
  ttype10_imex_timestep<DpFadType>(nm1,n0,np1,dt,eta_ave_w);
  printf(" -> Run forward problem (ttype10 CAAR/DIRK stages)...done!\n");

  printf(" -> Run forward problem (HV)...\n");
  hv_dp.run(np1,dt,eta_ave_w);
  printf(" -> Run forward problem (HV)...done!\n");

  // Tape the post-HV state (mirrors the store_fwd_state block prim_advance_exp.cpp
  // runs after HV); prim_advance_adj needs this, together with the pre-HV
  // snapshot ttype10_imex_timestep already taped, to linearize HV's step.
  {
    using tape_t = Tape<StateSnapshot>;
    auto& tape = std::any_cast<tape_t&>(c.any_map().at("imex_tape"));
    tape.shift_fwd();
    elems_dp.m_state.take_snapshot(tape.curr(),np1,false);
  }

  // The true tangent through the whole step (CAAR/DIRK + HV)
  auto duN = elems_dp.m_state.take_deriv_snapshot(np1,0);

  // Run BWD pass
  StateSnapshot lambda(num_elems);
  lambda.randomize(seed,1.0,1.0/100,0.0);
  auto lambdaN = lambda.clone(true);
  printf(" -> Run adjoint problem (prim_advance_adj)...\n");
  prim_advance_adj(dt,lambda);
  printf(" -> Run adjoint problem (prim_advance_adj)...done!\n");
  auto lambda0 = lambda.clone(true);

  // Compare du0*lambda0 with duN*lambdaN
  auto v_dot0 = dot(ekat::scalarize(du0.v),ekat::scalarize(lambda0.v),nlevs);
  auto v_dotN = dot(ekat::scalarize(duN.v),ekat::scalarize(lambdaN.v),nlevs);

  auto vth_dot0 = dot(ekat::scalarize(du0.vtheta_dp),ekat::scalarize(lambda0.vtheta_dp),nlevs);
  auto vth_dotN = dot(ekat::scalarize(duN.vtheta_dp),ekat::scalarize(lambdaN.vtheta_dp),nlevs);

  auto dp_dot0 = dot(ekat::scalarize(du0.dp3d),ekat::scalarize(lambda0.dp3d),nlevs);
  auto dp_dotN = dot(ekat::scalarize(duN.dp3d),ekat::scalarize(lambdaN.dp3d),nlevs);

  auto w_dot0 = dot(ekat::scalarize(du0.w_i),ekat::scalarize(lambda0.w_i),nlevs+1);
  auto w_dotN = dot(ekat::scalarize(duN.w_i),ekat::scalarize(lambdaN.w_i),nlevs+1);

  auto phi_dot0 = dot(ekat::scalarize(du0.phinh_i),ekat::scalarize(lambda0.phinh_i),nlevs+1);
  auto phi_dotN = dot(ekat::scalarize(duN.phinh_i),ekat::scalarize(lambdaN.phinh_i),nlevs+1);

  constexpr auto tol = std::numeric_limits<double>::epsilon()*1e4;
  {
    using namespace Catch::Matchers;

    auto full_dot0 = v_dot0 + vth_dot0 + dp_dot0 + w_dot0 + phi_dot0;
    auto full_dotN = v_dotN + vth_dotN + dp_dotN + w_dotN + phi_dotN;
    CHECK_THAT (full_dot0, WithinRel(full_dotN,tol));

    if (comm.am_i_root())
      std::cout << std::setprecision(15)
                << "   <du0, lambda0> = " << full_dot0
                << ",  <duN, lambdaN> = " << full_dotN << "\n";
  }
}

} // namespace Homme
