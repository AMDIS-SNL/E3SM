#include <catch2/catch.hpp>

#include "Types.hpp"
#include "Tape.hpp"
#include "Context.hpp"
#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"
#include "DirkFunctor.hpp"
#include "DirkFunctorImpl.hpp"
#include "LimiterFunctor.hpp"
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

template<typename ViewT>
double dot (ViewT v1, ViewT v2, int last_dim = -1) {
  auto v1h = Kokkos::create_mirror_view(v1);
  Kokkos::deep_copy(v1h,v1);
  auto v2h = Kokkos::create_mirror_view(v2);
  Kokkos::deep_copy(v2h,v2);

  EKAT_REQUIRE_MSG (v1.rank==v2.rank, "Error! Views have different rank.\n");
  EKAT_REQUIRE_MSG (v1.rank==3 or v1.rank==4 or v1.rank==5, "Error! Unsupported rank.\n");
  double prod = 0;
  if constexpr(v1.rank==3) {
    auto last_ext = last_dim > 0 ? last_dim : v1.extent_int(2);
    for (int i=0; i<v1.extent_int(0); ++i)
      for (int j=0; j<v1.extent_int(1); ++j)
        for (int k=0; k<last_ext; ++k)
          prod += v1h(i,j,k)*v2h(i,j,k);
  } else if constexpr(v1.rank==4) {
    auto last_ext = last_dim > 0 ? last_dim : v1.extent_int(3);
    for (int i=0; i<v1.extent_int(0); ++i)
      for (int j=0; j<v1.extent_int(1); ++j)
        for (int k=0; k<v1.extent_int(2); ++k)
          for (int l=0; l<last_ext; ++l)
            prod += v1h(i,j,k,l)*v2h(i,j,k,l);
  } else if constexpr(v1.rank==5) {
    auto last_ext = last_dim > 0 ? last_dim : v1.extent_int(4);
    for (int i=0; i<v1.extent_int(0); ++i)
      for (int j=0; j<v1.extent_int(1); ++j)
        for (int k=0; k<v1.extent_int(2); ++k)
          for (int l=0; l<v1.extent_int(3); ++l)
            for (int m=0; m<last_ext; ++m)
              prod += v1h(i,j,k,l,m)*v2h(i,j,k,l,m);
  }

  return prod;
}

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
void initialize_dp3d_from_ps_c ();
}

template<typename ST = ScalarValue>
void fake_imex_forward (const int nm1, const int n0, const int np1,
                        const Real dt_dyn,
                        const Real eta_ave_w)
{
  GPTLstart("fake_imex_forward");

  // The context
  auto& c = Context::singleton();
  SimulationParams& params = c.get<SimulationParams>();

  // Get elements, hvcoord, and functors
  auto& elements = c.get<ElementsST<ST>>();
  auto& hvcoord  = c.get<HybridVCoord>();
  auto& dirk_base  = c.get<DirkFunctorST<ST>>();
  auto& caar_base  = c.get<CaarFunctorST<ST>>();
  auto& dirk       = std::any_cast<DirkFunctorImplST<ST>&>(dirk_base.impl());
  auto& caar       = std::any_cast<CaarFunctorImplST<ST>&>(caar_base.impl());

  const int nelems = elements.num_elems();

  auto save = [&](int tl) {
    if (not params.store_fwd_state)
      return;
    using tape_t = Tape<StateSnapshot>;

    auto& tape = std::any_cast<tape_t&>(c.any_map().at("imex_tape"));
    tape.shift_fwd();
    auto& snap = tape.curr();
    elements.m_state.take_snapshot(snap,tl,false);
  };

  // ===================== IMEX STAGES ===================== //
  Real dt;

  // Last stage DIRK factors
  auto a1 = 0.24;
  auto a2 = 0.34;
  auto a3 = 1-(a1+a2);

  // Save initial state y0 (needed by adjoint for stage 5 DIRK background)
  save(n0);

  // Stage 1
  dt = dt_dyn/4.0;

  auto stage1_data = RKStageData(n0, n0, nm1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  caar.run(stage1_data);
  save(nm1);
  dirk.run(nm1, 0.0, n0, 0.0, nm1, dt, elements, hvcoord);
  save(nm1);

  // Stage 2
  dt = dt_dyn/2.0;

  auto stage2_data = RKStageData(n0, nm1, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  caar.run(stage2_data);
  save(np1);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elements, hvcoord);
  save(np1);

  // Stage 3
  dt = dt_dyn/4.0;

  auto stage3_data = RKStageData(n0, np1, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  caar.run(stage3_data);
  save(np1);
  dirk.run(nm1, a1*dt, n0, a2*dt, np1, a3*dt, elements, hvcoord);
  save(np1);

  GPTLstop("fake_imex_forward");
}

void fake_imex_adjoint(const Real dt_dyn,
                       const Real eta_ave_w,
                       StateSnapshot& adj_state)
{
  GPTLstart("fake_imex_adjoint");
  using const_tape_t = const Tape<StateSnapshot>;

  const auto& c = Context::singleton();
  SimulationParams& params = c.get<SimulationParams>();

  // Get elements, hvcoord, and functors
  auto& elems_dirk = c.get<ElementsST<DxFadTypeDirk>>();
  auto& elems_caar = c.get<ElementsST<DxFadTypeCaar>>();
  auto& state_dirk = elems_dirk.m_state;
  auto& state_caar = elems_caar.m_state;
  auto& hvcoord    = c.get<HybridVCoord>();
  auto& dirk_base  = c.get<DirkFunctorST<DxFadTypeDirk>>();
  auto& caar_base  = c.get<CaarFunctorST<DxFadTypeCaar>>();
  auto& dirk       = std::any_cast<DirkFunctorImplST<DxFadTypeDirk>&>(dirk_base.impl());
  auto& caar       = std::any_cast<CaarFunctorImplST<DxFadTypeCaar>&>(caar_base.impl());
  auto& tape       = std::any_cast<const_tape_t&>(c.any_map().at("imex_tape"));
  auto& geo        = c.get<ElementsGeometry>();

  auto rspheremp = geo.m_rspheremp;

  int nelem = adj_state.num_elems;
  Real dt;

  // For each functor, load fwd state we had right before
  // running it, run functor, then compute JtV (with V=adj_state)
  // NOTATION:
  //
  // State:
  //  - u_i: state after explicit CAAR stage
  //  - y_i: state after implicit DIRK stage
  // where y_0 is the state at the beginning of prim_advance_exp,
  // and y_5 is the state at the end (after 5th DIRK stage)
  //
  // Adjoint state:
  //  - lambda_i: deriv w.r.t. u_i
  //  - mu_i: deriv w.r.t. y_i
  // Hence, lambda is the adjoint var between a CAAR and DIRK stage,
  // while mu is the adjoint var between DIRK and CAAR stages.
  // So mu5 is the adj var at entry, while mu0 is the adj var at exit
  StateSnapshot lambda (nelem);
  StateSnapshot mu = adj_state;

  // These are all alias of lambda and mu, but they make the code underneath easier to follow
  auto mu0 = mu, mu1 = mu, mu2 = mu, mu3 = mu;
  auto lambda1 = lambda, lambda2 = lambda, lambda3 = lambda;

  int nm1 = 0;
  int n0  = 1;
  int np1 = 2;

  // TODO: this must be created ONCE, not every time
  auto be = create_adj_bex(lambda);

  const auto& y0 = tape.at(0);
  const auto& u1 = tape.at(1);
  const auto& y1 = tape.at(2);
  const auto& u2 = tape.at(3);
  const auto& y2 = tape.at(4);
  const auto& u3 = tape.at(5);
  const auto& y3 = tape.at(6);

  // Last stage DIRK factors
  auto a1 = 0.24;
  auto a2 = 0.34;
  auto a3 = 1-(a1+a2);

  // Departure point in CAAR is the same for all stages
  state_caar.import_snapshot(y0,nm1);

  // First, derivatives of DIRK in last stage w.r.t y0 and y1
  StateSnapshot dDdy0_mu3(nelem), dDdy1_mu3(nelem);
  dt = dt_dyn/4.0;  // stage 3 dt — must be set before these DIRK runs

  state_dirk.import_snapshot(u3, np1);
  state_dirk.import_snapshot(y0, n0);
  state_dirk.import_snapshot(y1, nm1);

  // dD3/dy0^T * mu3: seed FAD at n0 (y0 slot), run DIRK3, extract J^T*mu3.
  // run_JtV uses m_dx_tl (set by init_J) to decide whether to add the identity
  // block for v/vtheta/dp: only added when dx_tl==np1 (primary input), not for
  // background slots where d(v_np1)/d(v_bg)=0.
  dirk.init_J(n0,state_dirk);
  dirk.run(nm1, a1*dt, n0, a2*dt, np1, a3*dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu3,dDdy0_mu3);

  state_dirk.import_snapshot(u3,np1);
  dirk.init_J(nm1,state_dirk);
  dirk.run(nm1, a1*dt, n0, a2*dt, np1, a3*dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu3,dDdy1_mu3);

  // Then all the stages in bwd order. Keep also the sum of lambdas (the contribution from y0
  // in all CAAR steps)
  StateSnapshot lambda_sum(nelem);
  lambda_sum.zero();

  // Stage 3
  dt = dt_dyn/4.0;
  auto stage3_data = RKStageData(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);

  state_dirk.import_snapshot(u3,np1);
  dirk.init_J(np1,state_dirk);
  dirk.run(nm1, a1*dt, n0, a2*dt, np1, a3*dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu3,lambda3);

  caar.run_JtV_surf_bc(stage3_data,lambda3,lambda3);
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda3, geo.m_spheremp, stage3_data.scale3);
  state_caar.import_snapshot(y2,n0);
  caar.init_J(stage3_data);
  caar.run_pre_exchange(stage3_data);
  caar.run_JtV(stage3_data,lambda3,mu2);

  // Stage 2
  dt = dt_dyn/2.0;
  auto stage2_data = RKStageData(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);

  state_dirk.import_snapshot(u2,np1);
  dirk.init_J(np1,state_dirk);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu2,lambda2);

  caar.run_JtV_surf_bc(stage2_data,lambda2,lambda2);
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda2, geo.m_spheremp, stage2_data.scale3);
  state_caar.import_snapshot(y1,n0);
  caar.init_J(stage2_data);
  caar.run_pre_exchange(stage2_data);
  caar.run_JtV(stage2_data,lambda2,mu1);
  mu1.add(dDdy1_mu3);

  // Stage 1
  dt = dt_dyn/4.0;
  auto stage1_data = RKStageData(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);

  state_dirk.import_snapshot(u1, np1);
  dirk.init_J(np1, state_dirk);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1, elems_dirk.m_state, mu1, lambda1);

  caar.run_JtV_surf_bc(stage1_data,lambda1,lambda1);
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda1, geo.m_spheremp, stage1_data.scale3);
  state_caar.import_snapshot(y0,n0);
  caar.init_J(stage1_data);
  caar.run_pre_exchange(stage1_data);
  caar.run_JtV(stage1_data,lambda1,mu0);
  mu0.add(dDdy0_mu3);

  // Add the contributions corresponding to CAAR's departure point (which is always y0)
  mu0.add(lambda_sum);

  GPTLstop("fake_imex_adjoint");
}

TEST_CASE("fake_imex_adjoint")
{
  constexpr int ne = 2;
  constexpr int nlevs = NUM_PHYSICAL_LEV;
  const auto A = Kokkos::ALL();

  // The random numbers generator
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  // Use stuff from Context, to increase similarity with actual runs
  auto& c = Context::singleton();
  auto& comm = c.create<ekat::Comm>(MPI_COMM_WORLD);

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.dp3d_thresh = 0; // don't let the limiter do anything, for now
  params.vtheta_thresh = 0; // don't let the limiter do anything, for now
  params.params_set = true;
  params.qsplit = 1;
  params.rsplit = 1;
  params.store_fwd_state = true;
  params.theta_hydrostatic_mode = false;
  params.scale_factor = PhysicalConstants::rearth0;
  params.laplacian_rigid_factor = PhysicalConstants::rrearth0;

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
  const auto max_pressure = 1000.0 + hvcoord.ps0; // This ensures max_p > ps0

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

    // Aquaplanet: zero phis/gradphis before passing to f90
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

  sphop_dp.setup(geo,ref_FE);
  sphop_dx_caar.setup(geo,ref_FE);

  auto conn  = c.get_ptr<Connectivity>();
  auto  bm   = bmm[MPI_EXCHANGE];

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

  // Setup scratch buffers
  FunctorsBuffersManager fbm;
  fbm.request_size(caar_dp.requested_buffer_size());
  fbm.request_size(caar_dx.requested_buffer_size());
  fbm.request_size(dirk_dp.requested_buffer_size());
  fbm.request_size(dirk_dx.requested_buffer_size());
  fbm.request_size(limiter_dp.requested_buffer_size());
  fbm.request_size(limiter_dx.requested_buffer_size());

  fbm.allocate();

  caar_dp.init_buffers(fbm);
  caar_dx.init_buffers(fbm);
  dirk_dp.init_buffers(fbm);
  dirk_dx.init_buffers(fbm);
  limiter_dp.init_buffers(fbm);
  limiter_dx.init_buffers(fbm);

  // Scalar params
  double dt = 10;
  double eta_ave_w = 0.25;

  // Create the Imex tape
  c.any_map().try_emplace("imex_tape",std::in_place_type<Tape<StateSnapshot>>,11,num_elems);

  // Initial state
  StateSnapshot state_t0(num_elems);
  state_t0.randomize(seed,1e5,1e3,hvcoord.hybrid_ai0,geo.m_phis);

  // Run FWD sweep
  elems_dp.m_state.import_snapshot(state_t0,n0);
  elems_dp.m_state.randomize_derivs(seed,n0);
  auto du0 = elems_dp.m_state.take_deriv_snapshot(n0,0);
  printf(" -> Run forward problem...\n");
  fake_imex_forward<DpFadType>(nm1,n0,np1,dt,eta_ave_w);
  printf(" -> Run forward problem...done!\n");
  auto duN = elems_dp.m_state.take_deriv_snapshot(np1,0);

  // Run BWD pass
  StateSnapshot lambda(num_elems);
  lambda.randomize(seed,1.0,1.0/100,0.0);
  auto lambdaN = lambda.clone(true);
  printf(" -> Run adjoint problem...\n");
  fake_imex_adjoint(dt,eta_ave_w,lambda);
  printf(" -> Run adjoint problem...done!\n");
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

  cleanup_f90();
  c.finalize_singleton();
}

TEST_CASE("ttype10_imex_adjoint")
{
  constexpr int ne = 2;
  constexpr int nlevs = NUM_PHYSICAL_LEV;
  const auto A = Kokkos::ALL();

  // The random numbers generator
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  // Use stuff from Context, to increase similarity with actual runs
  auto& c = Context::singleton();
  auto& comm = c.create<ekat::Comm>(MPI_COMM_WORLD);

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.dp3d_thresh = 0; // don't let the limiter do anything, for now
  params.vtheta_thresh = 0; // don't let the limiter do anything, for now
  params.params_set = true;
  params.qsplit = 1;
  params.rsplit = 1;
  params.store_fwd_state = true;
  params.theta_hydrostatic_mode = false;
  params.scale_factor = PhysicalConstants::rearth0;
  params.laplacian_rigid_factor = PhysicalConstants::rrearth0;

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
  const auto max_pressure = 1000.0 + hvcoord.ps0; // This ensures max_p > ps0

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

    // Aquaplanet: zero phis/gradphis before passing to f90
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

  sphop_dp.setup(geo,ref_FE);
  sphop_dx_caar.setup(geo,ref_FE);

  auto conn  = c.get_ptr<Connectivity>();
  auto  bm   = bmm[MPI_EXCHANGE];

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

  // Setup scratch buffers
  FunctorsBuffersManager fbm;
  fbm.request_size(caar_dp.requested_buffer_size());
  fbm.request_size(caar_dx.requested_buffer_size());
  fbm.request_size(dirk_dp.requested_buffer_size());
  fbm.request_size(dirk_dx.requested_buffer_size());
  fbm.request_size(limiter_dp.requested_buffer_size());
  fbm.request_size(limiter_dx.requested_buffer_size());

  fbm.allocate();

  caar_dp.init_buffers(fbm);
  caar_dx.init_buffers(fbm);
  dirk_dp.init_buffers(fbm);
  dirk_dx.init_buffers(fbm);
  limiter_dp.init_buffers(fbm);
  limiter_dx.init_buffers(fbm);

  // Scalar params
  double dt = 10;
  double eta_ave_w = 0.25;

  // Create the Imex tape
  c.any_map().try_emplace("imex_tape",std::in_place_type<Tape<StateSnapshot>>,11,num_elems);

  // Initial state
  StateSnapshot state_t0(num_elems);
  state_t0.randomize(seed,1e5,1e3,hvcoord.hybrid_ai0,geo.m_phis);

  // Run FWD sweep
  elems_dp.m_state.import_snapshot(state_t0,n0);
  elems_dp.m_state.randomize_derivs(seed,n0);
  auto du0 = elems_dp.m_state.take_deriv_snapshot(n0,0);
  printf(" -> Run forward problem...\n");
  ttype10_imex_timestep<DpFadType>(nm1,n0,np1,dt,eta_ave_w);
  printf(" -> Run forward problem...done!\n");
  auto duN = elems_dp.m_state.take_deriv_snapshot(np1,0);

  // Run BWD pass
  StateSnapshot lambda(num_elems);
  lambda.randomize(seed,1.0,1.0/100,0.0);
  auto lambdaN = lambda.clone(true);
  printf(" -> Run adjoint problem...\n");
  ttype10_imex_adjoint(dt,eta_ave_w,lambda);
  printf(" -> Run adjoint problem...done!\n");
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

  cleanup_f90();
  c.finalize_singleton();
}

} // namespace Homme
