#include <catch2/catch.hpp>

#include "Types.hpp"
#include "Tape.hpp"
#include "Context.hpp"
#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"
#include "DirkFunctor.hpp"
#include "DirkFunctorImpl.hpp"
#include "ElementsDerivedState.hpp"
#include "EulerStepFunctor.hpp"
#include "EulerStepFunctorImpl.hpp"
#include "LimiterFunctor.hpp"
#include "HyperviscosityFunctor.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "PhysicalConstants.hpp"
#include "TimeLevel.hpp"
#include "Tracers.hpp"
#include "prim_advance_exp.hpp"
#include "prim_advance_adj.hpp"
#include "prim_step_adj.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_string_utils.hpp>
#include <ekat_comm.hpp>
#include <ekat_pack_kokkos.hpp>

#include <iomanip>
#include <random>
#include <string>
#include <vector>

#ifdef HOMMEXX_ENABLE_FAD_TYPES

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

// ---- Fad <-> Real helpers (same as in euler_step_adjoint_ut.cpp). All work via a
// host round trip. fad_view and real_view must have the same logical shape. ----

// fad_view's .val() := real_view. Since assigning a plain scalar to a Fad
// resets its derivative, call this *before* set_deriv0 on the same view.
template<typename FadView, typename RealView>
void set_val (const FadView& fad_view, const RealView& real_view) {
  auto fh = Kokkos::create_mirror(fad_view); Kokkos::deep_copy(fh, fad_view);
  auto rh = Kokkos::create_mirror(real_view); Kokkos::deep_copy(rh, real_view);
  auto fs = ekat::scalarize(fh);
  auto rs = ekat::scalarize(rh);
  EKAT_REQUIRE_MSG(fs.span()==rs.span(), "[set_val] Error! Mismatched spans.\n");
  for (size_t i=0; i<fs.span(); ++i) fs.data()[i] = rs.data()[i];
  Kokkos::deep_copy(fad_view, fh);
}

// fad_view's .fastAccessDx(0) := real_view; .val() untouched.
template<typename FadView, typename RealView>
void set_deriv0 (const FadView& fad_view, const RealView& real_view) {
  auto fh = Kokkos::create_mirror(fad_view); Kokkos::deep_copy(fh, fad_view);
  auto rh = Kokkos::create_mirror(real_view); Kokkos::deep_copy(rh, real_view);
  auto fs = ekat::scalarize(fh);
  auto rs = ekat::scalarize(rh);
  EKAT_REQUIRE_MSG(fs.span()==rs.span(), "[set_deriv0] Error! Mismatched spans.\n");
  for (size_t i=0; i<fs.span(); ++i) fs.data()[i].fastAccessDx(0) = rs.data()[i];
  Kokkos::deep_copy(fad_view, fh);
}

// real_view (a host view) := fad_view's .fastAccessDx(0)
template<typename RealView, typename FadView>
void get_deriv0 (const RealView& real_view, const FadView& fad_view) {
  auto fh = Kokkos::create_mirror(fad_view); Kokkos::deep_copy(fh, fad_view);
  auto fs = ekat::scalarize(fh);
  auto rs = ekat::scalarize(real_view);
  EKAT_REQUIRE_MSG(fs.span()==rs.span(), "[get_deriv0] Error! Mismatched spans.\n");
  for (size_t i=0; i<fs.span(); ++i) rs.data()[i] = fs.data()[i].fastAccessDx(0);
}

// sum_{ie,iq<qsize,i,j,lev} a(ie,iq,i,j,lev) * b(ie,slot,iq,i,j,lev), where a is
// qdp at a single time level, and b is the full qdp view. The latter has room
// for QSIZE_D tracers, but only the first qsize are meaningful.
template<typename View1, typename View2>
double dot_qdp (const View1& a, const View2& b, const int slot, const int qsize) {
  auto ah = Kokkos::create_mirror(a); Kokkos::deep_copy(ah,a);
  auto bh = Kokkos::create_mirror(b); Kokkos::deep_copy(bh,b);
  const int ne = ah.extent_int(0);
  double s = 0;
  for (int ie=0; ie<ne; ++ie)
    for (int iq=0; iq<qsize; ++iq)
      for (int i=0; i<NP; ++i)
        for (int j=0; j<NP; ++j)
          for (int lev=0; lev<NUM_PHYSICAL_LEV; ++lev) {
            const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
            s += ah(ie,iq,i,j,vpi)[vsi] * bh(ie,slot,iq,i,j,vpi)[vsi];
          }
  return s;
}

// The dot product over the whole (physical) dynamics state
double dot_state (const StateSnapshot& x, const StateSnapshot& y) {
  constexpr int nlevs = NUM_PHYSICAL_LEV;
  return dot(ekat::scalarize(x.v),        ekat::scalarize(y.v),        nlevs)
       + dot(ekat::scalarize(x.vtheta_dp),ekat::scalarize(y.vtheta_dp),nlevs)
       + dot(ekat::scalarize(x.dp3d),     ekat::scalarize(y.dp3d),     nlevs)
       + dot(ekat::scalarize(x.w_i),      ekat::scalarize(y.w_i),      nlevs+1)
       + dot(ekat::scalarize(x.phinh_i),  ekat::scalarize(y.phinh_i),  nlevs+1);
}

} // anonymous namespace

// Full adjoint test for prim_step (the Eulerian one, see prim_step_adj.hpp),
// via the same dot-product pattern used for prim_advance_adj in
// prim_advance_sacado_ut.cpp and euler_step_adj in euler_step_adjoint_ut.cpp:
//   <J*du0, lambdaN> == <du0, J^T*lambdaN>
// The tangent J*du0 is obtained by running a Fad-typed (DpFadType) replica of
// prim_step with a random tangent du0 on the initial (dynamics state, qdp).
// The adjoint J^T*lambdaN is computed by prim_step_adj, which recomputes the
// forward trajectory on the Context's Real-typed objects (i.e., the ones an
// actual run would use), from a PrimStepCheckpoint of the state at the start.
//
// The two seeds lambdaN are tested separately (state-only and qdp-only). The
// former checks the chaining of K>1 calls to prim_advance_adj (and
// recomputation of their tapes); the latter also exercises the dynamics->tracers
// coupling (vn0 and dp3d(n0)), the 3 chained euler_step adjoints (including
// the qlim chaining and the biharmonic term), and the qdp_time_avg/
// precompute_divdp adjoints.
TEST_CASE("prim_step_adjoint")
{
  constexpr int ne = 2;
  constexpr int qsize = 2;

  // The random numbers generator
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  using RPDF = std::uniform_real_distribution<Real>;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);

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
  params.store_fwd_state = false; // prim_step_adj takes care of turning it on when needed
  params.theta_hydrostatic_mode = false;
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

  // Tracers params: Eulerian transport, and the euler_step_adj's assumptions.
  // nu_q is huge, since on this tiny grid (and with dp0 tiny too) it takes a
  // huge viscosity for the biharmonic term to be non negligible. We want it to be
  // because, this way, we exercise the adjoint of the last euler_step
  // (rhs_multiplier=2), which is the only one that includes it.
  params.transport_alg = 0;
  params.qsize = qsize;
  params.limiter_option = 9;
  params.nu_q = 1e20;

  // >1 to exercise the chaining of several prim_advance_exp (and their
  // recomputation in the bwd sweep) inside a single prim_step
  params.dt_tracer_factor = 2;

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
    // prim_advance_sacado_ut.cpp (see there for the reasons). With
    // gradphis==0, the w_i(n0) surface fix degenerates to w_i(n0,surface):=0.
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

  // ===== Elements, tracers and derived state, for the 4 types involved =====
  //  - Real: what prim_step_adj replays the fwd trajectory with (as in a real run)
  //  - DpFadType: the fwd tangent J*du0 (ground truth)
  //  - DxFadType[Caar|Dirk]: the Jacobians prim_advance_adj is made of
  // As in production (see init_elements_impl in cxx_f90_interface_theta.cpp),
  // the derived state registered in the Context is the one inside Elements,
  // so that functors that grab it from the Context see the same views.
  auto& elems_r = c.create<ElementsST<Real>>();
  elems_r.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_r.m_geometry = geo;
  c.create_ref<ElementsDerivedStateST<Real>>(elems_r.m_derived);
  auto& tracers_r = c.create<TracersST<Real>>();
  tracers_r.init(num_elems,qsize);

  auto& elems_dp = c.create<ElementsST<DpFadType>>();
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dp.m_geometry = geo;
  c.create_ref<ElementsDerivedStateST<DpFadType>>(elems_dp.m_derived);
  auto& tracers_dp = c.create<TracersST<DpFadType>>();
  tracers_dp.init(num_elems,qsize);

  auto& elems_dx_caar = c.create<ElementsST<DxFadTypeCaar>>();
  elems_dx_caar.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx_caar.m_geometry = geo; // Use same views for geometry

  auto& elems_dx_dirk = c.create<ElementsST<DxFadTypeDirk>>();
  elems_dx_dirk.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx_dirk.m_geometry = geo; // Use same views for geometry

  // Create auxiliary structures
  auto& bmm = c.create<MpiBuffersManagerMap>();
  auto conn_ptr = c.get_ptr<Connectivity>();
  bmm[MPI_EXCHANGE]->set_connectivity(conn_ptr);
  bmm[MPI_EXCHANGE_MIN_MAX]->set_connectivity(conn_ptr); // Needed by the euler step

  auto& sphop_r = c.create<SphereOperatorsST<Real>>();
  auto& sphop_dp = c.create<SphereOperatorsST<DpFadType>>();
  auto& sphop_dx_caar = c.create<SphereOperatorsST<DxFadTypeCaar>>();

  sphop_r.setup(geo,ref_FE);
  sphop_dp.setup(geo,ref_FE);
  sphop_dx_caar.setup(geo,ref_FE);

  // Create and init Caar/Limiter/Dirk/HV functors
  auto& caar_r  = c.create<CaarFunctorST<Real>>();
  auto& caar_dp = c.create<CaarFunctorST<DpFadType>>();
  auto& caar_dx = c.create<CaarFunctorST<DxFadTypeCaar>>();
  caar_r.setup(elems_r,ref_FE,hvcoord,sphop_r);
  caar_dp.setup(elems_dp,ref_FE,hvcoord,sphop_dp);
  caar_dx.setup(elems_dx_caar,ref_FE,hvcoord,sphop_dx_caar);
  caar_r.init_boundary_exchanges(bmm[MPI_EXCHANGE]);
  caar_dp.init_boundary_exchanges(bmm[MPI_EXCHANGE]);
  caar_dx.init_boundary_exchanges(bmm[MPI_EXCHANGE]);
  std::any_cast<CaarFunctorImplST<Real>&>(caar_r.impl()).m_run_limiter = false;
  std::any_cast<CaarFunctorImplST<DpFadType>&>(caar_dp.impl()).m_run_limiter = false;
  std::any_cast<CaarFunctorImplST<DxFadTypeCaar>&>(caar_dx.impl()).m_run_limiter = false;

  auto& limiter_r  = c.create<LimiterFunctorST<Real>>(elems_r,hvcoord,params);
  auto& limiter_dp = c.create<LimiterFunctorST<DpFadType>>(elems_dp,hvcoord,params);
  auto& limiter_dx = c.create<LimiterFunctorST<DxFadTypeCaar>>(elems_dx_caar,hvcoord,params);
  limiter_r.m_verbose = false;
  limiter_dp.m_verbose = false;
  limiter_dx.m_verbose = false;

  auto& dirk_r  = c.create<DirkFunctorST<Real>>(elems_r.num_elems());
  auto& dirk_dp = c.create<DirkFunctorST<DpFadType>>(elems_dp.num_elems());
  auto& dirk_dx = c.create<DirkFunctorST<DxFadTypeDirk>>(elems_dx_dirk.num_elems());

  // The HV<Real> in the Context is the one prim_advance_exp uses. prim_advance_adj
  // uses its own private one for the adjoint (see prim_advance_adj.cpp).
  auto& hv_r  = c.create<HyperviscosityFunctorST<Real>>(num_elems,params);
  auto& hv_dp = c.create<HyperviscosityFunctorST<DpFadType>>(num_elems,params);
  hv_r.setup(geo,elems_r.m_state,elems_r.m_derived);
  hv_dp.setup(geo,elems_dp.m_state,elems_dp.m_derived);

  // Setup scratch buffers
  FunctorsBuffersManager fbm;
  fbm.request_size(caar_r.requested_buffer_size());
  fbm.request_size(caar_dp.requested_buffer_size());
  fbm.request_size(caar_dx.requested_buffer_size());
  fbm.request_size(dirk_r.requested_buffer_size());
  fbm.request_size(dirk_dp.requested_buffer_size());
  fbm.request_size(dirk_dx.requested_buffer_size());
  fbm.request_size(limiter_r.requested_buffer_size());
  fbm.request_size(limiter_dp.requested_buffer_size());
  fbm.request_size(limiter_dx.requested_buffer_size());
  fbm.request_size(hv_r.requested_buffer_size());
  fbm.request_size(hv_dp.requested_buffer_size());

  fbm.allocate();

  caar_r.init_buffers(fbm);
  caar_dp.init_buffers(fbm);
  caar_dx.init_buffers(fbm);
  dirk_r.init_buffers(fbm);
  dirk_dp.init_buffers(fbm);
  dirk_dx.init_buffers(fbm);
  limiter_r.init_buffers(fbm);
  limiter_dp.init_buffers(fbm);
  limiter_dx.init_buffers(fbm);
  hv_r.init_buffers(fbm);
  hv_dp.init_buffers(fbm);
  hv_r.init_boundary_exchanges();
  hv_dp.init_boundary_exchanges();

  // The Euler step functors have their own buffers: euler_step_adj uses the ones
  // of the euler_step that ran right before as its tape, so nothing else must
  // be allowed to overwrite them in between.
  auto& euler_r = c.create<EulerStepFunctorST<Real>>(num_elems);
  euler_r.setup();
  euler_r.reset(params);
  FunctorsBuffersManager fbm_euler_r;
  fbm_euler_r.request_size(euler_r.requested_buffer_size());
  fbm_euler_r.allocate();
  euler_r.init_buffers(fbm_euler_r);
  euler_r.init_boundary_exchanges();

  EulerStepFunctorST<DpFadType> euler_dp(num_elems);
  euler_dp.setup();
  euler_dp.reset(params);
  FunctorsBuffersManager fbm_euler_dp;
  fbm_euler_dp.request_size(euler_dp.requested_buffer_size());
  fbm_euler_dp.allocate();
  euler_dp.init_buffers(fbm_euler_dp);
  euler_dp.init_boundary_exchanges();

  // Time levels (this is what prim_step reads and updates). Start at nstep=1,
  // so that, by the time prim_step gets to the tracers (K-1 leapfrog updates
  // later), nstep/K=1, hence the tracers use time levels (n0_qdp,np1_qdp)=(1,0).
  // Note that this differs from the time levels the tracers have during the
  // first prim_advance_exp, which is what a real run does too.
  auto& tl = c.create<TimeLevel>();
  tl.nm1 = 0;
  tl.n0  = 1;
  tl.np1 = 2;
  tl.nstep = 1;
  tl.nstep0 = 0;
  tl.tevolve = 0;
  tl.update_tracers_levels(params.dt_tracer_factor);

  // Scalar params
  const Real dt = 2;
  const int K = params.dt_tracer_factor;
  const Real eta_ave_w = 1.0/K;

  // ===================== Initial condition ===================== //
  // Dynamics state...
  StateSnapshot state_t0(num_elems);
  state_t0.randomize(seed,1e5,1e3,hvcoord.hybrid_ai0,geo.m_phis);
  elems_r.m_state.import_snapshot(state_t0,tl.n0);
  elems_dp.m_state.import_snapshot(state_t0,tl.n0);

  // ...and tracers: qdp must stay positive, to stay away from the 0-floor
  // and mass-relaxation branches of the limiter (for which euler_step_adj
  // knowingly gives an approximate gradient).
  const int qdp_slot = prim_step_final_time_level(tl,K).n0_qdp;
  genRandArray(tracers_r.qdp, engine, RPDF(300.0,400.0));

  // Checkpoint of the initial state (for the adjoint)
  PrimStepCheckpoint ckpt(num_elems,qsize);
  ckpt.take();

  // ===================== Fwd sweep (DpFadType) ===================== //
  // Same as prim_step (see prim_step.cpp), but on Fad types.
  // Random tangent du0 on the initial dynamics state and qdp
  elems_dp.m_state.randomize_derivs(seed,tl.n0);
  auto du0 = elems_dp.m_state.take_deriv_snapshot(tl.n0,0);

  auto dqdp = Kokkos::create_mirror(tracers_r.qdp);
  genRandArray(dqdp, engine, RPDF(-1.0,1.0));
  set_val(tracers_dp.qdp, tracers_r.qdp);
  set_deriv0(tracers_dp.qdp, dqdp);

  TimeLevel tl_dp = tl;
  printf(" -> Run forward problem (prim_step replica)...\n");
  set_tracer_transport_derived_values_st<DpFadType>(params,elems_dp,tl_dp);
  for (int n=0; n<K; ++n) {
    if (n>0) {
      tl_dp.update_dynamics_levels(UpdateType::LEAPFROG);
    }

    // Mirror the w_i(n0) surface fix at the top of prim_advance_exp (see
    // prim_advance_sacado_ut.cpp for why, and what it does here)
    {
      const int n0 = tl_dp.n0;
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

    ttype10_imex_timestep<DpFadType>(tl_dp.nm1,tl_dp.n0,tl_dp.np1,dt,eta_ave_w);
    hv_dp.run(tl_dp.np1,dt,eta_ave_w);
  }
  // The tangent of the dynamics state at the end of the prim_step
  auto duN = elems_dp.m_state.take_deriv_snapshot(tl_dp.np1,0);

  // Tracers step (see prim_advec_tracers_remap_RK2 in prim_advec_tracers_remap.cpp)
  tl_dp.update_tracers_levels(K);
  const int n0_qdp  = tl_dp.n0_qdp;
  const int np1_qdp = tl_dp.np1_qdp;
  REQUIRE (n0_qdp==qdp_slot); // consistency with the time levels used by the checkpoint
  {
    const Real dt_q = dt*K;
    euler_dp.reset(params);
    euler_dp.precompute_divdp();
    euler_dp.euler_step(np1_qdp,n0_qdp, dt_q/2.0,0.0,DSSOption::DIV_VDP_AVE);
    euler_dp.euler_step(np1_qdp,np1_qdp,dt_q/2.0,1.0,DSSOption::ETA);
    euler_dp.euler_step(np1_qdp,np1_qdp,dt_q/2.0,2.0,DSSOption::OMEGA);
    euler_dp.qdp_time_avg(n0_qdp,np1_qdp);
  }
  auto duN_qdp = Kokkos::create_mirror(tracers_r.qdp);
  get_deriv0(duN_qdp, tracers_dp.qdp);
  printf(" -> Run forward problem (prim_step replica)...done!\n");

  // ===================== Adjoint ===================== //
  constexpr auto tol = std::numeric_limits<double>::epsilon()*1e6;

  // Runs prim_step_adj with the given seeds (random or zero), and checks
  //   <du0, J^T*lambdaN> == <J*du0, lambdaN>
  auto check = [&](const bool seed_state, const bool seed_qdp, const std::string& name) {
    StateSnapshot lambda(num_elems);
    if (seed_state) {
      lambda.randomize(seed,1.0,1.0/100,0.0);
    } else {
      lambda.zero();
    }
    auto lambdaN = lambda.clone(true);

    PrimStepQdpView lambda_qdp("lambda_qdp",num_elems);
    {
      auto h = Kokkos::create_mirror(lambda_qdp);
      Kokkos::deep_copy(h,Real(0));
      if (seed_qdp) {
        genRandArray(h, engine, RPDF(-1.0,1.0));
      }
      Kokkos::deep_copy(lambda_qdp,h);
    }
    PrimStepQdpView lambdaN_qdp("lambdaN_qdp",num_elems);
    Kokkos::deep_copy(lambdaN_qdp,lambda_qdp);

    printf(" -> Run adjoint problem (prim_step_adj) [%s]...\n",name.c_str());
    prim_step_adj(dt,ckpt,lambda,lambda_qdp);
    printf(" -> Run adjoint problem (prim_step_adj) [%s]...done!\n",name.c_str());
    auto lambda0 = lambda; // prim_step_adj overwrites the seed in place

    // <du0,lambda0> vs <duN,lambdaN>
    const double state_dot0 = dot_state(du0,lambda0);
    const double state_dotN = dot_state(duN,lambdaN);
    const double qdp_dot0   = dot_qdp(lambda_qdp, dqdp,    n0_qdp,  qsize);
    const double qdp_dotN   = dot_qdp(lambdaN_qdp,duN_qdp, np1_qdp, qsize);

    const double full_dot0 = state_dot0 + qdp_dot0;
    const double full_dotN = state_dotN + qdp_dotN;
    if (comm.am_i_root())
      std::cout << std::setprecision(15)
                << "   [" << name << "]\n"
                << "   <du0, lambda0> = " << full_dot0
                << " (state: " << state_dot0 << ", qdp: " << qdp_dot0 << ")\n"
                << "   <duN, lambdaN> = " << full_dotN
                << " (state: " << state_dotN << ", qdp: " << qdp_dotN << ")\n";

    using namespace Catch::Matchers;
    CHECK_THAT (full_dot0, WithinRel(full_dotN,tol));
  };

  check(true,  false, "state seed");
  check(false, true,  "qdp seed");
}

} // namespace Homme

#endif // HOMMEXX_ENABLE_FAD_TYPES
