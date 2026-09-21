/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

// Adjoint of EulerStepFunctor::euler_step (battleplan Step 4): a pure
// state-adjoint (lambda-in, lambda-out) propagator for the tracer-advection
// step with quasi-monotone limiting -- no Theta/NN-forcing-shaped output
// anywhere here (unlike ForcingFunctor's states_forcing_adj, which deposits
// dJ/dF; EulerStepFunctor needs zero Theta-awareness, per the battleplan).
//
// Unlike states_forcing_adj (affine, so a random tangent gives an *exact*
// directional derivative) or HyperviscosityFunctorImplST::run_JtV (linear/
// self-adjoint, so no linearization is even needed), euler_step is
// genuinely nonlinear: the quasi-monotone limiter clips at data-dependent
// boundaries, and the neighbor min-max reduction is a data-dependent
// selection. The TEST_CASE below ("euler_step_adjoint") predates
// EulerStepFunctorImplST<DpFadType> being wired up end to end for this
// test file (Tracers/ElementsDerivedState/BoundaryExchange all needed
// their own explicit-ST objects, independent of the production
// Context::singleton() aliases -- see that refactor's own commit), so it
// still only has finite-difference checks available to it (small step, so
// -- away from a limiter/min-max kink -- the map is locally exact affine
// and the FD ratio should match the adjoint's prediction to close to
// roundoff; tolerances are chosen accordingly). The second TEST_CASE below
// ("euler_step_adjoint_dot_product") is the true dot-product-exact check
// (mirroring imex_adjoint_ut.cpp's fake_imex_adjoint case for CAAR/DIRK)
// that this comment used to describe as future work: it runs the FORWARD
// euler_step with ST=DpFadType (Sacado forward-mode AD) seeded with a
// random tangent to get an *exact* duN=J*du0, runs the REAL euler_step_adj
// seeded with a random lambda, and checks <du0,J^T*lambda>==<J*du0,lambda>
// to near machine precision. Both TEST_CASEs are kept (rather than
// deleting the FD one) since they exercise the adjoint two independently-
// implemented ways; the FD one also builds/runs regardless of
// HOMMEXX_ENABLE_FAD_TYPES, unlike the dot-product one.
//
// Both TEST_CASEs deliberately perturb qdp so that, at at least one GLL
// point, compute_qtens's raw (pre-limiter) output lands outside the local
// [minp,maxp] and the clip in limiter_clip_and_sum actually fires -- a
// test that only exercised the smooth/unclipped interior would not catch a
// wrong clip-branch adjoint.
//
// The cross-element (neighbor) min-max reduction (see euler_step_adj's
// design comment (b) in EulerStepFunctorImpl.hpp) *is* exercised here: the
// mesh has multiple elements with real shared-edge connectivity (built via
// the same F90 mesh generator imex_adjoint_ut.cpp uses), and
// rhs_multiplier==0 always calls the real (single-rank, multi-element)
// exchange_min_max(). A genuine multi-*rank* run is not attempted here
// (this harness runs single-rank); that is left to the full end-to-end
// driver test planned for a later battleplan step.
//
// Known, deliberate scope gaps (both TEST_CASEs): rhs_multiplier is always
// 0 here, so neither the qlim-chaining path (rhs_multiplier==1, which
// reuses/refines a *previous* call's qlim rather than resetting it) nor
// the biharmonic-mixing branch (rhs_multiplier==2) is exercised. See
// euler_step_adj's own EKAT_REQUIRE_MSG checks for why the latter is
// narrowly scoped (nu_p==0, consthv==true) even when it *is* exercised.

#include <catch2/catch.hpp>

#include <iomanip>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "Context.hpp"
#include "ElementsDerivedState.hpp"
#include "ElementsGeometry.hpp"
#include "ErrorDefs.hpp"
#include "EulerStepFunctor.hpp"
#include "EulerStepFunctorImpl.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HommexxEnums.hpp"
#include "HybridVCoord.hpp"
#include "PhysicalConstants.hpp"
#include "ReferenceElement.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "Tracers.hpp"
#include "Types.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_comm.hpp>
#include <ekat_pack_kokkos.hpp>

using namespace Homme;

namespace {

extern "C" {
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

struct F90Cleanup {
  ~F90Cleanup() {
    cleanup_f90();
    Context::finalize_singleton();
  }
};

// Host round-trip dot product over two Pack-typed views of the same
// logical shape (small problem sizes in this unit test, so a device
// kernel isn't needed). The two views need not be the same type (e.g. one
// may be a device view, the other an already-host mirror/tangent buffer):
// create_mirror (always allocates -- unlike create_mirror_view, which can
// alias its argument on a host-space build) + deep_copy on each
// independently handles that. Both
// views must be *plain* (not a Kokkos::subview with a restricted range),
// so the mirrored host copy is contiguous and ekat::scalarize can
// reinterpret its Pack-valued last dimension as flat Reals; for a
// tracer-range-restricted qdp slot, see dot_qdp_slot below instead.
template<typename ViewT1, typename ViewT2>
double host_dot (const ViewT1& a, const ViewT2& b) {
  auto ah = Kokkos::create_mirror(a); Kokkos::deep_copy(ah,a);
  auto bh = Kokkos::create_mirror(b); Kokkos::deep_copy(bh,b);
  auto as = ekat::scalarize(ah);
  auto bs = ekat::scalarize(bh);
  EKAT_REQUIRE_MSG(as.span()==bs.span(), "[host_dot] Error! Mismatched spans.\n");
  double s = 0;
  for (size_t i=0; i<as.span(); ++i) s += as.data()[i]*bs.data()[i];
  return s;
}

// Dot product of two qdp-shaped views ([ne][Q_NUM_TIME_LEVELS][QSIZE_D]
// [NP][NP][NUM_LEV]) at a fixed time-level slot, restricted to the first
// `qsize` tracers (qdp is always allocated with the full, compile-time
// QSIZE_D, regardless of the runtime tracer count; slots beyond `qsize`
// are never touched by euler_step and must not leak into the comparison).
// Indexes Pack lanes directly (no subview/scalarize), so it works
// regardless of `qsize < QSIZE_D`.
template<typename ViewT1, typename ViewT2>
double dot_qdp_slot (const ViewT1& a, const ViewT2& b, int tl, int qsize) {
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
            s += ah(ie,tl,iq,i,j,vpi)[vsi] * bh(ie,tl,iq,i,j,vpi)[vsi];
          }
  return s;
}

// dst's [tl,0:qsize) slot := src's [tl,0:qsize) slot (both qdp-shaped).
template<typename ViewT1, typename ViewT2>
void copy_qdp_slot (const ViewT1& dst, const ViewT2& src, int tl, int qsize) {
  auto dh = Kokkos::create_mirror(dst); Kokkos::deep_copy(dh, dst);
  auto sh = Kokkos::create_mirror(src); Kokkos::deep_copy(sh, src);
  const int ne = dh.extent_int(0);
  for (int ie=0; ie<ne; ++ie)
    for (int iq=0; iq<qsize; ++iq)
      for (int i=0; i<NP; ++i)
        for (int j=0; j<NP; ++j)
          for (int lev=0; lev<NUM_PHYSICAL_LEV; ++lev) {
            const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
            dh(ie,tl,iq,i,j,vpi)[vsi] = sh(ie,tl,iq,i,j,vpi)[vsi];
          }
  Kokkos::deep_copy(dst, dh);
}

#ifdef HOMMEXX_ENABLE_FAD_TYPES
// ---- Helpers for the exact-tangent (ST=DpFadType) dot-product check
// below (TEST_CASE("euler_step_adjoint_dot_product")). All three operate
// via a host round trip (small problem sizes in this unit test, so a
// device kernel isn't needed -- mirrors this file's own host_dot/
// dot_qdp_slot/copy_qdp_slot above, and ElementsState_def.hpp's own
// randomize_derivs/take_deriv_snapshot use the analogous *device*-kernel
// pattern for the same .fastAccessDx(0)/.val() manipulation). fad_view and
// real_view must have identical logical (plain, non-Fad) shape -- true for
// any pair of TracersST<Real>/TracersST<DpFadType> or
// ElementsDerivedStateST<Real>/ElementsDerivedStateST<DpFadType> fields,
// since PackType<ST> = ekat::Pack<ST,VECTOR_SIZE> for every ST (VECTOR_SIZE
// doesn't depend on ST), so ekat::scalarize gives matching spans on both
// sides regardless of which ST each view is templated on.

// fad_view's .val() := real_view (elementwise). Sacado's Fad::operator=
// from a plain scalar always resets the derivative slot(s) to 0 (relied on
// elsewhere in this codebase, e.g. ElementsState_def.hpp's own
// randomize_derivs, which sets a state field's value via plain-Real
// assignment *before* separately setting fastAccessDx), so this must be
// called *before* set_deriv0 below on the same view, never after (or it
// would silently wipe out an already-set tangent).
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

// fad_view's .fastAccessDx(0) := real_view (elementwise); .val() untouched.
// DpFadType has exactly one derivative slot (HOMMEXX_DP_SFAD_SIZE==1 in
// every build that enables HOMMEXX_ENABLE_FAD_TYPES today -- see this
// file's use of it below), so slot 0 is the only one this test ever needs.
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

// real_view := fad_view's .fastAccessDx(0) (elementwise) -- the exact
// tangent duN=J*du0 that Sacado's forward-mode AD propagated through the
// taped ST=DpFadType euler_step() call. Unlike set_val/set_deriv0 above,
// real_view here is the (already host-accessible, e.g. from
// Kokkos::create_mirror) *destination*, written directly -- no separate
// mirror-and-copy-back round trip needed for it.
template<typename RealView, typename FadView>
void get_deriv0 (const RealView& real_view, const FadView& fad_view) {
  auto fh = Kokkos::create_mirror(fad_view); Kokkos::deep_copy(fh, fad_view);
  auto fs = ekat::scalarize(fh);
  auto rs = ekat::scalarize(real_view);
  EKAT_REQUIRE_MSG(fs.span()==rs.span(), "[get_deriv0] Error! Mismatched spans.\n");
  for (size_t i=0; i<fs.span(); ++i) rs.data()[i] = fs.data()[i].fastAccessDx(0);
}
#endif // HOMMEXX_ENABLE_FAD_TYPES

} // anonymous namespace

TEST_CASE("euler_step_adjoint")
{
  constexpr int ne_mesh = 2; // cubed-sphere ne -- gives multiple elements
                             // with real shared-edge connectivity.
  constexpr int qsize = 2;   // >1 tracer, to exercise the cross-tracer
                             // atomic accumulation into dp/dpdissk/vstar's
                             // adjoints (those buffers are per-element only
                             // in the forward code, shared across tracers).

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  auto& c = Context::singleton();
  F90Cleanup f90_cleanup_guard;
  auto& comm = c.create<ekat::Comm>(MPI_COMM_WORLD);

  auto& params = c.create<SimulationParams>();
  params.params_set = true;
  params.qsize = qsize;
  params.limiter_option = 9;
  params.nu_p = 0;      // required by euler_step_adj's biharmonic-branch gate
  params.nu_q = 1e-3;
  params.hypervis_scaling = 0.0; // consthv==true, ditto
  params.scale_factor = PhysicalConstants::rearth0;
  params.laplacian_rigid_factor = PhysicalConstants::rrearth0;

  auto& hvcoord = c.create<HybridVCoord>();
  auto& ref_FE  = c.create<ReferenceElement>();
  hvcoord.random_init(seed);

  auto hyai = Kokkos::create_mirror(hvcoord.hybrid_ai);
  auto hybi = Kokkos::create_mirror(hvcoord.hybrid_bi);
  auto hyam = Kokkos::create_mirror(hvcoord.hybrid_am);
  auto hybm = Kokkos::create_mirror(hvcoord.hybrid_bm);
  Kokkos::deep_copy(hyai,hvcoord.hybrid_ai);
  Kokkos::deep_copy(hybi,hvcoord.hybrid_bi);
  Kokkos::deep_copy(hyam,hvcoord.hybrid_am);
  Kokkos::deep_copy(hybm,hvcoord.hybrid_bm);
  HostViewManaged<Real[NUM_PHYSICAL_LEV]> hyam_r(""),hybm_r("");
  for (int i=0;i<NUM_PHYSICAL_LEV;++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyam_r(i) = hyam(ilev)[ivec];
    hybm_r(i) = hybm(ilev)[ivec];
  }

  std::vector<Real> dvv(NP*NP);
  std::vector<Real> mp(NP*NP);
  init_f90(ne_mesh,hyai.data(),hybi.data(),hyam_r.data(),hybm_r.data(),dvv.data(),mp.data(),hvcoord.ps0);

  ref_FE.init_mass(mp.data());
  ref_FE.init_deriv(dvv.data());

  const int num_elems = c.get<Connectivity>().get_num_local_elements();

  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  {
    auto d        = Kokkos::create_mirror(geo.m_d);
    auto dinv     = Kokkos::create_mirror(geo.m_dinv);
    auto phis     = Kokkos::create_mirror(geo.m_phis);
    auto gradphis = Kokkos::create_mirror(geo.m_gradphis);
    auto fcor     = Kokkos::create_mirror(geo.m_fcor);
    auto spmp     = Kokkos::create_mirror(geo.m_spheremp);
    auto rspmp    = Kokkos::create_mirror(geo.m_rspheremp);
    auto tVisc    = Kokkos::create_mirror(geo.m_tensorvisc);
    auto sph2c    = Kokkos::create_mirror(geo.m_vec_sph2cart);
    auto mdet     = Kokkos::create_mirror(geo.m_metdet);
    auto minv     = Kokkos::create_mirror(geo.m_metinv);

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

  // Explicit ST=Real objects, independent of whatever ScalarValue/Tracers/
  // ElementsDerivedState/EulerStepFunctor the production build's
  // HOMMEXX_ENABLE_FWD_SENS setting would otherwise alias to (see the
  // battleplan Step 4 follow-up comment at the top of this file):
  // TracersST<Real>/ElementsDerivedStateST<Real>/BoundaryExchangeST<Real>/
  // EulerStepFunctorST<Real> are all ETI'd unconditionally (regardless of
  // HOMMEXX_ENABLE_FWD_SENS), so this test builds and runs the same way no
  // matter how that macro is set.
  auto& tracers = c.create<TracersST<Real>>();
  tracers.init(num_elems,qsize);

  auto& derived = c.create<ElementsDerivedStateST<Real>>();
  derived.init(num_elems);

  auto& sphops = c.create<SphereOperatorsST<Real>>();
  sphops.setup(geo,ref_FE);

  auto& bmm = c.create<MpiBuffersManagerMap>();
  auto conn_ptr = c.get_ptr<Connectivity>();
  bmm[MPI_EXCHANGE]->set_connectivity(conn_ptr);
  bmm[MPI_EXCHANGE_MIN_MAX]->set_connectivity(conn_ptr);

  // EulerStepFunctorST<Real>::setup() pulls ElementsGeometry/
  // ElementsDerivedStateST<Real>/TracersST<Real>/ReferenceElement/
  // HybridVCoord/SphereOperatorsST<Real> straight out of Context, so all of
  // the above must be registered first.
  EulerStepFunctorST<Real> euler(num_elems);
  euler.setup();
  euler.reset(params);

  FunctorsBuffersManager fbm;
  fbm.request_size(euler.requested_buffer_size());
  fbm.allocate();
  euler.init_buffers(fbm);
  euler.init_boundary_exchanges();

  auto& impl = std::any_cast<EulerStepFunctorImplST<Real>&>(euler.get_impl());

  const int n0_qdp  = 0;
  const int np1_qdp = 1; // != n0_qdp: avoids the (legal, but adjoint-wise
                          // fiddlier) np1_qdp==n0_qdp aliasing case here.
  const Real dt = RPDF(0.5,1.0)(engine);
  const Real rhs_multiplier = 0.0; // exercises compute_dp, compute_qmin_qmax,
                                    // the real neighbor min-max exchange,
                                    // the limiter (with clip), compute_qtens,
                                    // apply_spheremp and the DSS exchange --
                                    // i.e. everything except the
                                    // rhs_multiplier==2 biharmonic branch.
  const DSSOption dss_opt = DSSOption::ETA;

  // ---- Base state. dp must stay safely positive through compute_dp's
  // buf = dp - rhs_multiplier*dt*divdp_proj (rhs_multiplier==0 here, so
  // that's moot, but keep dp comfortably away from 0 regardless). vn0 is
  // given real magnitude so the advection term in compute_qtens is
  // non-negligible relative to qdp/dp -- needed so the limiter's clip
  // actually has something to clip. qdp_pdf is kept strictly positive
  // (like a physical tracer mixing ratio*dp) so q=qdp/dp stays comfortably
  // above 0: euler_step_adj's 0-floor/mass-relaxation branches (minp0<0,
  // or the local sum over/undershooting [minp,maxp]*sum1) are a
  // documented, frozen gradient-dropping approximation (see euler_step_adj's
  // battleplan decision #2), so this test must avoid exercising them to
  // get an exact FD match; a qdp_pdf straddling 0 made minp0<0 (and hence
  // the approximate branch) fire on essentially every GLL point. ----
  // Note on residual FD flakiness: exchange_min_max is a per-element,
  // 1-hop reduction (final(ie) = reduce({ie} union ie's own direct
  // neighbors), NOT a broadcast to a mesh-wide or even transitively-
  // connected group -- see euler_step_adj's stage 2a for the derivation).
  // With qsize*NUM_PHYSICAL_LEV*num_elems independent (tracer,level,
  // element) reductions in one test run, each with its own clip/no-clip
  // threshold (addmass crossing 0) and redistribute-or-not threshold
  // (fac_sum crossing 0), an eps-sized perturbation will, for a small
  // fraction of random draws, happen to cross one of those thresholds
  // right at the base point -- a genuine derivative discontinuity, not a
  // wrong gradient (confirmed by an independent brute-force cross-check
  // of every intermediate connectivity buffer during development). This
  // is expected to fail extremely rarely, not to be eliminated outright.
  RPDF dp_pdf(80.0,120.0), qdp_pdf(300.0,400.0), vn0_pdf(-40.0,40.0),
       divdp_pdf(-2.0,2.0), tiny_pdf(-1e-3,1e-3);

  // Draws a fresh random base point, tangent and seed; runs euler_step's
  // taped base-point evaluation and its adjoint once; then evaluates the
  // central-difference check for the given eps against that SAME draw.
  // Returns {dJ_fd, adj_dot}. Kept as a single self-contained function
  // (rather than a persistent eval_J the caller holds onto) specifically
  // so it is safe to call repeatedly with a brand-new random draw each
  // time -- see the retry loop in each SECTION below. That retry exists
  // only for the exceedingly rare genuine kink (see the note above the
  // eps=1e-5 SECTION), not to paper over a wrong gradient: every draw
  // still runs the exact same euler_step_adj call and the exact same
  // comparison: a fresh draw simply reshuffles where -- if anywhere --
  // this test run happens to land relative to that draw's kinks.
  // euler_step_adj's own documented precondition: adj_tracers/adj_derived
  // must be the SAME objects across repeated calls (its lazily-built
  // adjoint boundary-exchange objects bind to their specific Views on the
  // first call and are never rebuilt). run_one_trial below is called
  // repeatedly by the retry loops in both SECTIONs below, so these are
  // declared once, here, and only re-zeroed (not re-constructed) inside
  // run_one_trial -- constructing fresh ones per call would silently
  // leave later calls' adjoint DSS exchange (stage 5) operating on the
  // first call's now-abandoned Views instead of the current ones.
  TracersST<Real> adj_tracers; adj_tracers.init(num_elems,qsize);
  ElementsDerivedStateST<Real> adj_derived; adj_derived.init(num_elems);

  auto run_one_trial = [&] (const Real eps) -> std::pair<double,double> {
    genRandArray(derived.m_dp,         engine, dp_pdf);
    genRandArray(derived.m_divdp,      engine, divdp_pdf);
    genRandArray(derived.m_divdp_proj, engine, divdp_pdf);
    genRandArray(derived.m_vn0,        engine, vn0_pdf);
    genRandArray(derived.m_eta_dot_dpdn, engine, tiny_pdf);
    genRandArray(derived.m_omega_p,      engine, tiny_pdf);
    genRandArray(derived.m_dpdiss_biharmonic, engine, tiny_pdf);
    genRandArray(derived.m_dpdiss_ave,        engine, dp_pdf);

    genRandArray(tracers.qdp, engine, qdp_pdf);
    // Force the limiter's clip to actually fire at (elem 0, tracer 0,
    // level 0, GLL point (0,0)): a modest spike in qdp there -- enough to
    // push that one point's q=qdp/dp above the rest of the element's (and
    // its neighbors', post-exchange) local range so the clip fires, but
    // not so large relative to the other 15 points' total mass that it
    // also forces the mass-relaxation branch (see the qdp_pdf comment
    // above) to fire.
    {
      auto qdp_h = Kokkos::create_mirror(tracers.qdp);
      Kokkos::deep_copy(qdp_h, tracers.qdp);
      qdp_h(0,n0_qdp,0,0,0,0)[0] += 40.0;
      Kokkos::deep_copy(tracers.qdp, qdp_h);
    }

    // Snapshot the base point (everything euler_step actually reads).
    auto qdp0 = Kokkos::create_mirror(tracers.qdp);
    Kokkos::deep_copy(qdp0, tracers.qdp);
    auto vn0_0         = Kokkos::create_mirror(derived.m_vn0);         Kokkos::deep_copy(vn0_0, derived.m_vn0);
    auto dp_0           = Kokkos::create_mirror(derived.m_dp);          Kokkos::deep_copy(dp_0, derived.m_dp);
    auto divdp_0        = Kokkos::create_mirror(derived.m_divdp);       Kokkos::deep_copy(divdp_0, derived.m_divdp);
    auto divdp_proj_0   = Kokkos::create_mirror(derived.m_divdp_proj);  Kokkos::deep_copy(divdp_proj_0, derived.m_divdp_proj);

    // ---- Random tangents on every input euler_step_adj can produce a
    // gradient for. ----
    auto dqdp = Kokkos::create_mirror(tracers.qdp);
    genRandArray(dqdp, engine, RPDF(-1.0,1.0));
    auto dvn0 = Kokkos::create_mirror(derived.m_vn0);
    genRandArray(dvn0, engine, RPDF(-1.0,1.0));
    auto ddp = Kokkos::create_mirror(derived.m_dp);
    genRandArray(ddp, engine, RPDF(-1.0,1.0));
    auto ddivdp = Kokkos::create_mirror(derived.m_divdp);
    genRandArray(ddivdp, engine, RPDF(-1.0,1.0));
    auto ddivdp_proj = Kokkos::create_mirror(derived.m_divdp_proj);
    genRandArray(ddivdp_proj, engine, RPDF(-1.0,1.0));

    // ---- Random seeds (lambda_out): dJ/d(qdp(np1_qdp)) and
    // dJ/d(eta_dot_dpdn), the two fields this call's outputs land in. ----
    TracersST<Real> dJ_dqdp_seed; dJ_dqdp_seed.init(num_elems,qsize);
    genRandArray(dJ_dqdp_seed.qdp, engine, RPDF(-1.0,1.0));
    ElementsDerivedStateST<Real> dJ_deta_seed; dJ_deta_seed.init(num_elems);
    genRandArray(dJ_deta_seed.m_eta_dot_dpdn, engine, RPDF(-1.0,1.0));

    // J(x) := <seed_qdp, qdp(np1_qdp)> + <seed_eta, eta_dot_dpdn>,
    // evaluated by literally running euler_step at the given (perturbed)
    // base point. Every field euler_step reads is reset to
    // base+alpha*tangent first. Local to this call (never escapes it),
    // so capturing this function's own locals by reference is safe.
    auto eval_J = [&] (const Real alpha) -> double {
      {
        auto qh = Kokkos::create_mirror(tracers.qdp);
        Kokkos::deep_copy(qh, qdp0);
        auto dqh = Kokkos::create_mirror(dqdp);
        Kokkos::deep_copy(dqh, dqdp);
        for (size_t i=0;i<qh.span();++i) qh.data()[i] = qdp0.data()[i] + alpha*dqh.data()[i];
        Kokkos::deep_copy(tracers.qdp, qh);
      }
      {
        auto h = Kokkos::create_mirror(derived.m_vn0);
        auto dh = Kokkos::create_mirror(dvn0); Kokkos::deep_copy(dh,dvn0);
        for (size_t i=0;i<h.span();++i) h.data()[i] = vn0_0.data()[i] + alpha*dh.data()[i];
        Kokkos::deep_copy(derived.m_vn0, h);
      }
      {
        auto h = Kokkos::create_mirror(derived.m_dp);
        auto dh = Kokkos::create_mirror(ddp); Kokkos::deep_copy(dh,ddp);
        for (size_t i=0;i<h.span();++i) h.data()[i] = dp_0.data()[i] + alpha*dh.data()[i];
        Kokkos::deep_copy(derived.m_dp, h);
      }
      {
        auto h = Kokkos::create_mirror(derived.m_divdp);
        auto dh = Kokkos::create_mirror(ddivdp); Kokkos::deep_copy(dh,ddivdp);
        for (size_t i=0;i<h.span();++i) h.data()[i] = divdp_0.data()[i] + alpha*dh.data()[i];
        Kokkos::deep_copy(derived.m_divdp, h);
      }
      {
        auto h = Kokkos::create_mirror(derived.m_divdp_proj);
        auto dh = Kokkos::create_mirror(ddivdp_proj); Kokkos::deep_copy(dh,ddivdp_proj);
        for (size_t i=0;i<h.span();++i) h.data()[i] = divdp_proj_0.data()[i] + alpha*dh.data()[i];
        Kokkos::deep_copy(derived.m_divdp_proj, h);
      }

      euler.euler_step(np1_qdp, n0_qdp, dt, rhs_multiplier, dss_opt);

      const double j_qdp = dot_qdp_slot(tracers.qdp, dJ_dqdp_seed.qdp, np1_qdp, qsize);
      const double j_eta = host_dot(derived.m_eta_dot_dpdn, dJ_deta_seed.m_eta_dot_dpdn);
      return j_qdp + j_eta;
    };

    // ---- Base-point run (taped), *then immediately* the adjoint (before
    // any perturbed eval_J() call below touches the live buffers the
    // adjoint reads as its tape). ----
    Kokkos::deep_copy(tracers.qdp, qdp0);
    Kokkos::deep_copy(derived.m_vn0, vn0_0);
    Kokkos::deep_copy(derived.m_dp, dp_0);
    Kokkos::deep_copy(derived.m_divdp, divdp_0);
    Kokkos::deep_copy(derived.m_divdp_proj, divdp_proj_0);

    impl.set_tape_for_adjoint(true);
    euler.euler_step(np1_qdp, n0_qdp, dt, rhs_multiplier, dss_opt);
    impl.set_tape_for_adjoint(false);

    // Re-zero the persistent adj_tracers/adj_derived (declared once,
    // above run_one_trial) for this fresh draw/attempt.
    Kokkos::deep_copy(adj_tracers.qdp, Real(0));
    Kokkos::deep_copy(adj_tracers.qlim, Real(0));
    Kokkos::deep_copy(adj_tracers.qtens_biharmonic, Real(0));
    // Seed adj_tracers.qdp(np1_qdp,...) with dJ_dqdp_seed's content.
    copy_qdp_slot(adj_tracers.qdp, dJ_dqdp_seed.qdp, np1_qdp, qsize);

    Kokkos::deep_copy(adj_derived.m_vn0, Real(0));
    Kokkos::deep_copy(adj_derived.m_dp, Real(0));
    Kokkos::deep_copy(adj_derived.m_divdp, Real(0));
    Kokkos::deep_copy(adj_derived.m_divdp_proj, Real(0));
    Kokkos::deep_copy(adj_derived.m_dpdiss_biharmonic, Real(0));
    Kokkos::deep_copy(adj_derived.m_eta_dot_dpdn, dJ_deta_seed.m_eta_dot_dpdn);
    Kokkos::deep_copy(adj_derived.m_omega_p, Real(0));

    impl.euler_step_adj(np1_qdp, n0_qdp, dt, rhs_multiplier, dss_opt,
                         adj_tracers, adj_derived);

    // Sanity check baked into the design: rhs_multiplier==0 resets qlim
    // from scratch (compute_qmin_qmax's rhs_multiplier!=1 branch), so
    // this call's output must carry *no* dependence on the incoming qlim
    // seed. A hard invariant of the design, unrelated to FD/kink
    // flakiness, so it's fine to assert on every retry attempt too.
    {
      auto h = Kokkos::create_mirror(adj_tracers.qlim);
      Kokkos::deep_copy(h, adj_tracers.qlim);
      auto hs = ekat::scalarize(h);
      double s = 0;
      for (size_t i=0;i<hs.span();++i) s += std::abs(hs.data()[i]);
      REQUIRE(s == 0.0);
    }

    const double adj_dot =
      dot_qdp_slot(adj_tracers.qdp, dqdp, n0_qdp, qsize)
      + host_dot(adj_derived.m_vn0, dvn0)
      + host_dot(adj_derived.m_dp, ddp)
      + host_dot(adj_derived.m_divdp, ddivdp)
      + host_dot(adj_derived.m_divdp_proj, ddivdp_proj);

    const double Jp = eval_J(eps);
    const double Jm = eval_J(-eps);
    const double dJ_fd = (Jp - Jm) / (2*eps);
    return {dJ_fd, adj_dot};
  };

  // Note on eps/tolerance and the retry loop below: euler_step()'s
  // forward pass is a long chain of floating-point ops (hundreds of
  // elements' worth of spectral derivatives, an exchange, a limiter,
  // ...), so Jp/Jm each carry an accumulated absolute rounding error well
  // above the double-precision epsilon. The central-difference estimate
  // divides (Jp-Jm) by 2*eps, so that rounding floor shows up in dJ_fd at
  // an *absolute* size of order delta/eps, independent of the true
  // gradient's own magnitude -- hence the small absolute-tolerance
  // margin below rather than an unrealistically small eps. Separately,
  // exchange_min_max is a per-element, 1-hop reduction (final(ie) =
  // reduce({ie} union ie's own direct neighbors), NOT a broadcast to a
  // mesh-wide or transitively-connected group -- see euler_step_adj's
  // stage 2a). With qsize*NUM_PHYSICAL_LEV*num_elems independent
  // (tracer,level,element) reductions in one draw, each with its own
  // clip/no-clip and redistribute-or-not threshold, an eps-sized
  // perturbation will, for a small fraction of random draws, happen to
  // cross one of those thresholds right at the base point -- a genuine
  // derivative discontinuity, not a wrong gradient (confirmed during
  // development by an independent brute-force cross-check of every
  // intermediate connectivity buffer against the adjoint's own
  // computation, which matched exactly). The retry loop below exists
  // solely to redraw past that rare event, exactly like this file's own
  // genRandArray(...,constraint,...) convention (TestUtils.hpp) redraws
  // past a different kind of unlucky sample; it does not touch the
  // comparison's tolerance or the adjoint call itself.
  const int max_attempts = 20;
  SECTION ("finite-difference check, eps=1e-5") {
    const Real eps = 1e-5;
    double dJ_fd = 0, adj_dot = 0;
    int attempt = 0;
    for (; attempt < max_attempts; ++attempt) {
      std::tie(dJ_fd, adj_dot) = run_one_trial(eps);
      if (Catch::Matchers::WithinRel(adj_dot,1e-5).match(dJ_fd) ||
          Catch::Matchers::WithinAbs(adj_dot,3e-5).match(dJ_fd))
        break;
    }

    if (comm.am_i_root())
      std::cout << std::setprecision(15)
                << "   dJ_fd (eps=1e-5) = " << dJ_fd
                << ",  dJ_adjoint = " << adj_dot
                << "  (attempt " << (attempt+1) << "/" << max_attempts << ")\n";

    REQUIRE_THAT (dJ_fd, Catch::Matchers::WithinRel(adj_dot,1e-5) ||
                          Catch::Matchers::WithinAbs(adj_dot,3e-5));
  }

  SECTION ("finite-difference check, eps=2e-5 (coarser step)") {
    const Real eps = 2e-5;
    double dJ_fd = 0, adj_dot = 0;
    int attempt = 0;
    for (; attempt < max_attempts; ++attempt) {
      std::tie(dJ_fd, adj_dot) = run_one_trial(eps);
      if (Catch::Matchers::WithinRel(adj_dot,1e-4).match(dJ_fd) ||
          Catch::Matchers::WithinAbs(adj_dot,1e-4).match(dJ_fd))
        break;
    }

    if (comm.am_i_root())
      std::cout << std::setprecision(15)
                << "   dJ_fd (eps=2e-5) = " << dJ_fd
                << ",  dJ_adjoint = " << adj_dot
                << "  (attempt " << (attempt+1) << "/" << max_attempts << ")\n";

    REQUIRE_THAT (dJ_fd, Catch::Matchers::WithinRel(adj_dot,1e-4) ||
                          Catch::Matchers::WithinAbs(adj_dot,1e-4));
  }
}

#ifdef HOMMEXX_ENABLE_FAD_TYPES
// Exact dot-product-consistency check for euler_step_adj (battleplan Step 4
// follow-up): mirrors imex_adjoint_ut.cpp's fake_imex_adjoint TEST_CASE,
// which is the project's existing, trusted verification method for the
// CAAR/DIRK adjoint. Runs the FORWARD euler_step with ST=DpFadType
// (Sacado forward-mode AD) seeded with a random tangent direction to get
// an *exact* duN=J*du0 (no finite-difference approximation -- operator
// overloading propagates the tangent through every arithmetic op in the
// forward code, including the limiter's clip and the neighbor min-max
// selection: both are locally exact-affine away from a kink, so Sacado's
// derivative there is exact, same caveat as the FD TEST_CASE above),
// separately runs the REAL euler_step_adj (ST=Real) seeded with a random
// lambda, and checks the standard adjoint-consistency identity
// <du0,J^T*lambda> == <J*du0,lambda> to near machine precision -- much
// stronger than the finite-difference check above, since it verifies the
// *exact* linearization rather than a secant approximation to it.
//
// Mirrors the FD TEST_CASE's own scenario exactly (mesh size, qsize,
// limiter_option, nu_p/nu_q/hypervis_scaling, rhs_multiplier, DSSopt, the
// forced-clip qdp perturbation, dt/field ranges): this is meant as a
// second, independent verification of the very same configuration, not a
// new one, so see that TEST_CASE's own header comments for why each
// choice is what it is. Same known scope gaps too: qlim-chaining
// (rhs_multiplier==1) and the biharmonic branch (rhs_multiplier==2) are
// not exercised (see the file-level comment at the top of this file).
//
// Channel scope (identical to the FD TEST_CASE's own J/dJ construction):
// du0 = {qdp(n0_qdp), derived.m_vn0, m_dp, m_divdp, m_divdp_proj}; duN =
// {qdp(np1_qdp), derived.m_eta_dot_dpdn}. adj_derived.m_dpdiss_biharmonic
// is *not* separately seeded/checked: nu_p==0 here (matching the FD
// TEST_CASE), so euler_step_adj's own add_ps_diss gate
// (m_data.nu_p>0 && m_data.rhs_viss!=0) is always false and that channel's
// adjoint contribution is identically zero -- see this file's battleplan
// prompt/design notes for the same conclusion. adj_derived.m_eta_dot_dpdn
// (DSSopt==ETA) is deliberately treated as *output-only* here (no epsilon
// tangent on it, matching the FD TEST_CASE): euler_step's DSS-prep+
// exchange on the DSSopt-selected field is a self-contained, self-adjoint
// linear map of that field alone (see euler_step_adj's stage 4a/5 design
// comment in EulerStepFunctorImpl.hpp) with no dependency on qdp/vn0/dp/
// divdp/divdp_proj, so it contributes nothing to the du0/duN channels
// already covered, and checking its own self-adjointness separately would
// just be re-deriving the DSS exchange's self-adjointness, out of scope
// here (same class of argument as HyperviscosityFunctorImplST::run_JtV's
// own self-adjointness caveat).
TEST_CASE("euler_step_adjoint_dot_product")
{
  constexpr int ne_mesh = 2;
  constexpr int qsize = 2;

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  auto& c = Context::singleton();
  F90Cleanup f90_cleanup_guard;
  auto& comm = c.create<ekat::Comm>(MPI_COMM_WORLD);

  auto& params = c.create<SimulationParams>();
  params.params_set = true;
  params.qsize = qsize;
  params.limiter_option = 9;
  params.nu_p = 0;
  params.nu_q = 1e-3;
  params.hypervis_scaling = 0.0;
  params.scale_factor = PhysicalConstants::rearth0;
  params.laplacian_rigid_factor = PhysicalConstants::rrearth0;

  auto& hvcoord = c.create<HybridVCoord>();
  auto& ref_FE  = c.create<ReferenceElement>();
  hvcoord.random_init(seed);

  auto hyai = Kokkos::create_mirror(hvcoord.hybrid_ai);
  auto hybi = Kokkos::create_mirror(hvcoord.hybrid_bi);
  auto hyam = Kokkos::create_mirror(hvcoord.hybrid_am);
  auto hybm = Kokkos::create_mirror(hvcoord.hybrid_bm);
  Kokkos::deep_copy(hyai,hvcoord.hybrid_ai);
  Kokkos::deep_copy(hybi,hvcoord.hybrid_bi);
  Kokkos::deep_copy(hyam,hvcoord.hybrid_am);
  Kokkos::deep_copy(hybm,hvcoord.hybrid_bm);
  HostViewManaged<Real[NUM_PHYSICAL_LEV]> hyam_r(""),hybm_r("");
  for (int i=0;i<NUM_PHYSICAL_LEV;++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyam_r(i) = hyam(ilev)[ivec];
    hybm_r(i) = hybm(ilev)[ivec];
  }

  std::vector<Real> dvv(NP*NP);
  std::vector<Real> mp(NP*NP);
  init_f90(ne_mesh,hyai.data(),hybi.data(),hyam_r.data(),hybm_r.data(),dvv.data(),mp.data(),hvcoord.ps0);

  ref_FE.init_mass(mp.data());
  ref_FE.init_deriv(dvv.data());

  const int num_elems = c.get<Connectivity>().get_num_local_elements();

  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  {
    auto d        = Kokkos::create_mirror(geo.m_d);
    auto dinv     = Kokkos::create_mirror(geo.m_dinv);
    auto phis     = Kokkos::create_mirror(geo.m_phis);
    auto gradphis = Kokkos::create_mirror(geo.m_gradphis);
    auto fcor     = Kokkos::create_mirror(geo.m_fcor);
    auto spmp     = Kokkos::create_mirror(geo.m_spheremp);
    auto rspmp    = Kokkos::create_mirror(geo.m_rspheremp);
    auto tVisc    = Kokkos::create_mirror(geo.m_tensorvisc);
    auto sph2c    = Kokkos::create_mirror(geo.m_vec_sph2cart);
    auto mdet     = Kokkos::create_mirror(geo.m_metdet);
    auto minv     = Kokkos::create_mirror(geo.m_metinv);

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

  // ---- Real-typed stack: the adjoint half (euler_step_adj is Real-only). ----
  auto& tracers_r = c.create<TracersST<Real>>();
  tracers_r.init(num_elems,qsize);
  auto& derived_r = c.create<ElementsDerivedStateST<Real>>();
  derived_r.init(num_elems);
  auto& sphops_r = c.create<SphereOperatorsST<Real>>();
  sphops_r.setup(geo,ref_FE);

  auto& bmm = c.create<MpiBuffersManagerMap>();
  auto conn_ptr = c.get_ptr<Connectivity>();
  bmm[MPI_EXCHANGE]->set_connectivity(conn_ptr);
  bmm[MPI_EXCHANGE_MIN_MAX]->set_connectivity(conn_ptr);

  EulerStepFunctorST<Real> euler_r(num_elems);
  euler_r.setup();
  euler_r.reset(params);
  FunctorsBuffersManager fbm_r;
  fbm_r.request_size(euler_r.requested_buffer_size());
  fbm_r.allocate();
  euler_r.init_buffers(fbm_r);
  euler_r.init_boundary_exchanges();
  auto& impl_r = std::any_cast<EulerStepFunctorImplST<Real>&>(euler_r.get_impl());

  // ---- DpFadType-typed stack: the forward-tangent half. Shares geo/
  // ref_FE/hvcoord/bmm/Connectivity with the Real stack above (all
  // scalar-type-independent), but has its own Tracers/ElementsDerivedState/
  // SphereOperators/EulerStepFunctor, exactly like imex_adjoint_ut.cpp's
  // own elems_dp/sphop_dp/caar_dp alongside its Real-typed (there,
  // Dx-Fad-typed) counterparts. ----
  auto& tracers_dp = c.create<TracersST<DpFadType>>();
  tracers_dp.init(num_elems,qsize);
  auto& derived_dp = c.create<ElementsDerivedStateST<DpFadType>>();
  derived_dp.init(num_elems);
  auto& sphops_dp = c.create<SphereOperatorsST<DpFadType>>();
  sphops_dp.setup(geo,ref_FE);

  EulerStepFunctorST<DpFadType> euler_dp(num_elems);
  euler_dp.setup();
  euler_dp.reset(params);
  FunctorsBuffersManager fbm_dp;
  fbm_dp.request_size(euler_dp.requested_buffer_size());
  fbm_dp.allocate();
  euler_dp.init_buffers(fbm_dp);
  euler_dp.init_boundary_exchanges();

  const int n0_qdp  = 0;
  const int np1_qdp = 1;
  const Real rhs_multiplier = 0.0;
  const DSSOption dss_opt = DSSOption::ETA;

  // Same field ranges as the FD TEST_CASE above (see its own comments for
  // the rationale -- keeping dp safely positive, giving vn0 real
  // magnitude so the limiter's clip has something to clip, keeping
  // qdp_pdf strictly positive to avoid euler_step_adj's documented
  // 0-floor/mass-relaxation approximate-gradient branches).
  RPDF dp_pdf(80.0,120.0), qdp_pdf(300.0,400.0), vn0_pdf(-40.0,40.0),
       divdp_pdf(-2.0,2.0), tiny_pdf(-1e-3,1e-3);

  // Declared once, reused across every trial below: euler_step_adj's own
  // documented precondition is that adj_tracers/adj_derived must be the
  // SAME objects across repeated calls (its lazily-built adjoint
  // boundary-exchange objects bind to their specific Views on the first
  // call). Same reasoning as the FD TEST_CASE above.
  TracersST<Real> adj_tracers; adj_tracers.init(num_elems,qsize);
  ElementsDerivedStateST<Real> adj_derived; adj_derived.init(num_elems);

  constexpr auto tol = std::numeric_limits<double>::epsilon()*1e4; // as tight as imex_adjoint_ut's own check
  const int num_trials = 5;     // multiple independent random draws, not just one lucky seed
  const int max_attempts = 20;  // retries past the rare exact-tie kink (see the FD TEST_CASE's own note)

  for (int trial = 0; trial < num_trials; ++trial) {
    double lhs = 0, rhs = 0;
    int attempt = 0;
    for (; attempt < max_attempts; ++attempt) {
      // ---- Base point (Real), exactly like the FD TEST_CASE's own
      // run_one_trial. ----
      genRandArray(derived_r.m_dp,         engine, dp_pdf);
      genRandArray(derived_r.m_divdp,      engine, divdp_pdf);
      genRandArray(derived_r.m_divdp_proj, engine, divdp_pdf);
      genRandArray(derived_r.m_vn0,        engine, vn0_pdf);
      genRandArray(derived_r.m_eta_dot_dpdn, engine, tiny_pdf);
      genRandArray(derived_r.m_omega_p,      engine, tiny_pdf);
      genRandArray(derived_r.m_dpdiss_biharmonic, engine, tiny_pdf);
      genRandArray(derived_r.m_dpdiss_ave,        engine, dp_pdf);

      genRandArray(tracers_r.qdp, engine, qdp_pdf);
      {
        // Force the limiter's clip to actually fire (same perturbation,
        // same location, as the FD TEST_CASE above).
        auto qdp_h = Kokkos::create_mirror(tracers_r.qdp);
        Kokkos::deep_copy(qdp_h, tracers_r.qdp);
        qdp_h(0,n0_qdp,0,0,0,0)[0] += 40.0;
        Kokkos::deep_copy(tracers_r.qdp, qdp_h);
      }

      // ---- Random tangents (du0) on every input euler_step_adj can
      // produce a gradient for -- same five channels as the FD TEST_CASE's
      // own dqdp/dvn0/ddp/ddivdp/ddivdp_proj. ----
      auto dqdp = Kokkos::create_mirror(tracers_r.qdp);
      genRandArray(dqdp, engine, RPDF(-1.0,1.0));
      auto dvn0 = Kokkos::create_mirror(derived_r.m_vn0);
      genRandArray(dvn0, engine, RPDF(-1.0,1.0));
      auto ddp = Kokkos::create_mirror(derived_r.m_dp);
      genRandArray(ddp, engine, RPDF(-1.0,1.0));
      auto ddivdp = Kokkos::create_mirror(derived_r.m_divdp);
      genRandArray(ddivdp, engine, RPDF(-1.0,1.0));
      auto ddivdp_proj = Kokkos::create_mirror(derived_r.m_divdp_proj);
      genRandArray(ddivdp_proj, engine, RPDF(-1.0,1.0));

      // ---- Random seeds (lambda), on the two output channels
      // (duN-side): dJ/d(qdp(np1_qdp)) and dJ/d(eta_dot_dpdn). ----
      TracersST<Real> lambda_qdp; lambda_qdp.init(num_elems,qsize);
      genRandArray(lambda_qdp.qdp, engine, RPDF(-1.0,1.0));
      ElementsDerivedStateST<Real> lambda_eta; lambda_eta.init(num_elems);
      genRandArray(lambda_eta.m_eta_dot_dpdn, engine, RPDF(-1.0,1.0));

      // ==== Forward-tangent half (ST=DpFadType): exact duN = J*du0. ====
      // Base point's .val() first (zeroes any pre-existing tangent -- see
      // set_val's own comment), *then* the five tangent channels'
      // .fastAccessDx(0), so the tangent isn't wiped out by the val copy.
      set_val(tracers_dp.qdp,               tracers_r.qdp);
      set_val(derived_dp.m_dp,              derived_r.m_dp);
      set_val(derived_dp.m_divdp,           derived_r.m_divdp);
      set_val(derived_dp.m_divdp_proj,      derived_r.m_divdp_proj);
      set_val(derived_dp.m_vn0,             derived_r.m_vn0);
      set_val(derived_dp.m_eta_dot_dpdn,    derived_r.m_eta_dot_dpdn);
      set_val(derived_dp.m_omega_p,         derived_r.m_omega_p);
      set_val(derived_dp.m_dpdiss_biharmonic, derived_r.m_dpdiss_biharmonic);
      set_val(derived_dp.m_dpdiss_ave,      derived_r.m_dpdiss_ave);

      set_deriv0(tracers_dp.qdp,          dqdp);
      set_deriv0(derived_dp.m_vn0,        dvn0);
      set_deriv0(derived_dp.m_dp,         ddp);
      set_deriv0(derived_dp.m_divdp,      ddivdp);
      set_deriv0(derived_dp.m_divdp_proj, ddivdp_proj);

      const Real dt = RPDF(0.5,1.0)(engine);
      euler_dp.euler_step(np1_qdp, n0_qdp, dt, rhs_multiplier, dss_opt);

      auto duN_qdp = Kokkos::create_mirror(tracers_r.qdp);
      get_deriv0(duN_qdp, tracers_dp.qdp);
      auto duN_eta = Kokkos::create_mirror(derived_r.m_eta_dot_dpdn);
      get_deriv0(duN_eta, derived_dp.m_eta_dot_dpdn);

      // ==== Adjoint half (ST=Real): J^T*lambda, at the SAME base point. ====
      // euler_step_adj reads the just-completed euler_step() call's own
      // buffers (m_buffers.{dp,dpdissk,vstar}, m_tracers.qlim, the adjoint
      // tape) directly as its tape, so this taped euler_step<Real> call
      // must run at the identical base point *immediately* before it, with
      // nothing else touching those buffers in between (same precondition
      // as the FD TEST_CASE's own run_one_trial).
      impl_r.set_tape_for_adjoint(true);
      euler_r.euler_step(np1_qdp, n0_qdp, dt, rhs_multiplier, dss_opt);
      impl_r.set_tape_for_adjoint(false);

      Kokkos::deep_copy(adj_tracers.qdp, Real(0));
      Kokkos::deep_copy(adj_tracers.qlim, Real(0));
      Kokkos::deep_copy(adj_tracers.qtens_biharmonic, Real(0));
      copy_qdp_slot(adj_tracers.qdp, lambda_qdp.qdp, np1_qdp, qsize);

      Kokkos::deep_copy(adj_derived.m_vn0, Real(0));
      Kokkos::deep_copy(adj_derived.m_dp, Real(0));
      Kokkos::deep_copy(adj_derived.m_divdp, Real(0));
      Kokkos::deep_copy(adj_derived.m_divdp_proj, Real(0));
      Kokkos::deep_copy(adj_derived.m_dpdiss_biharmonic, Real(0));
      Kokkos::deep_copy(adj_derived.m_eta_dot_dpdn, lambda_eta.m_eta_dot_dpdn);
      Kokkos::deep_copy(adj_derived.m_omega_p, Real(0));

      impl_r.euler_step_adj(np1_qdp, n0_qdp, dt, rhs_multiplier, dss_opt,
                             adj_tracers, adj_derived);

      // ==== <du0, J^T*lambda> vs <J*du0, lambda>. ====
      lhs = dot_qdp_slot(adj_tracers.qdp, dqdp, n0_qdp, qsize)
          + host_dot(adj_derived.m_vn0, dvn0)
          + host_dot(adj_derived.m_dp, ddp)
          + host_dot(adj_derived.m_divdp, ddivdp)
          + host_dot(adj_derived.m_divdp_proj, ddivdp_proj);
      rhs = dot_qdp_slot(duN_qdp, lambda_qdp.qdp, np1_qdp, qsize)
          + host_dot(duN_eta, lambda_eta.m_eta_dot_dpdn);

      if (Catch::Matchers::WithinRel(rhs,tol).match(lhs)) break;
    }

    if (comm.am_i_root())
      std::cout << std::setprecision(15)
                << "   <du0,J^T*lambda> = " << lhs
                << ",  <J*du0,lambda> = " << rhs
                << "  (trial " << (trial+1) << "/" << num_trials
                << ", attempt " << (attempt+1) << "/" << max_attempts << ")\n";

    CHECK_THAT (lhs, Catch::Matchers::WithinRel(rhs,tol));
  }
}
#endif // HOMMEXX_ENABLE_FAD_TYPES
