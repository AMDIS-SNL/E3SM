#include <catch2/catch.hpp>

#include "Types.hpp"
#include "thetal_ut_session.hpp"
#include "thetal_ut_utils.hpp"
#include "Context.hpp"
#include "Elements.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HommexxEnums.hpp"
#include "HybridVCoord.hpp"
#include "PhysicalConstants.hpp"
#include "PpmRemap.hpp"
#include "RemapFunctor.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "Tracers.hpp"
#include "VerticalRemapManager.hpp"
#include "update_q.hpp"
#include "vertical_remap_adj.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_comm.hpp>
#include <ekat_pack_kokkos.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <utility>

#ifdef HOMMEXX_ENABLE_FAD_TYPES

namespace Homme
{

namespace {

// The dot product over the whole (physical) dynamics state
double dot_state (const StateSnapshot& x, const StateSnapshot& y) {
  constexpr int nlevs = NUM_PHYSICAL_LEV;
  return dot(ekat::scalarize(x.v),        ekat::scalarize(y.v),        nlevs)
       + dot(ekat::scalarize(x.vtheta_dp),ekat::scalarize(y.vtheta_dp),nlevs)
       + dot(ekat::scalarize(x.dp3d),     ekat::scalarize(y.dp3d),     nlevs)
       + dot(ekat::scalarize(x.w_i),      ekat::scalarize(y.w_i),      nlevs+1)
       + dot(ekat::scalarize(x.phinh_i),  ekat::scalarize(y.phinh_i),  nlevs+1);
}

// sum_{ie,iq<qsize,i,j,lev} a(ie,iq,i,j,lev)*b(ie,iq,i,j,lev), for two views of Packs with
// layout (ie,iq,i,j,lev), whose 2nd extent may differ (e.g., Tracers::Q and a single qdp slot)
template<typename View1, typename View2>
double dot_tracers (const View1& a, const View2& b, const int qsize) {
  auto ah = Kokkos::create_mirror(a); Kokkos::deep_copy(ah,a);
  auto bh = Kokkos::create_mirror(b); Kokkos::deep_copy(bh,b);
  double s = 0;
  for (int ie=0; ie<ah.extent_int(0); ++ie)
    for (int iq=0; iq<qsize; ++iq)
      for (int i=0; i<NP; ++i)
        for (int j=0; j<NP; ++j)
          for (int lev=0; lev<NUM_PHYSICAL_LEV; ++lev) {
            const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
            s += ah(ie,iq,i,j,vpi)[vsi] * bh(ie,iq,i,j,vpi)[vsi];
          }
  return s;
}

// max_i |a_i - b_i| / max(1,|a_i|), over the whole span of two same-shaped Real views
template<typename View>
double max_rel_diff (const View& a, const View& b) {
  auto ah = Kokkos::create_mirror(a); Kokkos::deep_copy(ah,a);
  auto bh = Kokkos::create_mirror(b); Kokkos::deep_copy(bh,b);
  auto as = ekat::scalarize(ah);
  auto bs = ekat::scalarize(bh);
  double d = 0;
  for (size_t i=0; i<as.span(); ++i) {
    const double x = as.data()[i], y = bs.data()[i];
    d = std::max(d, std::abs(x-y)/std::max(1.0,std::abs(x)));
  }
  return d;
}

// As above, for the first qsize tracers of qdp at the given slot, physical levels only
template<typename View>
double max_rel_diff_qdp (const View& a, const View& b, const int slot, const int qsize) {
  auto ah = Kokkos::create_mirror(a); Kokkos::deep_copy(ah,a);
  auto bh = Kokkos::create_mirror(b); Kokkos::deep_copy(bh,b);
  double d = 0;
  for (int ie=0; ie<ah.extent_int(0); ++ie)
    for (int iq=0; iq<qsize; ++iq)
      for (int i=0; i<NP; ++i)
        for (int j=0; j<NP; ++j)
          for (int lev=0; lev<NUM_PHYSICAL_LEV; ++lev) {
            const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
            const double x = ah(ie,slot,iq,i,j,vpi)[vsi], y = bh(ie,slot,iq,i,j,vpi)[vsi];
            d = std::max(d, std::abs(x-y)/std::max(1.0,std::abs(x)));
          }
  return d;
}

// As above, for the first qsize tracers of a (ie,iq,i,j,lev) view (e.g., Tracers::Q), physical levels only
template<typename View>
double max_rel_diff_tracers (const View& a, const View& b, const int qsize) {
  auto ah = Kokkos::create_mirror(a); Kokkos::deep_copy(ah,a);
  auto bh = Kokkos::create_mirror(b); Kokkos::deep_copy(bh,b);
  double d = 0;
  for (int ie=0; ie<ah.extent_int(0); ++ie)
    for (int iq=0; iq<qsize; ++iq)
      for (int i=0; i<NP; ++i)
        for (int j=0; j<NP; ++j)
          for (int lev=0; lev<NUM_PHYSICAL_LEV; ++lev) {
            const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
            const double x = ah(ie,iq,i,j,vpi)[vsi], y = bh(ie,iq,i,j,vpi)[vsi];
            d = std::max(d, std::abs(x-y)/std::max(1.0,std::abs(x)));
          }
  return d;
}

// Bitwise equality of two same-shaped Real views (of Packs or not)
template<typename View>
bool same_values (const View& a, const View& b) {
  auto ah = Kokkos::create_mirror(a); Kokkos::deep_copy(ah,a);
  auto bh = Kokkos::create_mirror(b); Kokkos::deep_copy(bh,b);
  auto as = ekat::scalarize(ah);
  auto bs = ekat::scalarize(bh);
  for (size_t i=0; i<as.span(); ++i)
    if (as.data()[i]!=bs.data()[i]) return false;
  return true;
}

// Bitwise equality of two snapshots (incl. ps_v)
bool same_state (const StateSnapshot& x, const StateSnapshot& y) {
  auto eq = [](const auto& a, const auto& b) { return same_values(a,b); };
  return eq(x.v,y.v) and eq(x.vtheta_dp,y.vtheta_dp) and eq(x.dp3d,y.dp3d) and
         eq(x.w_i,y.w_i) and eq(x.phinh_i,y.phinh_i) and eq(x.ps_v,y.ps_v);
}

// Run the DpFad-typed fwd remap, i.e. the templated fwd code (the ground truth for the tangent)
template<typename BCType>
void run_remap_dp (const int qsize, const ElementsST<DpFadType>& elems,
                   const TracersST<DpFadType>& tracers, const HybridVCoord& hvcoord,
                   const int np1, const int np1_qdp, const Real dt)
{
  using alg_t = Remap::Ppm::PpmVertRemap<BCType>;
  Remap::RemapFunctor<true,alg_t,DpFadType> remap(qsize,elems,tracers,hvcoord);
  FunctorsBuffersManager fbm;
  fbm.request_size(remap.requested_buffer_size());
  fbm.allocate();
  remap.init_buffers(fbm);
  remap.run_remap(np1,np1_qdp,dt);
  Kokkos::fence();
}

} // anonymous namespace

// Adjoint test for vertical_remap (rsplit>0) followed by update_q, via the dot-product test
//   <J*du, lambda> == <du, J^T*lambda>
// where
//  - du is a random tangent on the remap inputs: the dynamics state at np1 (incl. dp3d) and
//    qdp(np1_qdp);
//  - J*du is computed by the DpFad-typed fwd code (RemapFunctor<true,Alg,DpFadType>, then
//    update_q_impl<DpFadType>), and consists of the tangent of the dynamics state, ps_v(np1),
//    qdp(np1_qdp) and Q;
//  - J^T*lambda is computed by update_q_adj and vertical_remap_adj, on the Real objects in the
//    Context (as in an actual run).
// The seeds on state, ps_v, qdp and Q are first checked separately, then all together.
// This also checks that vertical_remap_adj leaves the live Real state untouched, that the
// Real and DpFad fwd remaps agree, and reports the cost of the adjoint relative to the fwd.
TEST_CASE("vertical_remap_adjoint")
{
  constexpr int ne = 2;
  constexpr int qsize = 3;
  static_assert(qsize<=QSIZE_D, "Not enough tracers in this build.\n");

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
  auto& comm = c.create<ekat::Comm>(MPI_COMM_WORLD);

  auto& params = c.create<SimulationParams>();
  params.params_set = true;
  params.qsize = qsize;
  params.rsplit = 2;
  params.transport_alg = 0;
  params.scale_factor = PhysicalConstants::rearth0;
  params.laplacian_rigid_factor = PhysicalConstants::rrearth0;

  // A smooth hvcoord, so that layer thicknesses are well behaved (see remap_sacado_ut.cpp)
  ThetalUnitTestSession session(ne, seed, ThetalUnitTestSession::VCoord::Smooth);
  auto& hvcoord = session.hvcoord;
  const int num_elems = session.num_elems();
  const auto max_pressure = 1000.0 + hvcoord.ps0; // This ensures max_p > ps0

  // Random geometry: in particular, gradphis!=0 exercises the w_i surface b.c. of the remap
  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems, true, PhysicalConstants::rearth0);
  geo.randomize(seed);

  // Real objects in the Context (what the adjoint works with, as in a run). Note: we do
  // *not* create the Dx-Fad Elements, so that vertical_remap_adj has to create it in the
  // first iteration below, and reuse it in the following ones.
  auto& elems_r = c.create<ElementsST<Real>>();
  elems_r.init(num_elems, true, PhysicalConstants::rearth0);
  elems_r.m_geometry = geo;
  auto& tracers_r = c.create<TracersST<Real>>();
  tracers_r.init(num_elems,qsize);

  // DpFad objects, for the fwd tangent
  ElementsST<DpFadType> elems_dp;
  elems_dp.init(num_elems, true, PhysicalConstants::rearth0);
  elems_dp.m_geometry = geo;
  TracersST<DpFadType> tracers_dp;
  tracers_dp.init(num_elems,qsize);

  // Time levels: use a nonzero qdp slot, to make sure the slots are handled right
  auto& tl = c.create<TimeLevel>();
  tl.nm1 = 0;
  tl.n0  = 1;
  tl.np1 = 2;
  tl.n0_qdp  = 0;
  tl.np1_qdp = 1;
  const int np1 = tl.np1;
  const int np1_qdp = tl.np1_qdp;

  constexpr auto tol = std::numeric_limits<double>::epsilon()*1e4;

  for (const auto alg : {RemapAlg::PPM_MIRRORED, RemapAlg::PPM_LIMITED_EXTRAP}) {
    for (const bool hydrostatic : {true, false}) {
      const std::string cfg = std::string("remap_alg=") + remapAlg2str(alg) +
                              (hydrostatic ? ", hydrostatic" : ", non-hydrostatic");
      if (comm.am_i_root()) std::cout << " -> " << cfg << "\n";

      params.remap_alg = alg;
      params.theta_hydrostatic_mode = hydrostatic;

      // dt is not used by the rsplit>0 remap, but pass something meaningful anyways
      Real dt = RPDF(1.0,10.0)(engine);
      MPI_Bcast(&dt,1,MPI_DOUBLE,0,comm.mpi_comm());

      // ===================== Fwd inputs ===================== //
      elems_r.m_state.randomize(seed, max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, geo.m_phis);
      tracers_r.randomize(seed, 1.0, 10.0);
      const auto state0 = elems_r.m_state.take_snapshot(np1,true);
      decltype(tracers_r.qdp) qdp0("qdp0",num_elems);
      Kokkos::deep_copy(qdp0,tracers_r.qdp);

      // ===================== Fwd tangent (DpFad) ===================== //
      elems_dp.m_state.import_values(elems_r.m_state,np1);
      elems_dp.m_state.randomize_derivs(seed+1,np1);
      const auto du = elems_dp.m_state.take_deriv_snapshot(np1,0);

      tracers_dp.import_values(tracers_r,np1_qdp);
      auto dqdp = Kokkos::create_mirror(tracers_r.qdp);
      genRandArray(dqdp, engine, RPDF(-1.0,1.0));
      ut::set_deriv0(tracers_dp.qdp, dqdp);

      if (alg==RemapAlg::PPM_MIRRORED) {
        run_remap_dp<Remap::Ppm::PpmMirrored<DpFadType>>(qsize,elems_dp,tracers_dp,hvcoord,np1,np1_qdp,dt);
      } else {
        run_remap_dp<Remap::Ppm::PpmLimitedExtrap<DpFadType>>(qsize,elems_dp,tracers_dp,hvcoord,np1,np1_qdp,dt);
      }
      update_q_impl(elems_dp.m_state,tracers_dp,hvcoord,np1_qdp,np1,qsize);
      Kokkos::fence();

      const auto duN = elems_dp.m_state.take_deriv_snapshot(np1,0,true);
      auto dqdpN = Kokkos::create_mirror(tracers_r.qdp);
      ut::get_deriv0(dqdpN, tracers_dp.qdp);
      auto dQN = Kokkos::create_mirror(tracers_r.Q);
      ut::get_deriv0(dQN, tracers_dp.Q);

      // ===================== Real fwd ===================== //
      // update_q_adj needs the state after the remap. This is what vertical_remap and
      // update_q do in prim_run_subcycle_c (with the VerticalRemapManager in the Context).
      VerticalRemapManagerST<Real> vrm;
      FunctorsBuffersManager fbm;
      fbm.request_size(vrm.requested_buffer_size());
      fbm.allocate();
      vrm.init_buffers(fbm);
      vrm.run_remap(np1,np1_qdp,dt);
      update_q(np1_qdp,np1);
      Kokkos::fence();
      const auto stateN = elems_r.m_state.take_snapshot(np1,true);
      decltype(tracers_r.qdp) qdpN("qdpN",num_elems);
      Kokkos::deep_copy(qdpN,tracers_r.qdp);

      // Sanity check: the Real and DpFad fwd agree (on the remapped time level/qdp slot)
      {
        ElementsStateST<Real> vals_dp;
        vals_dp.init(num_elems);
        vals_dp.import_values(elems_dp.m_state,np1);
        const auto stateN_dp = vals_dp.take_snapshot(np1,true);
        TracersST<Real> tvals_dp(num_elems,qsize);
        tvals_dp.import_values(tracers_dp,np1_qdp);
        const double diffs[8] = {max_rel_diff(stateN.v,        stateN_dp.v),
                                 max_rel_diff(stateN.vtheta_dp,stateN_dp.vtheta_dp),
                                 max_rel_diff(stateN.dp3d,     stateN_dp.dp3d),
                                 max_rel_diff(stateN.w_i,      stateN_dp.w_i),
                                 max_rel_diff(stateN.phinh_i,  stateN_dp.phinh_i),
                                 max_rel_diff(stateN.ps_v,     stateN_dp.ps_v),
                                 max_rel_diff_qdp(qdpN,tvals_dp.qdp,np1_qdp,qsize),
                                 max_rel_diff_tracers(tracers_r.Q,tvals_dp.Q,qsize)};
        const double d = *std::max_element(diffs,diffs+8);
        if (comm.am_i_root()) {
          std::cout << "   Real vs DpFad fwd, max rel diff (v,vtheta_dp,dp3d,w_i,phinh_i,ps_v,qdp,Q):";
          for (auto x : diffs) std::cout << " " << x;
          std::cout << "\n";
        }
        CHECK (d < 1e-12);
      }

      // ===================== Adjoint ===================== //
      // Runs update_q_adj and vertical_remap_adj on the given seeds (not modified), and
      // returns the state and qdp parts of <du, J^T*lambda>
      auto run_adjoint = [&](const StateSnapshot& lam, const TracersQdpAdjView& lam_qdp,
                             const TracersQdpAdjView& lam_Q) {
        auto a = lam.clone(true);
        TracersQdpAdjView a_qdp("a_qdp",num_elems);
        Kokkos::deep_copy(a_qdp,lam_qdp);

        // update_q_adj sees the state after the remap...
        elems_r.m_state.import_snapshot(stateN,np1,true);
        Kokkos::deep_copy(tracers_r.qdp,qdpN);
        update_q_adj(np1_qdp,np1,lam_Q,a_qdp,a.ps_v);

        // ...and vertical_remap_adj the one before
        elems_r.m_state.import_snapshot(state0,np1,true);
        Kokkos::deep_copy(tracers_r.qdp,qdp0);
        vertical_remap_adj(dt,a,a_qdp);

        // The live state is left untouched, and the input ps_v has no adjoint
        const auto state_after = elems_r.m_state.take_snapshot(np1,true);
        CHECK (same_state(state_after,state0));
        CHECK (same_values(tracers_r.qdp,qdp0));
        auto a_ps_h = Kokkos::create_mirror(a.ps_v);
        Kokkos::deep_copy(a_ps_h,a.ps_v);
        bool zero_ps = true;
        for (size_t i=0; i<a_ps_h.span(); ++i) zero_ps = zero_ps and a_ps_h.data()[i]==0;
        CHECK (zero_ps);

        return std::make_pair(dot_state(du,a), ut::dot_qdp(a_qdp,dqdp,np1_qdp,qsize));
      };

      auto check = [&](const bool seed_state, const bool seed_ps, const bool seed_qdp,
                       const bool seed_Q, const std::string& name) {
        StateSnapshot lam(num_elems,true);
        lam.zero();
        if (seed_state) {
          genRandArray(lam.v,        engine, RPDF(-1.0,1.0));
          genRandArray(lam.vtheta_dp,engine, RPDF(-1.0,1.0));
          genRandArray(lam.dp3d,     engine, RPDF(-1.0,1.0));
          genRandArray(lam.w_i,      engine, RPDF(-1.0,1.0));
          genRandArray(lam.phinh_i,  engine, RPDF(-1.0,1.0));
        }
        if (seed_ps) {
          genRandArray(lam.ps_v, engine, RPDF(-1.0,1.0));
        }
        TracersQdpAdjView lam_qdp("lam_qdp",num_elems);
        TracersQdpAdjView lam_Q("lam_Q",num_elems);
        if (seed_qdp) genRandArray(lam_qdp, engine, RPDF(-1.0,1.0));
        if (seed_Q)   genRandArray(lam_Q,   engine, RPDF(-1.0,1.0));

        const auto dots0 = run_adjoint(lam,lam_qdp,lam_Q);
        const double dot0 = dots0.first + dots0.second;

        const double dotN_state = dot_state(duN,lam);
        const double dotN_ps    = dot(duN.ps_v,lam.ps_v);
        const double dotN_qdp   = ut::dot_qdp(lam_qdp,dqdpN,np1_qdp,qsize);
        const double dotN_Q     = dot_tracers(lam_Q,dQN,qsize);
        const double dotN = dotN_state + dotN_ps + dotN_qdp + dotN_Q;

        if (comm.am_i_root())
          std::cout << std::setprecision(15)
                    << "   [" << name << "]\n"
                    << "     <du, J^T*lambda> = " << dot0
                    << " (state: " << dots0.first << ", qdp: " << dots0.second << ")\n"
                    << "     <J*du, lambda>   = " << dotN
                    << " (state: " << dotN_state << ", ps: " << dotN_ps
                    << ", qdp: " << dotN_qdp << ", Q: " << dotN_Q << ")\n";

        using namespace Catch::Matchers;
        CHECK (dotN != 0);
        CHECK_THAT (dot0, WithinRel(dotN,tol));
        return dots0;
      };

      check(true,  false, false, false, "state seed");
      check(false, true,  false, false, "ps_v seed");
      const auto qdp_dots = check(false, false, true,  false, "qdp seed");
      check(false, false, false, true,  "Q seed");
      check(true,  true,  true,  true,  "all seeds");

      // The qdp seed must reach the dynamics, through the dependence of the grids on dp3d
      CHECK (qdp_dots.first != 0);

      // ===================== Cost ===================== //
      {
        constexpr int nrep = 5;
        StateSnapshot lam(num_elems,true);
        lam.zero();
        TracersQdpAdjView lam_qdp("lam_qdp",num_elems);
        using clock = std::chrono::steady_clock;
        double t_fwd = 0, t_adj = 0;
        for (int r=0; r<nrep; ++r) {
          elems_r.m_state.import_snapshot(state0,np1,true);
          Kokkos::deep_copy(tracers_r.qdp,qdp0);
          Kokkos::fence();
          auto t0 = clock::now();
          vrm.run_remap(np1,np1_qdp,dt);
          Kokkos::fence();
          auto t1 = clock::now();
          t_fwd += std::chrono::duration<double>(t1-t0).count();
        }
        elems_r.m_state.import_snapshot(state0,np1,true);
        Kokkos::deep_copy(tracers_r.qdp,qdp0);
        for (int r=0; r<nrep; ++r) {
          Kokkos::fence();
          auto t0 = clock::now();
          vertical_remap_adj(dt,lam,lam_qdp);
          Kokkos::fence();
          auto t1 = clock::now();
          t_adj += std::chrono::duration<double>(t1-t0).count();
        }
        if (comm.am_i_root())
          std::cout << "   cost (nlev=" << NUM_PHYSICAL_LEV << ", qsize=" << qsize
                    << ", nelem=" << num_elems << ", Dx-Fad size=" << DerivSz<DxFadTypeRemap>::value << "):"
                    << " fwd " << 1e3*t_fwd/nrep << " ms, adj " << 1e3*t_adj/nrep
                    << " ms, ratio " << t_adj/t_fwd << "\n";
      }
    }
  }
}

} // namespace Homme

#endif // HOMMEXX_ENABLE_FAD_TYPES
