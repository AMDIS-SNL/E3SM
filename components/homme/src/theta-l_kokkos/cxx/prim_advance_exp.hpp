#ifndef HOMME_PRIM_ADVANCE_EXP_HPP
#define HOMME_PRIM_ADVANCE_EXP_HPP

#include "CaarFunctor.hpp"
#include "DirkFunctor.hpp"
#include "Context.hpp"
#include "Elements.hpp"
#include "SimulationParams.hpp"
#include "Tape.hpp"
#include "TimeLevel.hpp"

#include "profiling.hpp"

namespace Homme
{

template<typename ST = ScalarValue>
void ttype5_timestep (const TimeLevel& tl, const Real dt, const Real eta_ave_w)
{
  using tape_t = Tape<StateSnapshot>;

  GPTLstart("ttype5_timestep");

  auto& c = Context::singleton();

  // Get elements structure
  auto& elements = c.get<ElementsST<ST>>();
  auto& params   = c.get<SimulationParams>();

  auto save = [&](int tl) {
    if (not params.store_fwd_state)
      return;

    auto& tape = std::any_cast<tape_t&>(c.any_map().at("imex_tape"));
    tape.shift_fwd();
    auto& snap = tape.curr();
    elements.m_state.take_snapshot(snap,tl,false);
  };

  // Create the functor
  auto& functor = c.get<CaarFunctorST<ST>>();

  const int nm1 = tl.nm1;
  const int n0  = tl.n0;
  const int np1 = tl.np1;
  const int qn0 = tl.n0_qdp;

  // ===================== RK STAGES ===================== //

  save(n0);

  // Stage 1: u1 = u0 + dt/5 RHS(u0),          t_rhs = t
  functor.run(RKStageData(n0, n0, nm1, qn0, dt/5.0, eta_ave_w/4.0));
  save(nm1);

  // Stage 2: u2 = u0 + dt/5 RHS(u1),          t_rhs = t + dt/5
  functor.run(RKStageData(n0, nm1, np1, qn0, dt/5.0, 0.0));
  save(np1);

  // Stage 3: u3 = u0 + dt/3 RHS(u2),          t_rhs = t + dt/5 + dt/5
  functor.run(RKStageData(n0, np1, np1, qn0, dt/3.0, 0.0));
  save(np1);

  // Stage 4: u4 = u0 + 2dt/3 RHS(u3),         t_rhs = t + dt/5 + dt/5 + dt/3
  functor.run(RKStageData(n0, np1, np1, qn0, 2.0*dt/3.0, 0.0));
  save(np1);

  // Compute (5u1-u0)/4 and store it in timelevel nm1
  {
    const auto v         = elements.m_state.m_v;
    const auto w         = elements.m_state.m_w_i;
    const auto vtheta_dp = elements.m_state.m_vtheta_dp;
    const auto phinh     = elements.m_state.m_phinh_i;
    const auto dp3d      = elements.m_state.m_dp3d;
    const auto hydrostatic_mode = params.theta_hydrostatic_mode;

    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, elements.num_elems()*NP*NP*NUM_LEV),
      KOKKOS_LAMBDA(const int it) {
        const int ie = it / (NP*NP*NUM_LEV);
        const int igp = (it / (NP*NUM_LEV)) % NP;
        const int jgp = (it / NUM_LEV) % NP;
        const int ilev = it % NUM_LEV;
        v(ie,nm1,0,igp,jgp,ilev) = (5.0*v(ie,nm1,0,igp,jgp,ilev)-v(ie,n0,0,igp,jgp,ilev))/4.0;
        v(ie,nm1,1,igp,jgp,ilev) = (5.0*v(ie,nm1,1,igp,jgp,ilev)-v(ie,n0,1,igp,jgp,ilev))/4.0;
        vtheta_dp(ie,nm1,igp,jgp,ilev) = (5.0*vtheta_dp(ie,nm1,igp,jgp,ilev)-vtheta_dp(ie,n0,igp,jgp,ilev))/4.0;
        dp3d(ie,nm1,igp,jgp,ilev) = (5.0*dp3d(ie,nm1,igp,jgp,ilev)-dp3d(ie,n0,igp,jgp,ilev))/4.0;
        if (!hydrostatic_mode) {
          w(ie,nm1,igp,jgp,ilev) = (5.0*w(ie,nm1,igp,jgp,ilev)-w(ie,n0,igp,jgp,ilev))/4.0;
          phinh(ie,nm1,igp,jgp,ilev) = (5.0*phinh(ie,nm1,igp,jgp,ilev)-phinh(ie,n0,igp,jgp,ilev))/4.0;
        }
    });
    // If NUM_LEV==NUM_LEV_P, the code above will take care also of the last interface
    if (NUM_LEV_P>NUM_LEV && !hydrostatic_mode) {
      const int LAST_INT = NUM_LEV_P-1;
      Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, elements.num_elems()*NP*NP),
        KOKKOS_LAMBDA(const int it) {
           const int ie  =  it / (NP*NP);
           const int igp = (it / NP) % NP;
           const int jgp =  it % NP;
           w(ie,nm1,igp,jgp,LAST_INT) = (5.0*w(ie,nm1,igp,jgp,LAST_INT)-w(ie,n0,igp,jgp,LAST_INT))/4.0;
      });
    }
  }
  Kokkos::fence();

  // Stage 5: u5 = (5u1-u0)/4 + 3dt/4 RHS(u4), t_rhs = t + dt/5 + dt/5 + dt/3 + 2dt/3
  functor.run(RKStageData(nm1, np1, np1, qn0, 3.0*dt/4.0, 3.0*eta_ave_w/4.0));
  save(np1);
  GPTLstop("ttype5_timestep");
}

template<typename ST = ScalarValue>
void ttype10_imex_timestep(const TimeLevel& tl, const Real dt_dyn, const Real eta_ave_w)
{
  using tape_t = Tape<StateSnapshot>;

  GPTLstart("ttype10_imex_timestep");

  // The context
  auto& c = Context::singleton();
  SimulationParams& params = c.get<SimulationParams>();

  // Get elements, hvcoord, and functors
  auto& elements = c.get<ElementsST<ST>>();
  auto& hvcoord  = c.get<HybridVCoord>();
  auto& dirk     = c.get<DirkFunctorST<ST>>();
  auto& caar     = c.get<CaarFunctorST<ST>>();

  const int qn0 = -1; // Unused in theta model RK stages
  const int nelems = elements.num_elems();
  const int nm1 = tl.nm1;
  const int n0  = tl.n0;
  const int np1 = tl.np1;

  if (params.store_fwd_state) {
    // Reset the tape so we start storing snaps at position 0
    auto& tape = std::any_cast<tape_t&>(c.any_map().at("imex_tape"));
    tape.head = -1;
  }

  auto save = [&](int tl) {
    if (not params.store_fwd_state)
      return;

    auto& tape = std::any_cast<tape_t&>(c.any_map().at("imex_tape"));
    tape.shift_fwd();
    auto& snap = tape.curr();
    elements.m_state.take_snapshot(snap,tl,false);
  };

  // ===================== IMEX STAGES ===================== //

/////////////////////
//  Time level indices
//    caar: nm1, n0, np1
//    dirk: nm1, n0, np1
/////////////////////

  // Save initial state y0 (needed by adjoint for stage 5 DIRK background)
  save(n0);

  // Stage 1
  Real dt = dt_dyn/4.0;

  caar.run(RKStageData(n0, n0, nm1, qn0, dt, 0.0, 1.0, 0.0, 1.0));
  save(nm1);
  dirk.run(nm1, 0.0, n0, 0.0, nm1, dt, elements, hvcoord);
  save(nm1);

  // Stage 2
  dt = dt_dyn/6.0;

  caar.run(RKStageData(n0, nm1, np1, qn0, dt, 0.0, 1.0, 0.0, 1.0));
  save(np1);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elements, hvcoord);
  save(np1);

  // Stage 3
  dt = 3.0*dt_dyn/8.0;

  caar.run(RKStageData(n0, np1, np1, qn0, dt, 0.0, 1.0, 0.0, 1.0));
  save(np1);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elements, hvcoord);
  save(np1);

  // Stage 4
  dt = dt_dyn/2.0;

  caar.run(RKStageData(n0, np1, np1, qn0, dt, 0.0, 1.0, 0.0, 1.0));
  save(np1);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elements, hvcoord);
  save(np1);

  // Stage 5
  Real a1 = 0.24362;
  Real a2 = 0.34184;
  Real a3 = 1-(a1+a2);
  dt = dt_dyn;

  caar.run(RKStageData(n0, np1, np1, qn0, dt, eta_ave_w, 1.0, 0.0, 1.0));
  save(np1);
  dirk.run(nm1, a2*dt, n0, a1*dt, np1, a3*dt, elements, hvcoord);
  save(np1);

  GPTLstop("ttype10_imex_timestep");
}

} // namespace Homme

#endif // HOMME_PRIM_ADVANCE_EXP_HPP
