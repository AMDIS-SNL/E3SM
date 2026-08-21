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
void ttype10_imex_timestep(const int nm1, const int n0, const int np1,
                           const Real dt_dyn,
                           const Real eta_ave_w)
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
