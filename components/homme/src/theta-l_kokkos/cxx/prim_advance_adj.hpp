#ifndef HOMMEXX_PRIM_ADVANCE_ADJ_HPP
#define HOMMEXX_PRIM_ADVANCE_ADJ_HPP

#include "mpi/BoundaryExchange.hpp"
#include "ElementsDerivedState.hpp"
#include "StateSnapshot.hpp"
#include "Types.hpp"

#include <memory>

namespace Homme {

std::shared_ptr<BoundaryExchangeST<Real>> create_adj_bex (StateSnapshot& adj_state);

// Adjoint (dJ/d vn0) of the tracer-transport mass flux accumulated by CAAR's
// last RK stage, vn0 += eta_ave_w*v*dp3d. Same shape as
// ElementsDerivedStateST<Real>::m_vn0. This is the only derived quantity
// prim_step's tracer advection consumes from the dynamics (besides dp3d(n0)
// at the start of the step, which is handled by the prim_step adjoint itself).
using AdjVn0View = decltype(ElementsDerivedStateST<Real>::m_vn0);

// If adj_vn0 is not null, the seed dJ/d vn0 is added, at the point where the
// forward accumulated it (the input to the last CAAR stage, i.e. y4), to the
// adjoint state mu4 -- see prim_step_adj.
void ttype10_imex_adjoint(const Real dt_dyn,
                          const Real eta_ave_w,
                          StateSnapshot& adj_state,
                          const AdjVn0View* adj_vn0 = nullptr);

// Full adjoint of prim_advance_exp (theta-l_kokkos), covering the top w_i
// surface fix (if !theta_hydrostatic_mode), the hyperviscosity step (if
// hypervis_order==2 && nu>0), and the time-stepping scheme's own stages
// (dispatched on params.time_step_type, mirroring the switch in
// prim_advance_exp.cpp). Only ttype10_imex has an adjoint implemented so
// far; other schemes error out at runtime until they get one too.
//
// Assumes: !params.prescribed_wind, and that the fwd call to
// prim_advance_exp that produced the trajectory being differentiated was
// run with params.store_fwd_state=true (so that the "imex_tape" entry in
// Context holds the needed checkpoints).
//
// On entry, adj_state holds the seed (e.g. dJ/d state(np1)); on exit, it
// holds dJ/d state(n0) (the state as it was before prim_advance_exp ran).
//
// prim_advance_exp also accumulates, into the derived state, vn0 += eta_ave_w*v*dp
// (of the input to its last RK stage). If the caller has a seed dJ/d vn0 for
// the *increment* this single call added to vn0 (for a prim_step running
// dt_tracer_factor calls of prim_advance_exp, that's the same seed for all of
// them, since vn0 is just their sum), pass it as adj_vn0 and its contribution
// will be added to dJ/d state(n0). The other accumulated derived quantities
// (eta_dot_dpdn, omega_p) are NOT differentiated (nothing in prim_step
// consumes them; a future vertical_remap adjoint will).
void prim_advance_adj(const Real dt, StateSnapshot& adj_state,
                      const AdjVn0View* adj_vn0 = nullptr);

} // namespace Homme

#endif // HOMMEXX_PRIM_ADVANCE_ADJ_HPP
