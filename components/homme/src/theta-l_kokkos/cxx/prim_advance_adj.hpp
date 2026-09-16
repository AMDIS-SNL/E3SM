#ifndef HOMMEXX_PRIM_ADVANCE_ADJ_HPP
#define HOMMEXX_PRIM_ADVANCE_ADJ_HPP

#include "mpi/BoundaryExchange.hpp"
#include "StateSnapshot.hpp"
#include "Types.hpp"

#include <memory>

namespace Homme {

std::shared_ptr<BoundaryExchangeST<Real>> create_adj_bex (StateSnapshot& adj_state);

void ttype10_imex_adjoint(const Real dt_dyn,
                          const Real eta_ave_w,
                          StateSnapshot& adj_state);

// Full adjoint of prim_advance_exp (theta-l_kokkos), covering the top w_i
// surface fix (if !theta_hydrostatic_mode), the hyperviscosity step (if
// hypervis_order==2 && nu>0), and the ttype10 IMEX CAAR/DIRK stages.
//
// Assumes: params.time_step_type==ttype10_imex, !params.prescribed_wind,
// and that the fwd call to prim_advance_exp that produced the trajectory
// being differentiated was run with params.store_fwd_state=true (so that
// the "imex_tape" entry in Context holds the needed checkpoints).
//
// On entry, adj_state holds the seed (e.g. dJ/d state(np1)); on exit, it
// holds dJ/d state(n0) (the state as it was before prim_advance_exp ran).
void prim_advance_adj(const Real dt, StateSnapshot& adj_state);

} // namespace Homme

#endif // HOMMEXX_PRIM_ADVANCE_ADJ_HPP
