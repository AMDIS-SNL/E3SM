#ifndef HOMMEXX_VERTICAL_REMAP_ADJ_HPP
#define HOMMEXX_VERTICAL_REMAP_ADJ_HPP

#include "StateSnapshot.hpp"
#include "prim_advec_tracers_remap_adj.hpp"
#include "Types.hpp"

namespace Homme {

// Adjoint of dJ/d(surface pressure), same shape as StateSnapshot::ps_v
using RemapPsAdjView = ExecViewManaged<Real*[NP][NP]>;

// Adjoint of vertical_remap(dt) (src/share/cxx/vertical_remap.cpp), for vertically
// Lagrangian dynamics (rsplit>0). With rsplit>0 the fwd remap, at time levels
// (tl.np1, tl.np1_qdp) of the TimeLevel in the Context:
//   - recomputes ps_v(np1) = hyai0*ps0 + sum_k dp3d(np1,k)  (the incoming ps_v is not read)
//   - builds the target grid dp_tgt = hyai_delta*ps0 + hybi_delta*ps_v(np1)
//   - remaps the dynamics state (v, vtheta_dp, w_i, phinh_i) and the first qsize tracers
//     qdp(np1_qdp) from dp3d(np1) to dp_tgt, and then sets dp3d(np1) = dp_tgt.
// It reads nothing else that evolves in time: in particular, eta_dot_dpdn and omega_p
// (and dt) are only used by the rsplit=0 path, so no seeds on them are needed (D2 in the
// battleplan).
//
// The transposed Jacobian is obtained as in RemapFunctor::run_JtV: the Context's live
// (Real) state and qdp at (np1,np1_qdp) are copied into Dx-Fad-typed copies, seeded
// (init_J/init_J_qdp), and the Dx-Fad-typed RemapFunctor (the templated fwd code) is run
// on them. This is the fwd replay: the live Real state is NOT modified.
//
//   in : adj_state = dJ/d(state(np1) after the remap); adj_state.ps_v, if allocated,
//                    holds dJ/d(ps_v(np1) after the remap) (if not, it is taken as 0)
//        adj_qdp   = dJ/d(qdp(np1_qdp) after the remap), first qsize tracers
//   out: adj_state = dJ/d(state(np1) before the remap); adj_state.ps_v (if allocated)
//                    is set to 0, since the incoming ps_v is not consumed
//        adj_qdp   = dJ/d(qdp(np1_qdp) before the remap)
//
// Assumes (checked): ScalarValue=Real, rsplit>0, and a known remap_alg. The Context must
// hold the Real Elements, Tracers, HybridVCoord, SimulationParams and TimeLevel, with the
// state as vertical_remap sees it on entry. The Dx-Fad Elements (ElementsST<DxFadTypeRemap>,
// the same type prim_advance_adj uses for DIRK) are taken from the Context if there, or
// created there otherwise (with the geometry of the Real Elements). The Dx-Fad tracers and
// remap functor are kept in a Context scratch, rebuilt if the config changes.
void vertical_remap_adj (const Real dt,
                         StateSnapshot& adj_state,
                         TracersQdpAdjView& adj_qdp);

// Adjoint of update_q(np1_qdp,np1) (Q = qdp(np1_qdp)/dp, dp = hyai_delta*ps0 + hybi_delta*ps_v(np1)),
// for the first qsize tracers. Since Q is overwritten by update_q, and the forcing of the next
// call reads it (moist runs), its seed can come from J and/or from the adjoint of what follows.
//   in : adj_Q   = dJ/dQ (same layout as qdp at a single slot)
//   out: adj_qdp += dJ/dQ * dQ/dqdp     (dJ/d(qdp(np1_qdp)), first qsize tracers)
//        adj_ps_v += dJ/dQ * dQ/dps_v   (dJ/d(ps_v(np1)))
// The outputs are *accumulated*, so that the direct seeds of J on qdp and ps_v are preserved.
// The Context must hold the Real Tracers and Elements as update_q sees them (i.e., after the
// vertical remap); they are not modified.
void update_q_adj (const int np1_qdp, const int np1,
                   const TracersQdpAdjView& adj_Q,
                   TracersQdpAdjView& adj_qdp,
                   RemapPsAdjView& adj_ps_v);

} // namespace Homme

#endif // HOMMEXX_VERTICAL_REMAP_ADJ_HPP
