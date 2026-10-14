#ifndef HOMMEXX_PRIM_ADVEC_TRACERS_REMAP_ADJ_HPP
#define HOMMEXX_PRIM_ADVEC_TRACERS_REMAP_ADJ_HPP

#include "ElementsDerivedState.hpp"
#include "EulerStepFunctor.hpp"
#include "Types.hpp"

namespace Homme {

// qdp at a single (implied) time level: the adjoint of the tracers does not depend on
// the (relative) time level slots the fwd code used.
using TracersQdpAdjView = ExecViewManaged<PackType<Real>*[QSIZE_D][NP][NP][NUM_LEV]>;

// Same as above, for the full qdp view (all time levels), as stored in Tracers
using TracersQdpFullView = ExecViewManaged<PackType<Real>*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>;

// Adjoint of the derived quantities the tracers step consumes from the dynamics
using TracersVn0AdjView = decltype(ElementsDerivedStateST<Real>::m_vn0);
using TracersDpAdjView  = decltype(ElementsDerivedStateST<Real>::m_dp);

// dst(ie,slot,iq,...) = src(ie,iq,...) / dst(ie,iq,...) = src(ie,slot,iq,...), for the
// first qsize tracers
void copy_qdp_to_slot (const TracersQdpFullView& dst, const int slot,
                       const TracersQdpAdjView& src,
                       const int num_elems, const int qsize);
void copy_qdp_from_slot (const TracersQdpAdjView& dst,
                         const TracersQdpFullView& src, const int slot,
                         const int num_elems, const int qsize);

// Adjoint of prim_advec_tracers_remap (src/share/cxx/prim_advec_tracers_remap.cpp), for
// the RK2 Eulerian tracers (transport_alg==0). The fwd does
//   precompute_divdp; euler_step x3 (rhs_multiplier=0,1,2); qdp_time_avg
// and depends on the dynamics only through the derived quantities vn0 and dp (the
// latter being dp3d at the beginning of the step), so the adjoint variables are
//   in :  adj_qdp = dJ/d(qdp after the step)
//   out:  adj_qdp = dJ/d(qdp before the step)
//         adj_vn0 = dJ/d(vn0)        (to be handed to prim_advance_adj, for the dynamics)
//         adj_dp  = dJ/d(derived dp) (dp is a copy of dp3d(n0) at the beginning of the step)
// (Seeds on eta_dot_dpdn/omega_p, which the Euler steps DSS but nothing here consumes, are
// not supported. They'll be needed once a vertical_remap adjoint exists.)
//
// Assumes (this is what the orchestrator of the adjoint is in charge of providing):
//  - the Context holds the state as it is when prim_advec_tracers_remap starts in the fwd:
//    the TimeLevel as it is after the last prim_advance_exp, the derived state (vn0, dp)
//    as the dynamics left it, and qdp at the n0_qdp slot as it is before the tracers step.
//    Since each euler_step overwrites what the adjoint of the next one needs, this function
//    replays the fwd steps internally; qdp(n0_qdp), vn0 and dp are left untouched, but
//    other derived fields and qdp(np1_qdp) are overwritten (as they would be by the fwd).
//  - dt is the same dt passed to prim_advec_tracers_remap in the fwd
//    (i.e., dt_tracer_factor * the dynamics dt).
//  - esf is the (set up, with buffers and boundary exchanges inited) Real-typed Euler
//    step functor of the fwd (in a production build, c.get<EulerStepFunctor>()); passing it
//    explicitly avoids tying this to whatever ScalarValue the build uses.
//  - transport_alg==0, params.qsize>0, nu_p==0, and the other restrictions of
//    euler_step_adj (limiter_option==9, single MPI rank, ...).
// Like the fwd, it updates the tracers time levels of the TimeLevel in the Context.
void prim_advec_tracers_remap_adj (EulerStepFunctorST<Real>& esf,
                                   const Real dt,
                                   TracersQdpAdjView& adj_qdp,
                                   TracersVn0AdjView& adj_vn0,
                                   TracersDpAdjView&  adj_dp);

} // namespace Homme

#endif // HOMMEXX_PRIM_ADVEC_TRACERS_REMAP_ADJ_HPP
