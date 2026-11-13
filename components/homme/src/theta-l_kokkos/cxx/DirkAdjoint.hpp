#ifndef HOMMEXX_DIRK_ADJOINT_HPP
#define HOMMEXX_DIRK_ADJOINT_HPP

#include "ElementsGeometry.hpp"
#include "HybridVCoord.hpp"
#include "StateSnapshot.hpp"
#include "Types.hpp"

namespace Homme {

// Adjoint (J^T*mu) of one implicit (DIRK) stage of the ttype10 IMEX scheme, via the implicit
// function theorem, and without Fad types.
//
// The stage (see DirkFunctorImpl::run_newton) computes, for each column independently, the
// w_np1 (levels 0..nlev-1; the surface one is not solved for) that is the root of
//
//   r(w) = w - ( w0 + g*dt2*(dpnh_dp(vtheta_dp,dp3d,dphi) - 1) ) ,    dphi = dphi0 + dt2*g*(w_{k+1}-w_k)
//
// (see below for the surface level), and then sets phi_np1 = phi0 + dt2*g*w_np1. Here w0 and phi0
// (and dphi0=diff(phi0)) are the sum of the np1 state before the stage, and of the terms coming
// from the states at n0 and nm1 (if alphadt_n0 and alphadt_nm1 are not 0), and the one from the
// surface-induced vertical velocity gwh(v,dp3d). The solution depends on all the levels of the
// column, so differentiating through the Newton iterations (what the dense Dx-Fad version does)
// needs one derivative per input of the column. At the converged solution w*, however,
//
//   dw*/dp = - (dr/dw)^{-1} dr/dp ,
//
// where dr/dw is the (tridiagonal) Jacobian that Newton itself uses, and dr/dp only has a
// local (3 levels) dependency. So, the adjoint needs a single transposed tridiagonal solve per
// column, and the local vector-Jacobian products of the equation of state, regardless of how many
// Newton iterations the forward run needed.
//
// The Newton iterations are not replayed: the converged solution is read from s_post.
//
// Inputs:
//  - alphadt_nm1, alphadt_n0, dt2: the coefficients passed to DirkFunctorImpl::run.
//  - s_nm1, s_n0: the states at nm1 and n0. Needed (not null) iff the corresponding
//    alphadt is not 0 (they are not used otherwise).
//  - s_pre: the state at np1 right before the stage (the 'u' state of the IMEX scheme).
//  - s_post: the state at np1 right after the stage (the 'y' state). Only w_i is used.
//  - mu: the seed, i.e., dJ/d(state after the stage).
// Outputs (all fully overwritten, for all the levels, so the old contents do not matter):
//  - out_pre: dJ/d(s_pre), i.e., J_np1^T*mu, with J_np1=d(state after)/d(s_pre).
//    For v, vtheta_dp and dp3d the Jacobian has an identity block (the stage does not change
//    them), which is included.
//  - out_n0, out_nm1: dJ/d(s_n0), dJ/d(s_nm1), if not null.
// mu and out_pre may be the same object.
//
// The formulas are those of DirkFunctorImpl (EOS, dpnh_dp, calc_jacobian, calc_gwphis): if they
// change, so must this.
void dirk_stage_adjoint_ift (const Real alphadt_nm1, const Real alphadt_n0, const Real dt2,
                             const ElementsGeometry& geo, const HybridVCoord& hvcoord,
                             const StateSnapshot* s_nm1, const StateSnapshot* s_n0,
                             const StateSnapshot& s_pre, const StateSnapshot& s_post,
                             const StateSnapshot& mu,
                             StateSnapshot& out_pre,
                             StateSnapshot* out_n0, StateSnapshot* out_nm1);

} // namespace Homme

#endif // HOMMEXX_DIRK_ADJOINT_HPP
