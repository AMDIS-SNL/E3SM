/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_EULER_STEP_ADJOINT_TAPE_HPP
#define HOMMEXX_EULER_STEP_ADJOINT_TAPE_HPP

#include "Dimensions.hpp"
#include "Types.hpp"
#include "mpi/BoundaryExchange.hpp"

namespace Homme {

// Real-only scratch/tape storage for EulerStepFunctorImplST<ST>::euler_step_adj
// (the adjoint of euler_step, limiter_option==9 only). This used to be a set
// of plain data members of EulerStepFunctorImplST<ST>, but that meant every
// instantiation of the class template -- including ST==DpFadType,
// ST==DxFadTypeCaar, etc., none of which ever touch the adjoint -- carried
// this (unused, unallocated) state. It is now held once, lazily, in
// Context::singleton().any_map() (key "euler_step_adjoint_tape"), mirroring
// how Tape<StateSnapshot> is stashed there under "imex_tape" for the
// CAAR/DIRK adjoint (see Tape.hpp, prim_advance_exp.hpp,
// prim_advance_adj.cpp). Unlike Tape<T>, which is a generic push/pop-style
// stack container, this is just a flat bundle of scratch Views plus the
// lazily-built boundary-exchange objects used by the adjoint's self-adjoint
// DSS exchanges -- so it is its own plain struct, not built on Tape<T>.
//
// Single-consumer assumption: like Tape<StateSnapshot>'s own implicit
// single-consumer assumption, this struct assumes there is only ever one
// live "adjoint session" using the Euler-step adjoint at a time. In
// particular, `bes`/`mmqb_be` are lazily built on the first
// euler_step_adj() call and then bind, for the lifetime of the program (or
// until the Context entry is reallocated by a qsize/num-elems change), to
// the specific adj_tracers/adj_derived field Views passed on that first
// call -- this was already a documented precondition on
// EulerStepFunctorImplST::euler_step_adj before this refactor (the caller
// must keep reusing the SAME adj_tracers/adj_derived instances across
// calls). Before this refactor, `bes`/`mmqb_be` were private members of a
// single EulerStepFunctorImplST<Real> instance, so the precondition's scope
// was implicitly "across repeated calls on that one instance". Now that
// they live in a Context-wide struct shared by ALL EulerStepFunctorImplST<
// Real> instances, the precondition's scope is implicitly widened to
// "across repeated calls on ANY EulerStepFunctorImplST<Real> instance in
// the program" -- if two different instances both called euler_step_adj
// with different adj_tracers/adj_derived objects, the second call would
// silently reuse the first call's stale, already-bound BE objects. This is
// not believed to be a new *practical* hazard: EulerStepFunctorST<Real> (and
// the underlying implementation it wraps) is itself accessed as a
// Context-managed singleton-per-type (see Context::get<EulerStepFunctorST<
// Real>>()), so there is, in practice, only ever one live
// EulerStepFunctorImplST<Real> touching the adjoint at a time -- exactly the
// same single-consumer assumption Tape<StateSnapshot>/"imex_tape" already
// relies on. Flagging it here explicitly rather than silently papering over
// it, per the refactor's own instructions.
struct EulerStepAdjointTape {
  using RPT = PackType<Real>;

  EulerStepAdjointTape (int ne, int qsize)
    : qtens_prelimiter        ("euler_step_adj qtens prelimiter tape",  ne, qsize)
    , qlim_pre_local          ("euler_step_adj qlim pre-local tape",    ne, qsize)
    , qlim_pre_exchange       ("euler_step_adj qlim pre-exchange tape", ne, qsize)
    , qlim_final              ("euler_step_adj qlim final tape",        ne, qsize)
    , dp                      ("euler_step_adj dp accum",       ne)
    , dpdissk                  ("euler_step_adj dpdissk accum",  ne)
    , vstar                    ("euler_step_adj vstar accum",    ne)
    , qlim_final_grad          ("euler_step_adj qlim final grad",        ne, qsize)
    , qlim_final_grad_summed   ("euler_step_adj qlim final grad summed", ne, qsize)
  {}

  // Taped forward-pass data (recorded by compute_qmin_qmax/run_tracer_phase
  // when m_tape_for_adjoint is set), consumed by euler_step_adj:
  //  - qtens_prelimiter: compute_qtens's raw output, per (elem, tracer, i,
  //    j, level), before limiter_clip_and_sum mutates it in place.
  //  - qlim_pre_local: qlim as it stood on entry to compute_qmin_qmax
  //    (matters only for rhs_multiplier==1, where the running min/max is
  //    seeded from a *previous* euler_step call's qlim).
  //  - qlim_pre_exchange: qlim right after compute_qmin_qmax's own
  //    per-element reduction, before any neighbor/MPI min-max exchange.
  //  - qlim_final: qlim as it enters limiter_clip_and_sum (i.e. after
  //    whichever neighbor exchange ran, or unchanged if none did), before
  //    with_limiter_shell's own 0-floor/mass-relaxation adjustments.
  ExecViewManaged<RPT**[NP][NP][NUM_LEV]>    qtens_prelimiter;
  ExecViewManaged<RPT*[QSIZE_D][2][NUM_LEV]> qlim_pre_local;
  ExecViewManaged<RPT*[QSIZE_D][2][NUM_LEV]> qlim_pre_exchange;
  ExecViewManaged<RPT*[QSIZE_D][2][NUM_LEV]> qlim_final;

  // Adjoint-only scratch (zeroed and populated fresh by every
  // euler_step_adj call): running accumulators for quantities that are
  // shared across the qsize tracer loop (dp, dpdissk, vstar are all
  // per-element only, not per-tracer, in the forward code), and the
  // group-level dJ/d(qlim_final) produced by the limiter's reverse pass.
  ExecViewManaged<RPT*[NP][NP][NUM_LEV]>    dp;
  ExecViewManaged<RPT*[NP][NP][NUM_LEV]>    dpdissk;
  ExecViewManaged<RPT*[2][NP][NP][NUM_LEV]> vstar;
  ExecViewManaged<RPT*[QSIZE_D][2][NUM_LEV]> qlim_final_grad;
  // qlim_final_grad, after being summed across each element's connectivity
  // group (adjoint of exchange_min_max's implicit multi-way min/max
  // broadcast -- see euler_step_adj stage 2 for the derivation). Only
  // meaningful (and only computed) when has_exchange; single-rank only,
  // see there.
  ExecViewManaged<RPT*[QSIZE_D][2][NUM_LEV]> qlim_final_grad_summed;

  // Boundary-exchange objects for the DSS-type (non-min/max) exchanges,
  // registered against the *adjoint* tracers/derived-state fields passed to
  // euler_step_adj, mirroring EulerStepFunctorImplST's own m_bes/m_mmqb_be
  // but pointed at adjoint storage. Self-adjoint DSS exchanges (see
  // euler_step_adj) are applied directly to the adjoint fields via these.
  // Lazily built by EulerStepFunctorImplST::init_adjoint_boundary_exchanges
  // the first time euler_step_adj is called; NOTE: they bind to the
  // specific adj_tracers/adj_derived objects passed on that first call, so
  // the caller must keep reusing the *same* adj_tracers/adj_derived
  // instances across calls (documented precondition, akin to run_JtV's "x
  // and y must be different objects" -- see the single-consumer comment
  // above for how this scope changed with this struct's move into Context).
  Kokkos::Array<std::shared_ptr<BoundaryExchangeST<Real>>, 3*Q_NUM_TIME_LEVELS> bes;
  std::shared_ptr<BoundaryExchangeST<Real>> mmqb_be;
  bool bex_ready = false;
};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_ADJOINT_TAPE_HPP
