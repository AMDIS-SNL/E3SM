#ifndef HOMMEXX_PRIM_STEP_ADJ_HPP
#define HOMMEXX_PRIM_STEP_ADJ_HPP

#include "Elements.hpp"
#include "SimulationParams.hpp"
#include "StateSnapshot.hpp"
#include "TimeLevel.hpp"
#include "Types.hpp"

namespace Homme {

// Adjoint of prim_step (src/share/cxx/prim_step.cpp), the *Eulerian* one (NOT
// prim_step_flexible, which is for semi-Lagrangian transport). prim_step runs:
//
//   (0) set_tracer_transport_derived_values:  derived.{vn0,eta_dot_dpdn,omega_p}=0,
//                                             derived.dp = dp3d(n0)
//   (1) dt_tracer_factor (=K) times:          prim_advance_exp   [ accumulates
//                                             vn0 += eta_ave_w*v*dp of its last RK stage ]
//   (2) prim_advec_tracers_remap (RK2):       precompute_divdp (divdp=div(vn0)),
//                                             3 euler_steps, qdp_time_avg
//
// There is NO vertical remap in prim_step (that is done by prim_run_subcycle_c,
// after prim_step), so this adjoint does not need one.
//
// The dynamics affect the tracers only through vn0 and dp3d(n0) (the derived
// eta_dot_dpdn/omega_p are DSS'd by the Euler steps, but nothing in prim_step
// consumes them), so the seeds/results of this adjoint are just the dynamics
// state and qdp:
//   - on entry: adj_state = dJ/d(dyn state after the K-th prim_advance_exp),
//               adj_qdp   = dJ/d(qdp after qdp_time_avg)
//   - on exit:  adj_state = dJ/d(dyn state before the 1st prim_advance_exp),
//               adj_qdp   = dJ/d(qdp before the 1st euler_step)
// (Seeds on eta_dot_dpdn/omega_p, needed once a vertical_remap adjoint exists,
// are not supported yet.)
//
// Since neither the in-memory imex Tape nor the Euler-step adjoint tape can hold
// more than a single prim_advance_exp/euler_step call, the adjoint recomputes
// the forward trajectory from a checkpoint of the state at the start of prim_step
// (PrimStepCheckpoint below, a coarse-grained checkpoint that a disk-based
// CheckpointStore could later provide). The live Elements/Tracers/TimeLevel in
// Context are clobbered, and are restored to the checkpoint on exit.

// Time levels as they stand at the end of a prim_step that started at tl_start.
// In particular, .n0_qdp is the qdp slot that prim_step reads its tracers from,
// and .np1_qdp the one where it writes them.
TimeLevel prim_step_final_time_level (const TimeLevel& tl_start,
                                      const int dt_tracer_factor);

// qdp at a single (implied) time level
using PrimStepQdpView = ExecViewManaged<PackType<Real>*[QSIZE_D][NP][NP][NUM_LEV]>;

// Everything needed to replay a prim_step from its beginning, for the objects
// in Context (Elements, Tracers, TimeLevel; theta-l, ScalarValue=Real only).
struct PrimStepCheckpoint {
  PrimStepCheckpoint (const int num_elems, const int qsize);

  // Capture the live state, as the start of the prim_step about to be run
  void take ();

  // Overwrite the live state (and TimeLevel) with the captured one
  void restore () const;

  TimeLevel         tl = {};   // TimeLevel at the start of prim_step
  StateSnapshot     state;     // Dynamics state, at tl.n0
  PrimStepQdpView   qdp;       // Tracers, at the qdp slot prim_step reads them from
  int               qsize;
};

// Computes the adjoint of ONE prim_step(dt), starting from the checkpoint of the
// state at its beginning. See the top-of-file comment for what adj_state and
// adj_qdp hold on entry and exit.
//
// Assumes (checked): ScalarValue=Real, params.time_step_type==ttype10_imex (the
// only time stepping scheme with an adjoint), transport_alg==0, limiter_option==9,
// nu_p==0, hypervis_scaling==0 (the Euler-step adjoint's restrictions), single MPI
// rank. The Context must hold everything needed both by the fwd code (Elements,
// CaarFunctor, DirkFunctor, LimiterFunctor, HyperviscosityFunctor, EulerStepFunctor,
// Tracers, TimeLevel, ...) and by prim_advance_adj (the Dx-Fad-typed Caar/Dirk stacks).
void prim_step_adj (const Real dt,
                    const PrimStepCheckpoint& ckpt,
                    StateSnapshot& adj_state,
                    PrimStepQdpView& adj_qdp);

// This mirrors set_tracer_transport_derived_values() in prim_step.cpp (which is
// file-static there), for any scalar type (so that tests can run it on
// Fad-typed elements). Keep the two in sync.
template<typename ST>
void set_tracer_transport_derived_values_st (const SimulationParams& params,
                                             const ElementsST<ST>& elements,
                                             const TimeLevel& tl)
{
  const auto eta_dot_dpdn = elements.m_derived.m_eta_dot_dpdn;
  const auto derived_vn0 = elements.m_derived.m_vn0;
  const auto omega_p = elements.m_derived.m_omega_p;
  const auto derived_dpdiss_ave = elements.m_derived.m_dpdiss_ave;
  const auto derived_dpdiss_biharmonic = elements.m_derived.m_dpdiss_biharmonic;
  const auto derived_dp = elements.m_derived.m_dp;
  const auto dp3d = elements.m_state.m_dp3d;
  const auto vstar = elements.m_derived.m_vstar;
  const auto v = elements.m_state.m_v;
  const auto n0 = tl.n0;
  const bool nu_p_pos = params.nu_p>0;
  const bool transport_alg_pos = params.transport_alg>0;
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace> (0,elements.num_elems()*NP*NP*NUM_LEV),
                       KOKKOS_LAMBDA(const int idx) {
    const int ie   = ((idx / NUM_LEV) / NP) / NP;
    const int igp  = ((idx / NUM_LEV) / NP) % NP;
    const int jgp  =  (idx / NUM_LEV) % NP;
    const int ilev =   idx % NUM_LEV;
    eta_dot_dpdn(ie,igp,jgp,ilev) = 0;
    derived_vn0(ie,0,igp,jgp,ilev) = 0;
    derived_vn0(ie,1,igp,jgp,ilev) = 0;
    omega_p(ie,igp,jgp,ilev) = 0;
    if (nu_p_pos) {
      derived_dpdiss_ave(ie,igp,jgp,ilev) = 0;
      derived_dpdiss_biharmonic(ie,igp,jgp,ilev) = 0;
    }
    derived_dp(ie,igp,jgp,ilev) = dp3d(ie,n0,igp,jgp,ilev);
    if (transport_alg_pos) {
      vstar(ie,0,igp,jgp,ilev) = v(ie,n0,0,igp,jgp,ilev);
      vstar(ie,1,igp,jgp,ilev) = v(ie,n0,1,igp,jgp,ilev);
    }
  });
  Kokkos::fence();
}

} // namespace Homme

#endif // HOMMEXX_PRIM_STEP_ADJ_HPP
