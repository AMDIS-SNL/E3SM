#include "prim_step_adj.hpp"

#include "prim_advance_adj.hpp"
#include "prim_advance_exp.hpp"
#include "prim_advec_tracers_remap_adj.hpp"
#include "prim_step.hpp"

#include "Context.hpp"
#include "ElementsDerivedState.hpp"
#include "Elements.hpp"
#include "EulerStepFunctor.hpp"
#include "SimulationParams.hpp"
#include "Tape.hpp"
#include "TimeLevel.hpp"
#include "Tracers.hpp"

#include "profiling.hpp"

#include <ekat_pack_kokkos.hpp>

#include <type_traits>
#include <vector>

namespace Homme
{

namespace {

// adj_state.dp3d += adj_dp
void add_to_dp3d (StateSnapshot& adj_state,
                  const decltype(ElementsDerivedStateST<Real>::m_dp)& adj_dp)
{
  const int nelem = adj_state.num_elems;
  auto a_dp = ekat::scalarize(adj_state.dp3d);
  auto g    = ekat::scalarize(adj_dp);
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> pol({0,0,0,0},{nelem,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie, const int i, const int j, const int k) {
    a_dp(ie,i,j,k) += g(ie,i,j,k);
  });
  Kokkos::fence();
}

// Puts things back as they were when prim_step_adj started, even if it throws:
// params.store_fwd_state, and the live state (restored to the checkpoint).
struct PrimStepAdjGuard {
  PrimStepAdjGuard (SimulationParams& p, const PrimStepCheckpoint& c)
   : params(p), ckpt(c), store_fwd_state(p.store_fwd_state) {}
  ~PrimStepAdjGuard () {
    params.store_fwd_state = store_fwd_state;
    try {
      ckpt.restore();
    } catch (...) {
      // Destructors must not throw; we're likely already unwinding from an error
    }
  }

  SimulationParams& params;
  const PrimStepCheckpoint& ckpt;
  const bool store_fwd_state;
};

} // anonymous namespace

TimeLevel prim_step_final_time_level (const TimeLevel& tl_start,
                                      const int dt_tracer_factor)
{
  EKAT_REQUIRE_MSG (dt_tracer_factor>=1,
      "[prim_step_final_time_level] Error! dt_tracer_factor must be >= 1.\n");

  // Same sequence of level updates as in prim_step: K-1 leapfrog updates
  // between the K calls to prim_advance_exp, then the tracers levels update
  // done by prim_advec_tracers_remap.
  TimeLevel tl = tl_start;
  for (int n=1; n<dt_tracer_factor; ++n) {
    tl.update_dynamics_levels(UpdateType::LEAPFROG);
  }
  tl.update_tracers_levels(dt_tracer_factor);
  return tl;
}

PrimStepCheckpoint::PrimStepCheckpoint (const int num_elems, const int qsize_in)
 : state (num_elems)
 , qdp   ("PrimStepCheckpoint qdp", num_elems)
 , qsize (qsize_in)
{
  EKAT_REQUIRE_MSG (qsize>=0 and qsize<=QSIZE_D,
      "[PrimStepCheckpoint] Error! Invalid number of tracers.\n");
}

void PrimStepCheckpoint::take ()
{
  auto& c = Context::singleton();
  const auto& params = c.get<SimulationParams>();
  auto& elements = c.get<Elements>();
  auto& tracers  = c.get<Tracers>();

  tl = c.get<TimeLevel>();
  const int qdp_slot = prim_step_final_time_level(tl,params.dt_tracer_factor).n0_qdp;

  elements.m_state.take_snapshot(state,tl.n0,false);
  Kokkos::deep_copy(qdp,Real(0));
  copy_qdp_from_slot(qdp,tracers.qdp,qdp_slot,elements.num_elems(),qsize);
}

void PrimStepCheckpoint::restore () const
{
  auto& c = Context::singleton();
  const auto& params = c.get<SimulationParams>();
  auto& elements = c.get<Elements>();
  auto& tracers  = c.get<Tracers>();

  c.get<TimeLevel>() = tl;
  const int qdp_slot = prim_step_final_time_level(tl,params.dt_tracer_factor).n0_qdp;

  elements.m_state.import_snapshot(state,tl.n0,false);
  copy_qdp_to_slot(tracers.qdp,qdp_slot,qdp,elements.num_elems(),qsize);
}

void prim_step_adj (const Real dt,
                    const PrimStepCheckpoint& ckpt,
                    StateSnapshot& adj_state,
                    PrimStepQdpView& adj_qdp)
{
  GPTLstart("prim_step_adj");

  auto& c = Context::singleton();
  auto& params   = c.get<SimulationParams>();
  auto& elements = c.get<Elements>();
  auto& tl       = c.get<TimeLevel>();

  EKAT_REQUIRE_MSG ((std::is_same_v<ScalarValue,Real>),
      "[prim_step_adj] Error! The fwd trajectory must be recomputed with Real scalar type.\n");
  EKAT_REQUIRE_MSG (params.time_step_type==TimeStepType::ttype10_imex,
      "[prim_step_adj] Error! Only ttype10_imex has an adjoint.\n");
  EKAT_REQUIRE_MSG (params.transport_alg==0,
      "[prim_step_adj] Error! Only Eulerian tracer transport (transport_alg=0) is supported.\n");
  EKAT_REQUIRE_MSG (params.qsize>0 and params.qsize==ckpt.qsize,
      "[prim_step_adj] Error! Invalid number of tracers (must be >0 and match the checkpoint's).\n");
  EKAT_REQUIRE_MSG (params.nu_p==0,
      "[prim_step_adj] Error! nu_p>0 (dpdiss_ave/dpdiss_biharmonic) is not supported.\n");
  EKAT_REQUIRE_MSG (not params.prescribed_wind,
      "[prim_step_adj] Error! 'prescribed_wind' is not supported.\n");

  const int nelem = elements.num_elems();
  const int K     = params.dt_tracer_factor;

  // prim_advance_exp only stores the checkpoints prim_advance_adj needs if
  // store_fwd_state=true. We only need them when replaying (below), so we
  // don't waste time taking snapshots in the first sweep. The guard restores
  // the flag and the live state on exit (also if we throw).
  PrimStepAdjGuard guard(params,ckpt);
  ensure_imex_tape(nelem);

  // ===================== 1. Fwd sweep of the dynamics ===================== //
  // Recompute the trajectory of the dynamics from the checkpoint, storing the
  // state at the beginning of each of the K calls to prim_advance_exp
  // (that's all we need to replay each call individually, later on).
  ckpt.restore();
  params.store_fwd_state = false;

  set_tracer_transport_derived_values<Real>(params,elements,tl);

  // Note: this holds K snapshots of the dyn state (on device), on top of the
  // checkpoint. That's the cost of replaying each call individually; a disk-based
  // checkpoint store could replace it.
  std::vector<TimeLevel> tl_calls;
  std::vector<StateSnapshot> y_calls;
  tl_calls.reserve(K);
  y_calls.reserve(K);
  for (int n=0; n<K; ++n) {
    if (n>0) {
      tl.update_dynamics_levels(UpdateType::LEAPFROG);
    }
    tl_calls.push_back(tl);
    y_calls.emplace_back(nelem);
    elements.m_state.take_snapshot(y_calls.back(),tl.n0,false);

    prim_advance_exp(tl,dt,false);
  }

  // ===================== 2. Adjoint of the tracers step ===================== //
  // At this point, the live derived state (vn0 and dp), the live qdp(n0_qdp)
  // and the time levels are as they are when prim_advec_tracers_remap starts,
  // which is what prim_advec_tracers_remap_adj needs (it replays the Euler steps).
  GPTLstart("prim_step_adj-tracers");
  TracersVn0AdjView adj_vn0("adj_vn0",nelem);
  TracersDpAdjView  adj_dp("adj_dp",nelem);
  prim_advec_tracers_remap_adj(c.get<EulerStepFunctor>(),dt*K,adj_qdp,adj_vn0,adj_dp);
  GPTLstop("prim_step_adj-tracers");

  // ===================== 3. Adjoint of the dynamics ===================== //
  // The tracers step depends on the dynamics only through vn0 (sum over
  // the K calls of eta_ave_w*v*dp of the last RK stage) and through derived_dp,
  // which was set to dp3d(n0) *before* the first call. So, adj_vn0 is the
  // seed to be injected in each call, while adj_dp goes straight to
  // the adjoint state at the beginning of the step.
  GPTLstart("prim_step_adj-dynamics");
  params.store_fwd_state = true;
  for (int n=K-1; n>=0; --n) {
    // Replay the n-th call to prim_advance_exp, to regenerate its tape...
    tl = tl_calls[n];
    elements.m_state.import_snapshot(y_calls[n],tl.n0,false);
    prim_advance_exp(tl,dt,false);

    // ...and then run its adjoint.
    prim_advance_adj(dt,adj_state,&adj_vn0);
  }
  add_to_dp3d(adj_state,adj_dp);
  GPTLstop("prim_step_adj-dynamics");

  // Note: the guard puts the live state back to the beginning of the prim_step
  GPTLstop("prim_step_adj");
}

} // namespace Homme
