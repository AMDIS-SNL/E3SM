#include "prim_step_adj.hpp"

#include "prim_advance_adj.hpp"

#include "Context.hpp"
#include "ElementsDerivedState.hpp"
#include "Elements.hpp"
#include "EulerStepFunctor.hpp"
#include "EulerStepFunctorImpl.hpp"
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

// The adjoint state of the tracers/derived state seen by the Euler-step adjoint.
// euler_step_adj lazily builds boundary-exchange objects bound to the specific
// Views it is first called with, so these must live (once) across all calls.
struct PrimStepAdjScratch {
  PrimStepAdjScratch (const int nelem, const int qsize) {
    adj_tracers.init(nelem,qsize);
    adj_derived.init(nelem);
  }

  TracersST<Real> adj_tracers;
  ElementsDerivedStateST<Real> adj_derived;
};

// dst(ie,slot,...) = src(ie,...), for the first qsize tracers
void copy_qdp_to_slot (const ExecViewManaged<PackType<Real>*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>& dst,
                       const int slot,
                       const PrimStepQdpView& src,
                       const int nelem, const int qsize)
{
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>> pol({0,0,0,0,0},{nelem,qsize,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie, const int iq, const int i, const int j, const int lev) {
    const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
    dst(ie,slot,iq,i,j,vpi)[vsi] = src(ie,iq,i,j,vpi)[vsi];
  });
  Kokkos::fence();
}

// dst(ie,...) = src(ie,slot,...), for the first qsize tracers
void copy_qdp_from_slot (const PrimStepQdpView& dst,
                         const ExecViewManaged<PackType<Real>*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>& src,
                         const int slot,
                         const int nelem, const int qsize)
{
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>> pol({0,0,0,0,0},{nelem,qsize,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie, const int iq, const int i, const int j, const int lev) {
    const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
    dst(ie,iq,i,j,vpi)[vsi] = src(ie,slot,iq,i,j,vpi)[vsi];
  });
  Kokkos::fence();
}

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

} // anonymous namespace

TimeLevel prim_step_final_time_level (const TimeLevel& tl_start,
                                      const int dt_tracer_factor)
{
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
  const int qsize = params.qsize;
  const int K     = params.dt_tracer_factor;

  // Scratch adjoint tracers/derived state (the ones euler_step_adj works with)
  auto& scratch = c.create_if_not_there<PrimStepAdjScratch>(nelem,qsize);
  auto& adj_tracers = scratch.adj_tracers;
  auto& adj_derived = scratch.adj_derived;

  // prim_advance_exp only stores the checkpoints prim_advance_adj needs if
  // store_fwd_state=true. We only need them when replaying (below), so we
  // don't waste time taking snapshots in the first sweep.
  const bool store_fwd_state_orig = params.store_fwd_state;
  if (c.any_map().find("imex_tape")==c.any_map().end()) {
    // ttype10: 11 CAAR/DIRK checkpoints, plus 1 for the state after HV
    c.any_map().try_emplace("imex_tape",std::in_place_type<Tape<StateSnapshot>>,12,nelem);
  }

  // ===================== 1. Fwd sweep of the dynamics ===================== //
  // Recompute the trajectory of the dynamics from the checkpoint, storing the
  // state at the beginning of each of the K calls to prim_advance_exp
  // (that's all we need to replay each call individually, later on).
  ckpt.restore();
  params.store_fwd_state = false;

  set_tracer_transport_derived_values<Real>(params,elements,tl);

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
  // and the time levels are as they are when prim_advec_tracers_remap starts.
  GPTLstart("prim_step_adj-tracers");
  auto& esf = c.get<EulerStepFunctor>();
  auto& esf_impl = std::any_cast<EulerStepFunctorImplST<Real>&>(esf.get_impl());

  // Same as in prim_advec_tracers_remap_RK2
  tl.update_tracers_levels(K);
  const int n0_qdp  = tl.n0_qdp;
  const int np1_qdp = tl.np1_qdp;
  esf.reset(params);

  // The 3 Euler steps (see prim_advec_tracers_remap_RK2). Each euler_step
  // overwrites the tape euler_step_adj reads, so, to differentiate step j,
  // we need to replay steps 0..j-1 first (only the last replayed step is
  // taped). Nothing that precompute_divdp and the previous steps modify
  // (divdp, divdp_proj, DSS'd eta_dot_dpdn/omega_p, qdp(np1), qlim) is needed
  // as input to the replay: precompute_divdp regenerates divdp/divdp_proj from
  // vn0, and qdp(n0_qdp) is never written by the Euler steps.
  const Real dt_stage = dt*K / 2.0;
  const int      stage_n0_qdp[3]  = {n0_qdp, np1_qdp, np1_qdp};
  const Real     stage_rhs[3]     = {0.0, 1.0, 2.0};
  const DSSOption stage_dss[3]    = {DSSOption::DIV_VDP_AVE, DSSOption::ETA, DSSOption::OMEGA};
  auto replay = [&](const int last_stage) {
    esf.precompute_divdp();
    for (int j=0; j<=last_stage; ++j) {
      const bool tape = (j==last_stage);
      esf_impl.set_tape_for_adjoint(tape);
      esf.euler_step(np1_qdp,stage_n0_qdp[j],dt_stage,stage_rhs[j],stage_dss[j]);
    }
    esf_impl.set_tape_for_adjoint(false);
  };

  // Init the adjoint state: seed on qdp(np1_qdp), nothing else
  Kokkos::deep_copy(adj_tracers.qdp,Real(0));
  Kokkos::deep_copy(adj_tracers.qlim,Real(0));
  Kokkos::deep_copy(adj_tracers.qtens_biharmonic,Real(0));
  Kokkos::deep_copy(adj_derived.m_vn0,Real(0));
  Kokkos::deep_copy(adj_derived.m_dp,Real(0));
  Kokkos::deep_copy(adj_derived.m_divdp,Real(0));
  Kokkos::deep_copy(adj_derived.m_divdp_proj,Real(0));
  Kokkos::deep_copy(adj_derived.m_dpdiss_biharmonic,Real(0));
  Kokkos::deep_copy(adj_derived.m_eta_dot_dpdn,Real(0));
  Kokkos::deep_copy(adj_derived.m_omega_p,Real(0));
  copy_qdp_to_slot(adj_tracers.qdp,np1_qdp,adj_qdp,nelem,qsize);

  esf_impl.qdp_time_avg_adj(n0_qdp,np1_qdp,adj_tracers);
  for (int j=2; j>=0; --j) {
    replay(j);
    esf_impl.euler_step_adj(np1_qdp,stage_n0_qdp[j],dt_stage,stage_rhs[j],stage_dss[j],
                            adj_tracers,adj_derived);
  }
  esf_impl.precompute_divdp_adj(adj_derived);

  // Output: adjoint of qdp before the first Euler step
  Kokkos::deep_copy(adj_qdp,Real(0));
  copy_qdp_from_slot(adj_qdp,adj_tracers.qdp,n0_qdp,nelem,qsize);
  GPTLstop("prim_step_adj-tracers");

  // ===================== 3. Adjoint of the dynamics ===================== //
  // The tracers step depends on the dynamics only through vn0 (sum over
  // the K calls of eta_ave_w*v*dp of the last RK stage) and through derived_dp,
  // which was set to dp3d(n0) *before* the first call. So, adj_derived.m_vn0 is the
  // seed to be injected in each call, while adj_derived.m_dp goes straight to
  // the adjoint state at the beginning of the step.
  GPTLstart("prim_step_adj-dynamics");
  params.store_fwd_state = true;
  for (int n=K-1; n>=0; --n) {
    // Replay the n-th call to prim_advance_exp, to regenerate its tape...
    tl = tl_calls[n];
    elements.m_state.import_snapshot(y_calls[n],tl.n0,false);
    prim_advance_exp(tl,dt,false);

    // ...and then run its adjoint.
    prim_advance_adj(dt,adj_state,&adj_derived.m_vn0);
  }
  add_to_dp3d(adj_state,adj_derived.m_dp);
  GPTLstop("prim_step_adj-dynamics");

  // Leave the live state as we found it (at the beginning of the prim_step)
  params.store_fwd_state = store_fwd_state_orig;
  ckpt.restore();

  GPTLstop("prim_step_adj");
}

} // namespace Homme
