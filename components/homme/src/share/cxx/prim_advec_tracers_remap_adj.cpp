#include "prim_advec_tracers_remap_adj.hpp"

#include "Context.hpp"
#include "EulerStepFunctor.hpp"
#include "EulerStepFunctorImpl.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "Tracers.hpp"

#include "profiling.hpp"

namespace Homme
{

namespace {

// The adjoint state of the tracers/derived state seen by the Euler-step adjoint.
// euler_step_adj lazily builds boundary-exchange objects bound to the specific
// Views it is first called with, so these must live (once) across all calls.
struct TracersAdjScratch {
  TracersAdjScratch (const int nelem, const int qsize) {
    adj_tracers.init(nelem,qsize);
    adj_derived.init(nelem);
  }

  TracersST<Real> adj_tracers;
  ElementsDerivedStateST<Real> adj_derived;
};

} // anonymous namespace

void copy_qdp_to_slot (const TracersQdpFullView& dst, const int slot,
                       const TracersQdpAdjView& src,
                       const int nelem, const int qsize)
{
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>> pol({0,0,0,0,0},{nelem,qsize,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie, const int iq, const int i, const int j, const int lev) {
    const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
    dst(ie,slot,iq,i,j,vpi)[vsi] = src(ie,iq,i,j,vpi)[vsi];
  });
  Kokkos::fence();
}

void copy_qdp_from_slot (const TracersQdpAdjView& dst,
                         const TracersQdpFullView& src, const int slot,
                         const int nelem, const int qsize)
{
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>> pol({0,0,0,0,0},{nelem,qsize,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie, const int iq, const int i, const int j, const int lev) {
    const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
    dst(ie,iq,i,j,vpi)[vsi] = src(ie,slot,iq,i,j,vpi)[vsi];
  });
  Kokkos::fence();
}

void prim_advec_tracers_remap_adj (EulerStepFunctorST<Real>& esf,
                                   const Real dt,
                                   TracersQdpAdjView& adj_qdp,
                                   TracersVn0AdjView& adj_vn0,
                                   TracersDpAdjView&  adj_dp)
{
  GPTLstart("prim_advec_tracers_remap_adj");

  auto& c = Context::singleton();
  auto& params = c.get<SimulationParams>();
  auto& tl     = c.get<TimeLevel>();

  EKAT_REQUIRE_MSG (params.transport_alg==0,
      "[prim_advec_tracers_remap_adj] Error! Only Eulerian tracer transport (transport_alg=0) is supported.\n");
  EKAT_REQUIRE_MSG (params.qsize>0,
      "[prim_advec_tracers_remap_adj] Error! There are no tracers.\n");
  EKAT_REQUIRE_MSG (params.nu_p==0,
      "[prim_advec_tracers_remap_adj] Error! nu_p>0 (dpdiss_ave/dpdiss_biharmonic) is not supported.\n");

  const int nelem = adj_qdp.extent_int(0);
  const int qsize = params.qsize;

  // Scratch adjoint tracers/derived state (the ones euler_step_adj works with)
  auto& scratch = c.create_if_not_there<TracersAdjScratch>(nelem,qsize);
  auto& adj_tracers = scratch.adj_tracers;
  auto& adj_derived = scratch.adj_derived;

  auto& esf_impl = std::any_cast<EulerStepFunctorImplST<Real>&>(esf.get_impl());

  // Same as in prim_advec_tracers_remap_RK2
  tl.update_tracers_levels(params.dt_tracer_factor);
  const int n0_qdp  = tl.n0_qdp;
  const int np1_qdp = tl.np1_qdp;
  esf.reset(params);

  // The 3 Euler steps (see prim_advec_tracers_remap_RK2). Each euler_step
  // overwrites the tape euler_step_adj reads, so, to differentiate step j,
  // we need to replay steps 0..j first (only the last replayed step is
  // taped). Nothing that precompute_divdp and the previous steps modify
  // (divdp, divdp_proj, DSS'd eta_dot_dpdn/omega_p, qdp(np1), qlim) is needed
  // as input to the replay: precompute_divdp regenerates divdp/divdp_proj from
  // vn0, and qdp(n0_qdp) is never written by the Euler steps.
  const Real dt_stage = dt / 2.0;
  const int       stage_n0_qdp[3] = {n0_qdp, np1_qdp, np1_qdp};
  const Real      stage_rhs[3]    = {0.0, 1.0, 2.0};
  const DSSOption stage_dss[3]    = {DSSOption::DIV_VDP_AVE, DSSOption::ETA, DSSOption::OMEGA};
  //
  // Stages 1 and 2 read and write the same qdp slot (n0_qdp==np1_qdp), but euler_step_adj
  // reads the stage's *input* qdp(n0_qdp) (not taped) to route the limiter's qlim and
  // min/max gradients. So, for the differentiated stage, we save its input and put it back
  // after the replayed fwd call overwrote it.
  auto& qdp_full = c.get<TracersST<Real>>().qdp;
  TracersQdpAdjView qdp_input("prim_advec_tracers_remap_adj qdp_input",nelem);
  auto replay = [&](const int last_stage) {
    // The 3rd stage's biharmonic set-up leaves m_data.rhs_viss=3 behind (the fwd resets it
    // once, at the start of each prim_advec_tracers_remap), which would make the replayed
    // stages 0 and 1 add the (garbage) biharmonic term. So start every replay from a fresh state.
    esf.reset(params);
    esf.precompute_divdp();
    for (int j=0; j<=last_stage; ++j) {
      const bool last = (j==last_stage);
      const bool aliased = (stage_n0_qdp[j]==np1_qdp);
      esf_impl.set_tape_for_adjoint(last);
      if (last and aliased) copy_qdp_from_slot(qdp_input,qdp_full,np1_qdp,nelem,qsize);
      esf.euler_step(np1_qdp,stage_n0_qdp[j],dt_stage,stage_rhs[j],stage_dss[j]);
      if (last and aliased) copy_qdp_to_slot(qdp_full,np1_qdp,qdp_input,nelem,qsize);
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

  // Output
  Kokkos::deep_copy(adj_qdp,Real(0));
  copy_qdp_from_slot(adj_qdp,adj_tracers.qdp,n0_qdp,nelem,qsize);
  Kokkos::deep_copy(adj_vn0,adj_derived.m_vn0);
  Kokkos::deep_copy(adj_dp,adj_derived.m_dp);

  GPTLstop("prim_advec_tracers_remap_adj");
}

} // namespace Homme
