#include "prim_advance_adj.hpp"

#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"
#include "DirkAdjoint.hpp"
#include "DirkFunctor.hpp"
#include "DirkFunctorImpl.hpp"
#include "Context.hpp"
#include "Diagnostics.hpp"
#include "Elements.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HyperviscosityFunctor.hpp"
#include "HyperviscosityFunctorImpl.hpp"
#include "PhysicalConstants.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "Tape.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "profiling.hpp"

// #define PRINT_TO_SCREEN
namespace {
void debug_print (const std::string& s) {
#ifdef PRINT_TO_SCREEN
  printf("%s",s.c_str());
#else
  (void) s;
#endif
}
}

namespace Homme
{

namespace {

// Scratch buffers used by the IMEX (CAAR/DIRK) stages adjoint -- currently
// only ttype10_imex_adjoint, but shaped so that other IMEX schemes (e.g. a
// future ttype7_imex/ttype9_imex adjoint) could reuse it too -- allocated
// once (the first time they're needed) and reused across calls, rather than
// reallocated (and, for 'be', re-registered with the boundary exchange
// machinery) on every single call.
struct ImexAdjointScratch {
  explicit ImexAdjointScratch (int nelem)
   : lambda(nelem), lambda_sum(nelem), dDdy0_mu5(nelem), dDdy1_mu5(nelem)
  {}

  StateSnapshot lambda, lambda_sum, dDdy0_mu5, dDdy1_mu5;
  std::shared_ptr<BoundaryExchangeST<Real>> be;
};

// A dedicated (Real-valued) HV functor + its own private state, used only to
// linearize the hyperviscosity step in prim_advance_adj. This is kept
// entirely separate from the production Elements/HyperviscosityFunctor: the
// adjoint never touches the live simulation state, and is instead driven
// purely by the two StateSnapshots taped by prim_advance_exp (state right
// before, and right after, HV ran). This avoids replaying HV::run() (with
// its halo exchanges) just to reconstruct the post-HV state.
struct HVAdjointScratch {
  explicit HVAdjointScratch (int nelem)
    : hv(nelem, Context::singleton().get<SimulationParams>())
    , seed(nelem)
  {
    state.init(nelem);
    derived.init(nelem);
    hv.setup(Context::singleton().get<ElementsGeometry>(), state, derived);

    fbm.request_size(hv.requested_buffer_size());
    fbm.allocate();
    hv.init_buffers(fbm);
    hv.init_boundary_exchanges();
  }

  ElementsStateST<Real> state;
  ElementsDerivedStateST<Real> derived;
  FunctorsBuffersManager fbm;
  HyperviscosityFunctorST<Real> hv;
  StateSnapshot seed; // holds a copy of the incoming seed (run_JtV needs x != y)
};

// mu += J_vn0^T * adj_vn0, where vn0 = eta_ave_w * v*dp evaluated at state y
// (pointwise, so its transposed Jacobian is elementwise too):
//   d mu_v[c] += eta_ave_w * dp * adj_vn0[c]
//   d mu_dp   += eta_ave_w * sum_c v[c] * adj_vn0[c]
void add_vn0_adjoint (const AdjVn0View& adj_vn0, const StateSnapshot& y,
                      const Real eta_ave_w, StateSnapshot& mu)
{
  const int nelem = mu.num_elems;
  auto y_v   = ekat::scalarize(y.v);
  auto y_dp  = ekat::scalarize(y.dp3d);
  auto mu_v  = ekat::scalarize(mu.v);
  auto mu_dp = ekat::scalarize(mu.dp3d);
  auto g     = ekat::scalarize(adj_vn0);

  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4 = md_range_t({0,0,0,0},{nelem,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(p4, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    const Real g0 = g(ie,0,ip,jp,k);
    const Real g1 = g(ie,1,ip,jp,k);
    mu_v(ie,0,ip,jp,k) += eta_ave_w*y_dp(ie,ip,jp,k)*g0;
    mu_v(ie,1,ip,jp,k) += eta_ave_w*y_dp(ie,ip,jp,k)*g1;
    mu_dp(ie,ip,jp,k)  += eta_ave_w*(y_v(ie,0,ip,jp,k)*g0 + y_v(ie,1,ip,jp,k)*g1);
  });
  Kokkos::fence();
}

} // anonymous namespace

void ensure_imex_tape (const int num_elems)
{
  using tape_t = Tape<StateSnapshot>;
  auto& map = Context::singleton().any_map();
  auto it = map.find("imex_tape");
  if (it==map.end()) {
    map.try_emplace("imex_tape",std::in_place_type<tape_t>,imex_tape_capacity,num_elems);
  } else {
    auto& tape = std::any_cast<tape_t&>(it->second);
    EKAT_REQUIRE_MSG (tape.capacity>=imex_tape_capacity,
        "[ensure_imex_tape] Error! The existing imex tape is too small.\n"
        " - capacity : " + std::to_string(tape.capacity) + "\n"
        " - needed   : " + std::to_string(imex_tape_capacity) + "\n");
  }
}

std::shared_ptr<BoundaryExchangeST<Real>> create_adj_bex (StateSnapshot& adj_state)
{
  auto& c = Context::singleton();
  const auto& params = c.get<SimulationParams>();
  auto& bmm = c.get<MpiBuffersManagerMap>();

  auto be = std::make_shared<BoundaryExchangeST<Real>>();
  be->m_label = std::string("AdjState");
  be->set_buffers_manager(bmm[MPI_EXCHANGE]);
  be->m_diagnostics_level = params.internal_diagnostics_level;
  if (params.theta_hydrostatic_mode) {
    be->set_num_fields(0,0,4);
  } else {
    be->set_num_fields(0,0,4,2);
  }

  be->register_field(adj_state.v,2,0);
  be->register_field(adj_state.vtheta_dp);
  be->register_field(adj_state.dp3d);
  if (!params.theta_hydrostatic_mode) {
    // Note: phinh_i at the surface (last level) is constant, so it doesn't *need* bex.
    //       If bex(constant)=constant, we might just do it. This would not eliminate
    //       the need for halo-exchange of interface-based quantities though, since
    //       we would still need to exchange w_i.
    be->register_field(adj_state.w_i);
    be->register_field(adj_state.phinh_i);
  }
  be->registration_completed();

  return be;
}

void ttype10_imex_adjoint(const Real dt_dyn,
                          const Real eta_ave_w,
                          StateSnapshot& adj_state,
                          const AdjVn0View* adj_vn0,
                          const DirkAdjMethod dirk_method)
{
  GPTLstart("ttype10_imex_adjoint");
  using const_tape_t = const Tape<StateSnapshot>;

  auto& c = Context::singleton();
  SimulationParams& params = c.get<SimulationParams>();

  // Get elements, hvcoord, and functors
  auto& elems_caar = c.get<ElementsST<DxFadTypeCaar>>();
  auto& state_caar = elems_caar.m_state;
  auto& hvcoord    = c.get<HybridVCoord>();
  auto& caar_base  = c.get<CaarFunctorST<DxFadTypeCaar>>();
  auto& caar       = std::any_cast<CaarFunctorImplST<DxFadTypeCaar>&>(caar_base.impl());

  // The Dx-Fad stack of DIRK is only needed by the dense DIRK adjoint
  ElementsST<DxFadTypeDirk>*        elems_dirk = nullptr;
  DirkFunctorImplST<DxFadTypeDirk>* dirk       = nullptr;
  if (dirk_method==DirkAdjMethod::Dense) {
    elems_dirk = &c.get<ElementsST<DxFadTypeDirk>>();
    dirk = &std::any_cast<DirkFunctorImplST<DxFadTypeDirk>&>(c.get<DirkFunctorST<DxFadTypeDirk>>().impl());
  }
  auto& tape       = std::any_cast<const_tape_t&>(c.any_map().at("imex_tape"));
  auto& geo        = c.get<ElementsGeometry>();

  auto rspheremp = geo.m_rspheremp;

  int nelem = adj_state.num_elems;
  int nm1 = 0;
  int n0  = 1;
  int np1 = 2;
  Real dt;

  // Last stage DIRK factors
  Real a1 = 0.24362;
  Real a2 = 0.34184;
  Real a3 = 1-(a1+a2);

  // For each functor, load fwd state we had right before
  // running it, run functor, then compute JtV (with V=adj_state)
  // NOTATION:
  //
  // State:
  //  - u_i: state after explicit CAAR stage
  //  - y_i: state after implicit DIRK stage
  // where y_0 is the state at the beginning of prim_advance_exp,
  // and y_5 is the state at the end (after 5th DIRK stage)
  //
  // Adjoint state:
  //  - lambda_i: deriv w.r.t. u_i
  //  - mu_i: deriv w.r.t. y_i
  // Hence, lambda is the adjoint var between a CAAR and DIRK stage,
  // while mu is the adjoint var between DIRK and CAAR stages.
  // So mu5 is the adj var at entry, while mu0 is the adj var at exit
  auto& scratch = c.create_if_not_there<ImexAdjointScratch>(nelem);
  StateSnapshot& lambda = scratch.lambda;
  StateSnapshot mu = adj_state;

  // These are all alias of lambda and mu, but they make the code underneath easier to follow
  auto mu0 = mu, mu1 = mu, mu2 = mu, mu3 = mu, mu4 = mu, mu5 = mu;
  auto lambda1 = lambda, lambda2 = lambda, lambda3 = lambda, lambda4 = lambda, lambda5 = lambda;

  // These helpers will contain, respectively:
  //  - sum lambda_i
  //  - dDirk / dx0 (y0,y1) * mu5
  //  - dDirk / dxnm1 (y0,y1) * mu5
  // The first comes from the u0 contrib in each CAAR stage, while the last two come from
  // the last DIRK stage, where the RHS contains contribs from y0 and y1
  StateSnapshot& lambda_sum = scratch.lambda_sum;
  StateSnapshot& dDdy0_mu5  = scratch.dDdy0_mu5;
  StateSnapshot& dDdy1_mu5  = scratch.dDdy1_mu5;
  lambda_sum.zero();

  if (not scratch.be) {
    scratch.be = create_adj_bex(lambda);
  }
  auto& be = scratch.be;

  const auto& y0 = tape.at(0);
  const auto& u1 = tape.at(1);
  const auto& y1 = tape.at(2);
  const auto& u2 = tape.at(3);
  const auto& y2 = tape.at(4);
  const auto& u3 = tape.at(5);
  const auto& y3 = tape.at(6);
  const auto& u4 = tape.at(7);
  const auto& y4 = tape.at(8);
  const auto& u5 = tape.at(9);
  const auto& y5 = tape.at(10);

  // J^T*mu_in for one implicit (DIRK) stage, which took u_pre (as the np1 state) and gave y_post.
  // J_np1 is d(state after)/d(u_pre), and goes to out_np1. If the stage's RHS also depends on the
  // states at n0 and nm1 (alphadt_n0, alphadt_nm1 != 0: the last stage, where those are y0 and y1),
  // the products with the Jacobians w.r.t. them can be requested too (out_n0, out_nm1).
  auto dirk_jtv = [&] (const Real alphadt_nm1, const Real alphadt_n0, const Real dt2,
                       const StateSnapshot& u_pre, const StateSnapshot& y_post,
                       const StateSnapshot& mu_in, StateSnapshot& out_np1,
                       StateSnapshot* out_n0, StateSnapshot* out_nm1) {
    if (dirk_method==DirkAdjMethod::IFT) {
      dirk_stage_adjoint_ift(alphadt_nm1,alphadt_n0,dt2,geo,hvcoord,
                             alphadt_nm1!=0 ? &y1 : nullptr, alphadt_n0!=0 ? &y0 : nullptr,
                             u_pre,y_post,mu_in,out_np1,out_n0,out_nm1);
    } else {
      auto& state_dirk = elems_dirk->m_state;
      state_dirk.import_snapshot(y0, n0);
      state_dirk.import_snapshot(y1, nm1);
      // Replay the stage with Dx-Fad types, seeding the derivs w.r.t. the state at dx_tl
      auto jtv = [&] (const int dx_tl, StateSnapshot& out) {
        state_dirk.import_snapshot(u_pre,np1);
        dirk->init_J(dx_tl,state_dirk);
        dirk->run(nm1, alphadt_nm1, n0, alphadt_n0, np1, dt2, *elems_dirk, hvcoord);
        dirk->run_JtV(np1,elems_dirk->m_state,mu_in,out);
      };
      jtv(np1,out_np1);
      if (out_n0 != nullptr) {
        jtv(n0,*out_n0);
      }
      if (out_nm1 != nullptr) {
        jtv(nm1,*out_nm1);
      }
    }
  };

  // Now do dirk/caar passes bwd
  state_caar.import_snapshot(y0,nm1); // Departure point is y0 for all stages

  // Stage 5
  debug_print("   stage 5...\n");
  dt = dt_dyn;

  // In the last DIRK stage we also need the products of mu5 with the derivs of DIRK w.r.t. y0
  // and y1 (dDdy0_mu5 and dDdy1_mu5). They don't depend on the CAAR passes, so we get them now
  debug_print("     DIRK...\n");
  dirk_jtv(a2*dt,a1*dt,a3*dt,u5,y5,mu5,lambda5,&dDdy0_mu5,&dDdy1_mu5);

  debug_print("     CAAR...\n");
  const RKStageData stage5_data(nm1, n0, np1, -1, dt, eta_ave_w, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage5_data,lambda5,lambda5);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda1, geo.m_spheremp, stage5_data.scale3);
  state_caar.import_snapshot(y4,n0);
  debug_print("       init J...\n");
  caar.init_J(stage5_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage5_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage5_data,lambda5,mu4);

  // Stage 5 of the fwd also accumulated vn0 += eta_ave_w*v*dp, with (v,dp)
  // taken from its input state (y4). Add the corresponding adjoint
  // contribution to mu4 (the adjoint of y4).
  // Notes:
  //  - CaarFunctorImpl::compute_accumulated_quantities (run in the pre-exchange
  //    kernel of *every* CAAR stage) is the only writer of vn0, but the
  //    increment is eta_ave_w*v*dp of the stage's input state n0, and in
  //    ttype10 only stage 5 has eta_ave_w!=0 (see ttype10_imex_timestep).
  //    A scheme with eta_ave_w!=0 in several stages (ttype5/9) needs this
  //    injection after each such stage's JtV, at that stage's input state.
  //  - The increment is pointwise in the (already DSS'd, by the previous
  //    stage's exchange) input state, so its transpose is pointwise too:
  //    exact for any local seed, and no exchange is needed here.
  if (adj_vn0 != nullptr) {
    debug_print("       vn0 adjoint...\n");
    add_vn0_adjoint(*adj_vn0,y4,eta_ave_w,mu4);
  }

  // Stage 4
  debug_print("   stage 4...\n");
  dt = dt_dyn/2.0;

  debug_print("     DIRK...\n");
  dirk_jtv(0.0,0.0,dt,u4,y4,mu4,lambda4,nullptr,nullptr);

  debug_print("     CAAR...\n");
  const RKStageData stage4_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage4_data,lambda4,lambda4);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda1, geo.m_spheremp, stage4_data.scale3);
  state_caar.import_snapshot(y3,n0);
  debug_print("       init J...\n");
  caar.init_J(stage4_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage4_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage4_data,lambda4,mu3);

  // Stage 3
  debug_print("   stage 3...\n");
  dt = 3.0*dt_dyn/8.0;

  debug_print("     DIRK...\n");
  dirk_jtv(0.0,0.0,dt,u3,y3,mu3,lambda3,nullptr,nullptr);

  debug_print("     CAAR...\n");
  const RKStageData stage3_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage3_data,lambda3,lambda3);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda1, geo.m_spheremp, stage3_data.scale3);
  state_caar.import_snapshot(y2,n0);
  debug_print("       init J...\n");
  caar.init_J(stage3_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage3_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage3_data,lambda3,mu2);

  // Stage 2
  debug_print("   stage 2...\n");
  dt = dt_dyn/6.0;

  debug_print("     DIRK...\n");
  dirk_jtv(0.0,0.0,dt,u2,y2,mu2,lambda2,nullptr,nullptr);

  debug_print("     CAAR...\n");
  const RKStageData stage2_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage2_data,lambda2,lambda2);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda1, geo.m_spheremp, stage2_data.scale3);
  state_caar.import_snapshot(y1,n0);
  debug_print("       init J...\n");
  caar.init_J(stage2_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage2_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage2_data,lambda2,mu1);
  mu1.add(dDdy1_mu5);

  // Stage 1
  debug_print("   stage 1...\n");
  dt = dt_dyn/4.0;

  debug_print("     DIRK...\n");
  dirk_jtv(0.0,0.0,dt,u1,y1,mu1,lambda1,nullptr,nullptr);

  debug_print("     CAAR...\n");
  const RKStageData stage1_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage1_data,lambda1,lambda1);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda1, geo.m_spheremp, stage1_data.scale3);
  state_caar.import_snapshot(y0,n0);
  debug_print("       init J...\n");
  caar.init_J(stage1_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage1_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage1_data,lambda1,mu0);
  mu0.add(dDdy0_mu5);

  // Add the contributions corresponding to CAAR's departure point (which is always y0)
  mu0.add(lambda_sum);

  GPTLstop("ttype10_imex_adjoint");
}

void prim_advance_adj (const Real dt, StateSnapshot& adj_state,
                       const AdjVn0View* adj_vn0,
                       const DirkAdjMethod dirk_method)
{
  GPTLstart("prim_advance_adj");

  auto& c = Context::singleton();
  SimulationParams& params = c.get<SimulationParams>();

  EKAT_REQUIRE_MSG(not params.prescribed_wind,
      "[prim_advance_adj] Error! 'prescribed_wind' is not supported.\n");

  const int  nelem     = adj_state.num_elems;
  const Real eta_ave_w = 1.0/params.dt_tracer_factor;

  // prim_advance_exp runs (in order): w_i(n0) surface fix, <time-stepping
  // scheme> stages, then HV. The adjoint runs the transposes in reverse
  // order, dispatching to the scheme-specific stages adjoint in the switch
  // below (mirroring the switch in prim_advance_exp.cpp).

  if (params.hypervis_order==2 and params.nu>0) {
    GPTLstart("prim_advance_adj-hv");
    using const_tape_t = const Tape<StateSnapshot>;
    auto& tape = std::any_cast<const_tape_t&>(c.any_map().at("imex_tape"));

    // Snapshots taped by prim_advance_exp: state right before, and right
    // after, HV ran (see prim_advance_exp.hpp's ttype10_imex_timestep, and
    // the store_fwd_state block at the end of prim_advance_exp.cpp). Indices
    // 10/11 assume ttype10_imex's tape layout (11 stage checkpoints before
    // the post-HV one); revisit if another scheme's adjoint changes that.
    const auto& y5    = tape.at(10);
    const auto& y5_hv = tape.at(11);

    auto& scratch = c.create_if_not_there<HVAdjointScratch>(nelem);
    auto& hv = std::any_cast<HyperviscosityFunctorImplST<Real>&>(scratch.hv.impl());
    constexpr int slot = 0;

    // Base point for the vtheta_dp<->theta linearization: pre-HV (dp,theta).
    scratch.state.import_snapshot(y5,slot);
    hv.init_J(slot,dt);

    // linearize_theta_out_adjoint needs the post-HV (dp,vtheta_dp) state.
    scratch.state.import_snapshot(y5_hv,slot);

    // run_JtV requires x and y to be different objects.
    scratch.seed.deep_copy(adj_state);
    hv.run_JtV(slot,scratch.seed,adj_state);
    GPTLstop("prim_advance_adj-hv");
  }

  switch (params.time_step_type) {
    case TimeStepType::ttype10_imex:
      ttype10_imex_adjoint(dt,eta_ave_w,adj_state,adj_vn0,dirk_method);
      break;
    default:
      {
        std::string msg = "[prim_advance_adj] Error! ";
        msg += "Adjoint not implemented for time step method ";
        msg += std::to_string(etoi(params.time_step_type));
        msg += ".\n";
        EKAT_ERROR_MSG(msg);
      }
  }

  if (not params.theta_hydrostatic_mode) {
    // Adjoint of the w_i(n0) surface fix at the top of prim_advance_exp:
    //   w(surface) = (u_last*gradphis_x + v_last*gradphis_y)/g
    // This is exactly linear in (u,v), and depends only on (constant)
    // geometry, so no base point/snapshot is needed. v is only *read* by
    // the fwd fix, so its adjoint contribution is added into adj_v;
    // w_i(surface) is fully *overwritten* by the fwd fix (never accumulated
    // into), so its incoming adjoint is zeroed once transferred to adj_v.
    GPTLstart("prim_advance_adj-wsurf");
    const auto& geo = c.get<ElementsGeometry>();
    auto gradphis = geo.m_gradphis;
    auto adj_v = ekat::scalarize(adj_state.v);
    auto adj_w = ekat::scalarize(adj_state.w_i);
    constexpr int last_mid = NUM_PHYSICAL_LEV-1;
    constexpr int last_int = NUM_INTERFACE_LEV-1;
    constexpr Real g = PhysicalConstants::g;
    using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>;
    auto p3 = md_range_t({0,0,0}, {nelem,NP,NP});
    Kokkos::parallel_for(p3, KOKKOS_LAMBDA (const int ie, const int ip, const int jp) {
      const Real gx = gradphis(ie,0,ip,jp);
      const Real gy = gradphis(ie,1,ip,jp);
      const Real w_adj = adj_w(ie,ip,jp,last_int);
      adj_v(ie,0,ip,jp,last_mid) += w_adj*gx/g;
      adj_v(ie,1,ip,jp,last_mid) += w_adj*gy/g;
      adj_w(ie,ip,jp,last_int) = 0;
    });
    Kokkos::fence();
    GPTLstop("prim_advance_adj-wsurf");
  }

  GPTLstop("prim_advance_adj");
}

} // namespace Homme
