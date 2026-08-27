#include "prim_advance_adj.hpp"

#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"
#include "DirkFunctor.hpp"
#include "DirkFunctorImpl.hpp"
#include "Context.hpp"
#include "Diagnostics.hpp"
#include "Elements.hpp"
#include "HyperviscosityFunctor.hpp"
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

void ttype5_imex_adjoint(const Real dt_dyn,
                         const Real eta_ave_w,
                         StateSnapshot& adj_state)
{
  GPTLstart("ttype5_imex_adjoint");
  using const_tape_t = const Tape<StateSnapshot>;

  const auto& c = Context::singleton();
  SimulationParams& params = c.get<SimulationParams>();

  // Get elements and functors
  auto& elems_caar = c.get<ElementsST<DxFadTypeCaar>>();
  auto& state_caar = elems_caar.m_state;
  auto& caar_base  = c.get<CaarFunctorST<DxFadTypeCaar>>();
  auto& caar       = std::any_cast<CaarFunctorImplST<DxFadTypeCaar>&>(caar_base.impl());
  auto& tape       = std::any_cast<const_tape_t&>(c.any_map().at("imex_tape"));
  auto& geo        = c.get<ElementsGeometry>();

  auto rspheremp = geo.m_rspheremp;

  int nelem = adj_state.num_elems;
  int nm1 = 0;
  int n0  = 1;
  int np1 = 2;
  Real dt;

  // NOTATION:
  //
  // State:
  //  - u_i: state after i-th explicit CAAR stage
  // where u_0 is the state at the beginning of prim_advance_exp,
  // and u_5 is the state at the end (after 5th stage)
  //
  // Adjoint state:
  //  - lambda_i: deriv w.r.t. u_i

  // These are the needed buffers for the ttype5 adjoints. All work vars will
  // TODO: create these ONCE
  auto buf0 = adj_state.clone();
  auto buf1 = adj_state.clone();
  auto buf2 = adj_state.clone();
  auto buf3 = adj_state.clone();

  // These are all alias of lambda but they make the code underneath easier to follow
  auto lambda0 = buf0,
       lambda1 = buf1,
       lambda2 = buf0,
       lambda3 = buf1,
       lambda4 = buf0,
       lambda5 = buf1,
       lambda_sum = buf2,
       dCdu0_lambda5 = buf3;

  // These temps are used to hold intermediate steps between lambdaN and lambdaN-1.
  // WARNING: ensure that they alias lambdaN, NOT lambdaN-1, since caar's run_JtV
  // will use the tmpN as X and lambdaN-1 as Y, and they CANNOT alias each other.
  auto tmp5 = lambda5,
       tmp4 = lambda4,
       tmp3 = lambda3,
       tmp2 = lambda2,
       tmp1 = lambda1;

  lambda_sum.zero();

  // TODO: this must be created ONCE, not every time
  auto be = create_adj_bex(lambda);

  const auto& u0 = tape.at(0);
  const auto& u1 = tape.at(1);
  const auto& u2 = tape.at(2);
  const auto& u3 = tape.at(3);
  const auto& u4 = tape.at(4);
  const auto& u5 = tape.at(5);

  auto u0_5 = buf3;
  u0_5.deep_copy(u1);
  u0_5.add(u0,-1.0/4.0,5.0/4.0);

  dt = dt_dyn;

  // CAAR passes bwd

  // Stage 5
  debug_print("   stage 5...\n");
  dt = dt_dyn;
  state_caar.import_snapshot(u0_5,nm1); // Departure point for stage 5 is different

  debug_print("     CAAR...\n");
  const RKStageData stage5_data(nm1, n0, np1, -1, dt, eta_ave_w, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage5_data,lambda5,tmp5);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(tmp5, geo.m_spheremp, -1.0/4.0*stage5_data.scale3);
  // dept point contrib w.r.t u1 in stage 5
  dCdu0_lambda5.add_weighted(tmp5, geo.m_spheremp, 5.0/4.0*stage5_data.scale3,0);
  state_caar.import_snapshot(u4,n0);
  debug_print("       init J...\n");
  caar.init_J(stage5_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage5_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage5_data,tmp5,lambda4);

  // For all other stages, the departure point is u0
  state_caar.import_snapshot(u0,nm1);

  // Stage 4
  debug_print("   stage 4...\n");
  dt = dt_dyn/2.0;

  debug_print("     CAAR...\n");
  const RKStageData stage4_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage4_data,lambda4,tmp4);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(tmp4, geo.m_spheremp, stage4_data.scale3);
  state_caar.import_snapshot(u3,n0);
  debug_print("       init J...\n");
  caar.init_J(stage4_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage4_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage4_data,tmp4,lambda3);

  // Stage 3
  debug_print("   stage 3...\n");
  dt = 3.0*dt_dyn/8.0;

  debug_print("     CAAR...\n");
  const RKStageData stage3_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage3_data,lambda3,tmp3);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(tmp3, geo.m_spheremp, stage3_data.scale3);
  state_caar.import_snapshot(u2,n0);
  debug_print("       init J...\n");
  caar.init_J(stage3_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage3_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage3_data,tmp3,lambda2);

  // Stage 2
  debug_print("   stage 2...\n");
  dt = dt_dyn/6.0;

  debug_print("     CAAR...\n");
  const RKStageData stage2_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage2_data,lambda2,tmp2);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(tmp2, geo.m_spheremp, stage2_data.scale3);
  state_caar.import_snapshot(u1,n0);
  debug_print("       init J...\n");
  caar.init_J(stage2_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage2_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage2_data,tmp2,lambda1);
  lambda1.add(dCdu0_lambda5);

  // Stage 1
  debug_print("   stage 1...\n");
  dt = dt_dyn/4.0;

  debug_print("     CAAR...\n");
  const RKStageData stage1_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage1_data,lambda1,tmp1);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(tmp1, geo.m_spheremp, stage1_data.scale3);
  state_caar.import_snapshot(u0,n0);
  debug_print("       init J...\n");
  caar.init_J(stage1_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage1_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage1_data,tmp1,lambda0);

  // Add the contributions corresponding to CAAR's departure point (which is always y0)
  lambda0.add(lambda_sum);

  GPTLstop("ttype5_imex_adjoint");
}

void ttype10_imex_adjoint(const Real dt_dyn,
                          const Real eta_ave_w,
                          StateSnapshot& adj_state)
{
  GPTLstart("ttype10_imex_adjoint");
  using const_tape_t = const Tape<StateSnapshot>;

  const auto& c = Context::singleton();
  SimulationParams& params = c.get<SimulationParams>();

  // Get elements, hvcoord, and functors
  auto& elems_dirk = c.get<ElementsST<DxFadTypeDirk>>();
  auto& elems_caar = c.get<ElementsST<DxFadTypeCaar>>();
  auto& state_dirk = elems_dirk.m_state;
  auto& state_caar = elems_caar.m_state;
  auto& hvcoord    = c.get<HybridVCoord>();
  auto& dirk_base  = c.get<DirkFunctorST<DxFadTypeDirk>>();
  auto& caar_base  = c.get<CaarFunctorST<DxFadTypeCaar>>();
  auto& dirk       = std::any_cast<DirkFunctorImplST<DxFadTypeDirk>&>(dirk_base.impl());
  auto& caar       = std::any_cast<CaarFunctorImplST<DxFadTypeCaar>&>(caar_base.impl());
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
  StateSnapshot lambda (nelem);
  StateSnapshot mu = adj_state;

  // These are all alias of lambda and mu, but they make the code underneath easier to follow
  auto mu0 = mu, mu1 = mu, mu2 = mu, mu3 = mu, mu4 = mu, mu5 = mu;
  auto lambda1 = lambda, lambda2 = lambda, lambda3 = lambda, lambda4 = lambda, lambda5 = lambda;
  auto lambda_tmp = lambda; // to hold some temporaries

  // These helpers will contain, respectively:
  //  - sum lambda_i
  //  - dDirk / dx0 (y0,y1) * mu5
  //  - dDirk / dxnm1 (y0,y1) * mu5
  // The first comes from the u0 contrib in each CAAR stage, while the last two come from
  // the last DIRK stage, where the RHS contains contribs from y0 and y1
  // TODO: these structs should be created ONCE, not every time
  StateSnapshot lambda_sum(nelem), dDdy0_mu5(nelem), dDdy1_mu5(nelem);
  lambda_sum.zero();

  // TODO: this must be created ONCE, not every time
  auto be = create_adj_bex(lambda);

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

  // First, compute dDdy0_mu5 and dDdy1_mu5 (deriv of DIRK w.r.t. y0 and y1, times mu5)
  state_dirk.import_snapshot(y0, n0);
  state_dirk.import_snapshot(y1, nm1);

  dt = dt_dyn;

  debug_print("   dDdy0_mu5...\n");
  state_dirk.import_snapshot(u5, np1);
  dirk.init_J(n0,state_dirk);
  dirk.run(nm1, a2*dt, n0, a1*dt, np1, a3*dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu5,dDdy0_mu5);

  debug_print("   dDdy1_mu5...\n");
  state_dirk.import_snapshot(u5,np1);
  dirk.init_J(nm1,state_dirk);
  dirk.run(nm1, a2*dt, n0, a1*dt, np1, a3*dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu5,dDdy1_mu5);

  // Now do dirk/caar passes bwd
  state_caar.import_snapshot(y0,nm1); // Departure point is y0 for all stages

  // Stage 5
  debug_print("   stage 5...\n");
  dt = dt_dyn;

  debug_print("     DIRK...\n");
  state_dirk.import_snapshot(u5,np1);
  dirk.init_J(np1,state_dirk);
  dirk.run(nm1, a2*dt, n0, a1*dt, np1, a3*dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu5,lambda5);

  debug_print("     CAAR...\n");
  const RKStageData stage5_data(nm1, n0, np1, -1, dt, eta_ave_w, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage5_data,lambda5,lambda_tmp);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda_tmp, geo.m_spheremp, stage5_data.scale3);
  state_caar.import_snapshot(y4,n0);
  debug_print("       init J...\n");
  caar.init_J(stage5_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage5_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage5_data,lambda_tmp,mu4);

  // Stage 4
  debug_print("   stage 4...\n");
  dt = dt_dyn/2.0;

  debug_print("     DIRK...\n");
  state_dirk.import_snapshot(u4,np1);
  dirk.init_J(np1,state_dirk);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu4,lambda4);

  debug_print("     CAAR...\n");
  const RKStageData stage4_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage4_data,lambda4,lambda_tmp);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda_tmp, geo.m_spheremp, stage4_data.scale3);
  state_caar.import_snapshot(y3,n0);
  debug_print("       init J...\n");
  caar.init_J(stage4_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage4_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage4_data,lambda_tmp,mu3);

  // Stage 3
  debug_print("   stage 3...\n");
  dt = 3.0*dt_dyn/8.0;

  debug_print("     DIRK...\n");
  state_dirk.import_snapshot(u3,np1);
  dirk.init_J(np1,state_dirk);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu3,lambda3);

  debug_print("     CAAR...\n");
  const RKStageData stage3_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage3_data,lambda3,lambda_tmp);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda_tmp, geo.m_spheremp, stage3_data.scale3);
  state_caar.import_snapshot(y2,n0);
  debug_print("       init J...\n");
  caar.init_J(stage3_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage3_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage3_data,lambda_tmp,mu2);

  // Stage 2
  debug_print("   stage 2...\n");
  dt = dt_dyn/6.0;

  debug_print("     DIRK...\n");
  state_dirk.import_snapshot(u2,np1);
  dirk.init_J(np1,state_dirk);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu2,lambda2);

  debug_print("     CAAR...\n");
  const RKStageData stage2_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage2_data,lambda2,lambda_tmp);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda_tmp, geo.m_spheremp, stage2_data.scale3);
  state_caar.import_snapshot(y1,n0);
  debug_print("       init J...\n");
  caar.init_J(stage2_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage2_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage2_data,lambda_tmp,mu1);
  mu1.add(dDdy1_mu5);

  // Stage 1
  debug_print("   stage 1...\n");
  dt = dt_dyn/4.0;

  debug_print("     DIRK...\n");
  state_dirk.import_snapshot(u1,np1);
  dirk.init_J(np1,state_dirk);
  dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elems_dirk, hvcoord);
  dirk.run_JtV(np1,elems_dirk.m_state,mu1,lambda1);

  debug_print("     CAAR...\n");
  const RKStageData stage1_data(nm1, n0, np1, -1, dt, 0.0, 1.0, 0.0, 1.0);
  debug_print("       surf bc...\n");
  caar.run_JtV_surf_bc(stage1_data,lambda1,lambda_tmp);
  debug_print("       exchange...\n");
  be->exchange(rspheremp);
  lambda_sum.add_weighted(lambda_tmp, geo.m_spheremp, stage1_data.scale3);
  state_caar.import_snapshot(y0,n0);
  debug_print("       init J...\n");
  caar.init_J(stage1_data);
  debug_print("       compute J...\n");
  caar.run_pre_exchange(stage1_data);
  debug_print("       apply Jt...\n");
  caar.run_JtV(stage1_data,lambda_tmp,mu0);
  mu0.add(dDdy0_mu5);

  // Add the contributions corresponding to CAAR's departure point (which is always y0)
  mu0.add(lambda_sum);

  GPTLstop("ttype10_imex_adjoint");
}

} // namespace Homme
