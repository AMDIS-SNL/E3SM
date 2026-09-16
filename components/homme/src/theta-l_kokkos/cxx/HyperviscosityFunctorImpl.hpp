/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP
#define HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP

#include "ElementsGeometry.hpp"
#include "ElementsState.hpp"
#include "ElementsDerivedState.hpp"
#include "ColumnOps.hpp"
#include "EquationOfState.hpp"
#include "ElementOps.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"

#include <memory>

#include "profiling.hpp"

namespace Homme
{

template<typename ST>
class BoundaryExchangeST;
struct FunctorsBuffersManager;

template<typename ST>
class HyperviscosityFunctorImplST
{
  using PT = PackType<ST>;

  // TODO: don't pass nu_ratio1/2. Instead, do like in F90: compute them from
  //       nu, nu_div, and hv_scaling
  struct HyperviscosityData {
    HyperviscosityData(const int hypervis_subcycle_in, 
                       const int hypervis_subcycle_tom_in, 
                       const Real nu_ratio1_in, const Real nu_ratio2_in, const Real nu_top_in,
                       const Real nu_in, const Real nu_p_in, const Real nu_s_in,
                       const Real hypervis_scaling_in)
                      : hypervis_subcycle(hypervis_subcycle_in) 
                      , hypervis_subcycle_tom(hypervis_subcycle_tom_in)
                      , nu_ratio1(nu_ratio1_in), nu_ratio2(nu_ratio2_in)
                      , nu_top(nu_top_in), nu(nu_in), nu_p(nu_p_in), nu_s(nu_s_in)
                      , consthv(hypervis_scaling_in == 0) {}

    const int   hypervis_subcycle;
    const int   hypervis_subcycle_tom;

    Real  nu_ratio1;
    Real  nu_ratio2;

#ifdef HOMMEXX_TEST_HYPERVIS_FAD
    ST rel_perturb;
    ST nu;
#else
    const Real  nu;
#endif
    const Real  nu_top;
    const Real  nu_p;
    const Real  nu_s;

    int         np1; // The time-level on which to apply hv
    Real        dt;
    Real        dt_hvs;
    Real        dt_hvs_tom;

    Real        eta_ave_w;

    bool consthv;
  };//hyperviscosityData

  struct Buffers {
    ExecViewManaged<PT * [NP][NP][NUM_LEV]>    dptens;
    ExecViewManaged<PT * [NP][NP][NUM_LEV]>    ttens;
    ExecViewManaged<PT * [NP][NP][NUM_LEV]>    wtens;
    ExecViewManaged<PT * [NP][NP][NUM_LEV]>    phitens;
    ExecViewManaged<PT * [2][NP][NP][NUM_LEV]> vtens;
  };//buffers

public:

  struct TagFirstLaplaceHV {};
  struct TagSecondLaplaceConstHV {};
  struct TagSecondLaplaceTensorHV {};
  struct TagUpdateStates {};
  struct TagApplyInvMass {};
  struct TagHyperPreExchange {};
  struct TagNutopUpdateStates {};
  struct TagNutopLaplace {};

  HyperviscosityFunctorImplST (const SimulationParams&           params,
                               const ElementsGeometry&           geometry,
                               const ElementsStateST<ST>&        state,
                               const ElementsDerivedStateST<ST>& derived);

  HyperviscosityFunctorImplST (const int num_elems, const SimulationParams& params);

  void init_params(const SimulationParams& params);

  void setup(const ElementsGeometry&           geometry,
             const ElementsStateST<ST>&        state,
             const ElementsDerivedStateST<ST>& derived);

  int requested_buffer_size () const;
  void init_buffers (const FunctorsBuffersManager& fbm);
  void init_boundary_exchanges();

  void run (const int np1, const Real dt, const Real eta_ave_w);

  void biharmonic_wk_theta () const;

  // ==========================================================================
  // Discrete tangent-linear / adjoint support: init_J / run_JV / run_JtV.
  //
  // HV's diffusion operator (the two Laplacian applications, the two boundary
  // exchanges, and the diagonal nu/dt scalings) is LINEAR and does not depend
  // on the state at all -- only on geometry and parameters. laplace_simple and
  // vlaplace_sphere_wk_contra are, by construction, the Galerkin "D^T*W*D"
  // form (divergence_sphere_wk is literally the discrete transpose of
  // gradient_sphere, weighted by quadrature/metric factors), hence exactly
  // (discretely, not just in the continuous limit) self-adjoint; the two
  // exchange() calls are exactly self-adjoint too, given a properly
  // DSS-assembled mass matrix. (Verified numerically in
  // hv_sphere_ops_discrete_self_adjoint, hv_sacado_ut.cpp.) So run_JV/run_JtV
  // below apply the *same* kernels run() uses, run_JtV just in reverse order,
  // with no Sacado/Fad machinery needed -- unlike CAAR, whose pre-exchange
  // dynamics are genuinely nonlinear and state-dependent throughout.
  //
  // The only state-dependent (nonlinear) pieces are the pointwise
  // vtheta_dp<->theta conversions (need a base-point linearization, via
  // init_J) and the w-surface fix (already exactly linear in u,v -- no
  // linearization needed, just reused/adjointed directly).
  //
  // Only the const-viscosity (hypervis_scaling==0) path and nu_top==0 are
  // supported for now. Tensor viscosity (laplace_tensor/
  // vlaplace_sphere_wk_cartesian) and the sponge layer are not exercised by
  // any test yet, so their self-adjointness hasn't been verified; extending
  // support means checking that first, then mirroring the same pattern below.

  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, Real>>
  init_J (const int) = delete;

  // Snapshot dp0, theta0=vtheta0/dp0 at time level np1, BEFORE run() is
  // called: needed to linearize the vtheta_dp<->theta conversion. Must be
  // called before run().
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  init_J (const int np1);

  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, Real>>
  run_JV (const int, const StateSnapshot&, StateSnapshot&) = delete;

  // y = J*x, where J = d(state(np1))/d(state(np1)) is the Jacobian of run(),
  // linearized about the state snapshotted by init_J (which must have been
  // called, followed by run(np1,dt,eta_ave_w), before this call). x and y
  // must be different objects.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  run_JV (const int np1, const StateSnapshot& x, StateSnapshot& y);

  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, Real>>
  run_JtV (const int, const StateSnapshot&, StateSnapshot&) = delete;

  // y = J^T*x, using the same J as run_JV. x and y must be different objects.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  run_JtV (const int np1, const StateSnapshot& x, StateSnapshot& y);

protected:

  // ---- Helpers for run_JV/run_JtV (see the comment block above them) ----

  // (dp,vtheta) -> (dp,theta), theta = (vtheta - theta0*dp)/dp0 (quotient
  // rule, linearized about the state snapshotted by init_J). Also copies the
  // fields this step doesn't touch from x to y unchanged. y.vtheta_dp holds
  // the *theta* tangent from here until linearize_theta_out runs, exactly
  // mirroring how run() keeps vtheta_dp holding theta through its own loop.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  linearize_theta_in (const StateSnapshot& x, StateSnapshot& y) const;

  // Adjoint of linearize_theta_in, in place: on entry y.dp3d/y.vtheta_dp hold
  // the adjoint w.r.t. (dp,theta) at the core's input; on exit, the adjoint
  // w.r.t. the original (dp,vtheta_dp).
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  linearize_theta_in_adjoint (StateSnapshot& y) const;

  // (theta,dp) [core output] -> (vtheta_dp,dp) [final]: vtheta_dp_final =
  // theta*dp_final_base + theta_final_base*dp (product rule). Uses the
  // state's POST-run() values (the base point at the *output* of the
  // diffusion), so must be called after run() has completed.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  linearize_theta_out (const int np1, StateSnapshot& y) const;

  // Adjoint of linearize_theta_out, in place: on entry y.dp3d/y.vtheta_dp
  // hold the adjoint w.r.t. the final (dp,vtheta_dp); on exit, the adjoint
  // w.r.t. the core's output (dp,theta).
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  linearize_theta_out_adjoint (const int np1, StateSnapshot& y) const;

  // w(surface) = (u_last*gradphis_x + v_last*gradphis_y)/g: exactly linear in
  // u,v, so no base point is needed, just the (constant) geometry.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  apply_w_surface_tangent (StateSnapshot& y) const;

  // Adjoint of apply_w_surface_tangent: scatters x.w_i(surface) into
  // y.v(u/v,last level); y.w_i(surface) itself is zeroed, since run() never
  // reads w_init at the surface (it's overwritten, not accumulated into).
  // All other fields pass through from x to y unchanged.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  apply_w_surface_adjoint (const StateSnapshot& x, StateSnapshot& y) const;

  // Laplacian(nu_ratio) of src's (dp3d,vtheta_dp,w_i,phinh_i,v) into
  // m_buffers (dptens,ttens,wtens,phitens,vtens).
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  hv_apply_laplace1 (const StateSnapshot& src, const Real nu_ratio) const;

  // Laplacian(nu_ratio) applied in place to m_buffers.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  hv_apply_laplace2 (const Real nu_ratio) const;

  // m_buffers *= -nu (v,vth), -nu_p (dp), -nu_s (w,phi if NH). Same scaling
  // TagHyperPreExchange applies to the real tens buffers.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  hv_scale_tens () const;

  // dst.field += dt_hvs*rspheremp * m_buffers.tens (matches TagUpdateStates).
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  hv_add_scaled_tens_to (StateSnapshot& dst) const;

  // m_buffers.tens = dt_hvs*rspheremp * src.field (adjoint of the *scaling*
  // half of TagUpdateStates; the "+=" half is peeled off separately, since it
  // is the first step of the reverse chain, not folded into a running sum).
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  hv_scale_state_into_tens (const StateSnapshot& src) const;

  // dst.field += m_buffers.tens (no scaling: used for the last step of the
  // reverse chain, where the dt_hvs*rspheremp scaling was already applied as
  // the very first step).
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  hv_add_tens_to (StateSnapshot& dst) const;

  // One subcycle iteration of y += T(y) [forward, matches biharmonic_wk_theta
  // + TagHyperPreExchange + exchange + TagUpdateStates exactly], or its
  // transpose [reverse: same 6 self-adjoint factors, in reverse order:
  // (f6 o f5 o f4 o f3 o f2 o f1)^T = f1 o f2 o f3 o f4 o f5 o f6, since each
  // fi is individually self-adjoint].
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  run_linear_core (StateSnapshot& y, const bool forward) const;

public:

  // first iter of laplace, const hv
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFirstLaplaceHV&, const TeamMember& team) const {
    using Column = decltype(Homme::subview(m_state.m_phinh_i,0,0,0,0));
    using RefColumn = decltype(Homme::subview(m_state.m_ref_states.phi_i_ref,0,0,0));

    KernelVariables kv(team, m_tu);
    // Subtract the reference states from the states
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto vtheta = Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.np1,igp,jgp);
      auto dp    = Homme::subview(m_state.m_dp3d,kv.ie,m_data.np1,igp,jgp);

      auto theta_ref = Homme::subview(m_state.m_ref_states.theta_ref,kv.ie,igp,jgp);
      auto dp_ref    = Homme::subview(m_state.m_ref_states.dp_ref,kv.ie,igp,jgp);

      Column phi_i;
      RefColumn phi_i_ref;

      if (m_process_nh_vars) {
        phi_i = Homme::subview(m_state.m_phinh_i,kv.ie,m_data.np1,igp,jgp);
        phi_i_ref = Homme::subview(m_state.m_ref_states.phi_i_ref,kv.ie,igp,jgp);
      }
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        vtheta(ilev) -= theta_ref(ilev);
        dp(ilev)     -= dp_ref(ilev);
        if (m_process_nh_vars) {
          phi_i(ilev)  -= phi_i_ref(ilev);
        }
      });

//defined/used where?
#ifndef XX_NONBFB_COMING
      // It would be fine to not even bother with the surface level, since
      // phitens is only NUM_LEV long, so all the hv stuff does not even happen
      // at NUM_LEV_P (unless NUM_LEV_P==NUM_LEV). However, removing the subtraction
      // and addition of phi_i_ref at NUM_LEV_P introduces NON BFB diffs.
      if (m_process_nh_vars && NUM_LEV!=NUM_LEV_P) {
        Kokkos::single(Kokkos::PerThread(kv.team),[&](){
          phi_i(NUM_LEV_P-1) -= phi_i_ref(NUM_LEV_P-1);
        });
      }
#endif
    }); //team thread range

    //to ensure profiles are fully subtracted
    kv.team_barrier();

    // Laplacian of layer thickness
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_state.m_dp3d,kv.ie,m_data.np1),
                   Homme::subview(m_buffers.dptens,kv.ie));
    // Laplacian of theta
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.np1),
                   Homme::subview(m_buffers.ttens,kv.ie));

    if (m_process_nh_vars) {
      // Laplacian of vertical velocity (do not compute last interface)
      m_sphere_ops.template laplace_simple<NUM_LEV,NUM_LEV_P>(kv,
                     Homme::subview(m_state.m_w_i,kv.ie,m_data.np1),
                     Homme::subview(m_buffers.wtens,kv.ie));
      // Laplacian of geopotential (do not compute last interface)
      m_sphere_ops.template laplace_simple<NUM_LEV,NUM_LEV_P>(kv,
                     Homme::subview(m_state.m_phinh_i,kv.ie,m_data.np1),
                     Homme::subview(m_buffers.phitens,kv.ie));
    }//if

    // Laplacian of velocity
    m_sphere_ops.vlaplace_sphere_wk_contra(kv, m_data.nu_ratio1,
                              Homme::subview(m_state.m_v,kv.ie,m_data.np1),
                              Homme::subview(m_buffers.vtens,kv.ie));
  }//TagFirstLaplaceHV

  // Laplace for nu_top
  KOKKOS_INLINE_FUNCTION
  void operator()(const TagNutopLaplace&, const TeamMember& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagNutopUpdateStates&, const TeamMember& team) const;

  //second iter of laplace, const hv
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagSecondLaplaceConstHV&, const TeamMember& team) const {
    KernelVariables kv(team, m_tu);
    // Laplacian of layers thickness
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_buffers.dptens,kv.ie),
                   Homme::subview(m_buffers.dptens,kv.ie));
    // Laplacian of theta
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_buffers.ttens,kv.ie),
                   Homme::subview(m_buffers.ttens,kv.ie));

    if (m_process_nh_vars) {
      // Laplacian of vertical velocity
      m_sphere_ops.laplace_simple(kv,
                     Homme::subview(m_buffers.wtens,kv.ie),
                     Homme::subview(m_buffers.wtens,kv.ie));
      // Laplacian of vertical geopotential
      m_sphere_ops.laplace_simple(kv,
                     Homme::subview(m_buffers.phitens,kv.ie),
                     Homme::subview(m_buffers.phitens,kv.ie));
    }
    // Laplacian of velocity
    m_sphere_ops.vlaplace_sphere_wk_contra(kv, m_data.nu_ratio2,
                              Homme::subview(m_buffers.vtens,kv.ie),
                              Homme::subview(m_buffers.vtens,kv.ie));
  } //tag second laplace const hv

  //second iter of laplace, tensor hv
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagSecondLaplaceTensorHV&, const TeamMember& team) const {
    KernelVariables kv(team, m_tu);
    // Laplacian of layers thickness
    m_sphere_ops.laplace_tensor(kv,
                   Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                   Homme::subview(m_buffers.dptens,kv.ie),
                   Homme::subview(m_buffers.dptens,kv.ie));
    // Laplacian of theta
    m_sphere_ops.laplace_tensor(kv,
                   Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                   Homme::subview(m_buffers.ttens,kv.ie),
                   Homme::subview(m_buffers.ttens,kv.ie));

    if (m_process_nh_vars) {
      // Laplacian of vertical velocity
      m_sphere_ops.laplace_tensor(kv,
                     Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                     Homme::subview(m_buffers.wtens,kv.ie),
                     Homme::subview(m_buffers.wtens,kv.ie));
      // Laplacian of geopotential
      m_sphere_ops.laplace_tensor(kv,
                     Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                     Homme::subview(m_buffers.phitens,kv.ie),
                     Homme::subview(m_buffers.phitens,kv.ie));
    }

    m_sphere_ops.vlaplace_sphere_wk_cartesian(kv, 
                   Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                   Homme::subview(m_geometry.m_vec_sph2cart,kv.ie),
                   Homme::subview(m_buffers.vtens,kv.ie),
                   Homme::subview(m_buffers.vtens,kv.ie));
  } //SecondLaplaceTensorHV

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagUpdateStates&, const TeamMember& team) const {
    KernelVariables kv(team, m_tu);

    using MidColumn = decltype(Homme::subview(m_buffers.wtens,0,0,0));
    using IntColumn = decltype(Homme::subview(m_state.m_w_i,0,0,0,0));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // Add Xtens quantities back to the states, except for vtheta
      auto u = Homme::subview(m_state.m_v,kv.ie,m_data.np1,0,igp,jgp);
      auto v = Homme::subview(m_state.m_v,kv.ie,m_data.np1,1,igp,jgp);
      auto vtheta = Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.np1,igp,jgp);
      auto dp     = Homme::subview(m_state.m_dp3d,kv.ie,m_data.np1,igp,jgp);

      auto utens   = Homme::subview(m_buffers.vtens,kv.ie,0,igp,jgp);
      auto vtens   = Homme::subview(m_buffers.vtens,kv.ie,1,igp,jgp);
      auto ttens   = Homme::subview(m_buffers.ttens,kv.ie,igp,jgp);
      auto dptens  = Homme::subview(m_buffers.dptens,kv.ie,igp,jgp);
      const auto& rspheremp = m_geometry.m_rspheremp(kv.ie,igp,jgp);

      MidColumn wtens, phitens;
      IntColumn w, phi_i;

      if (m_process_nh_vars) {
        wtens   = Homme::subview(m_buffers.wtens,kv.ie,igp,jgp);
        phitens = Homme::subview(m_buffers.phitens,kv.ie,igp,jgp);
        w       = Homme::subview(m_state.m_w_i,kv.ie,m_data.np1,igp,jgp);
        phi_i   = Homme::subview(m_state.m_phinh_i,kv.ie,m_data.np1,igp,jgp);
      }

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {

        utens(ilev)   *= m_data.dt_hvs*rspheremp;
        vtens(ilev)   *= m_data.dt_hvs*rspheremp;
        ttens(ilev)   *= m_data.dt_hvs*rspheremp;
        dptens(ilev)  *= m_data.dt_hvs*rspheremp;


        u(ilev)      += utens(ilev);
        v(ilev)      += vtens(ilev);
        vtheta(ilev) += ttens(ilev);
        dp(ilev)     += dptens(ilev);
        if (m_process_nh_vars) {
          wtens(ilev)   *= m_data.dt_hvs * rspheremp;
          phitens(ilev) *= m_data.dt_hvs * rspheremp;

          w(ilev)      += wtens(ilev);
          phi_i(ilev)  += phitens(ilev);
        }
      });
    });
  }  //tagupdatestates

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagHyperPreExchange, const TeamMember &team) const {
    using Column = decltype(Homme::subview(m_state.m_phinh_i,0,0,0,0));
    using RefColumn = decltype(Homme::subview(m_state.m_ref_states.phi_i_ref,0,0,0));

    KernelVariables kv(team, m_tu);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &point_idx) {
      const int igp = point_idx / NP;
      const int jgp = point_idx % NP;

      const auto dpdiss_ave = Homme::subview(m_derived.m_dpdiss_ave,kv.ie, igp, jgp);
      const auto dpdiss_bih = Homme::subview(m_derived.m_dpdiss_biharmonic,kv.ie, igp, jgp);
      const auto dp3d = Homme::subview(m_state.m_dp3d,kv.ie, m_data.np1, igp, jgp);
      const auto dp_ref = Homme::subview(m_state.m_ref_states.dp_ref,kv.ie, igp, jgp);
      const auto theta = Homme::subview(m_state.m_vtheta_dp,kv.ie, m_data.np1, igp, jgp);
      const auto theta_ref = Homme::subview(m_state.m_ref_states.theta_ref,kv.ie, igp, jgp);
      const auto dptens = Homme::subview(m_buffers.dptens,kv.ie, igp, jgp);

      Column phi;
      RefColumn phi_ref;

      if (m_process_nh_vars) {
        phi = Homme::subview(m_state.m_phinh_i,kv.ie, m_data.np1, igp, jgp);
        phi_ref = Homme::subview(m_state.m_ref_states.phi_i_ref,kv.ie, igp, jgp);
      }

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &ilev) {

        dp3d(ilev) += dp_ref(ilev);
        theta(ilev) += theta_ref(ilev);
        if (m_process_nh_vars) {
          phi(ilev) += phi_ref(ilev);
        }
        if (m_data.nu_p>0) {
          dpdiss_ave(ilev) += m_data.eta_ave_w*dp3d(ilev) / m_data.hypervis_subcycle;
          dpdiss_bih(ilev) += m_data.eta_ave_w*dptens(ilev) / m_data.hypervis_subcycle;
        }
      });

//where is it set?
#ifndef XX_NONBFB_COMING
      // It would be fine to not even bother with the surface level, since
      // phitens is only NUM_LEV long, so all the hv stuff does not even happen
      // at NUM_LEV_P (unless NUM_LEV_P==NUM_LEV). However, removing the subtraction
      // and addition of phi_i_ref at NUM_LEV_P introduces NON BFB diffs.
      if (m_process_nh_vars && NUM_LEV!=NUM_LEV_P) {
        Kokkos::single(Kokkos::PerThread(kv.team),[&](){
          phi(NUM_LEV_P-1) += phi_ref(NUM_LEV_P-1);
        });
      }
#endif
    });//teamthreadrange loop
    kv.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &point_idx) {
      const int igp = point_idx / NP;
      const int jgp = point_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &lev) {
        m_buffers.vtens(kv.ie, 0, igp, jgp, lev) *= -m_data.nu;
        m_buffers.vtens(kv.ie, 1, igp, jgp, lev) *= -m_data.nu;
        m_buffers.ttens(kv.ie, igp, jgp, lev) *= -m_data.nu;
        m_buffers.dptens(kv.ie, igp, jgp, lev) *= -m_data.nu_p;
        if (m_process_nh_vars) {
          m_buffers.wtens(kv.ie, igp, jgp, lev) *= -m_data.nu;
          m_buffers.phitens(kv.ie, igp, jgp, lev) *= -m_data.nu_s;
        }
      });//thread vector

    });//parallel 4
  } //taghyperpreexchange

protected:

  const int                   m_num_elems;
  HyperviscosityData          m_data;
  ElementsStateST<ST>         m_state;
  ElementsDerivedStateST<ST>  m_derived;
  ElementsGeometry            m_geometry;
  SphereOperatorsST<ST>       m_sphere_ops;
  ElementOps                  m_elem_ops;
  EquationOfState<>           m_eos;
  Buffers                     m_buffers;
  HybridVCoord                m_hvcoord;

  bool m_process_nh_vars;

  // Policies
  Kokkos::TeamPolicy<ExecSpace,TagUpdateStates>     m_policy_update_states;
  Kokkos::TeamPolicy<ExecSpace,TagFirstLaplaceHV>   m_policy_first_laplace;
  Kokkos::TeamPolicy<ExecSpace,TagHyperPreExchange> m_policy_pre_exchange;

  Kokkos::TeamPolicy<ExecSpace,TagNutopLaplace>      m_policy_nutop_laplace;
  Kokkos::TeamPolicy<ExecSpace,TagNutopUpdateStates> m_policy_nutop_update_states;

  TeamUtils<ExecSpace> m_tu; // If the policies only differ by tag, just need one tu

  std::shared_ptr<BoundaryExchangeST<ST>> m_be, m_be_tom;

  ExecViewManaged<PT[NUM_LEV]> m_nu_scale_top;
  int m_nu_scale_top_ilev_pack_lim;

  // Base-point snapshot for run_JV/run_JtV's vtheta_dp<->theta linearization;
  // allocated lazily on first init_J() call (ST=Real only).
  ExecViewManaged<PT*[NP][NP][NUM_LEV]> m_adj_dp0, m_adj_theta0;
}; //HVfunctorImplST

using HyperviscosityFunctorImpl = HyperviscosityFunctorImplST<ScalarValue>;

} // namespace Homme

#endif // HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP
