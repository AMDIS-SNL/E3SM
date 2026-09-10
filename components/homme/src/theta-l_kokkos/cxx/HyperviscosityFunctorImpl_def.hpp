/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "HyperviscosityFunctorImpl.hpp"

#include "Context.hpp"
#include "FunctorsBuffersManager.hpp"
#include "PhysicalConstants.hpp"
#include "profiling.hpp"

#include "mpi/BoundaryExchange.hpp"
#include "mpi/MpiBuffersManager.hpp"
#include "mpi/Connectivity.hpp"

namespace Homme
{

template<typename ST>
HyperviscosityFunctorImplST<ST>::
HyperviscosityFunctorImplST (const SimulationParams&           params,
                             const ElementsGeometry&           geometry,
                             const ElementsStateST<ST>&        state,
                             const ElementsDerivedStateST<ST>& derived)
 : m_num_elems(state.num_elems())
 , m_data (params.hypervis_subcycle,params.hypervis_subcycle_tom,
		       params.nu_ratio1,params.nu_ratio2,params.nu_top,params.nu,
		       params.nu_p,params.nu_s,params.hypervis_scaling)
 , m_state   (state)
 , m_derived (derived)
 , m_geometry (geometry)
 , m_sphere_ops (Context::singleton().get<SphereOperatorsST<ST>>())
 , m_hvcoord (Context::singleton().get<HybridVCoord>())
 , m_policy_update_states (Homme::get_default_team_policy<ExecSpace,TagUpdateStates>(m_num_elems))
 , m_policy_first_laplace (Homme::get_default_team_policy<ExecSpace,TagFirstLaplaceHV>(m_num_elems))
 , m_policy_pre_exchange (Homme::get_default_team_policy<ExecSpace, TagHyperPreExchange>(m_num_elems))
 , m_policy_nutop_laplace (Homme::get_default_team_policy<ExecSpace, TagNutopLaplace>(m_num_elems))
 , m_policy_nutop_update_states (Homme::get_default_team_policy<ExecSpace,TagNutopUpdateStates>(m_num_elems))
 , m_tu(m_policy_update_states)
{
  init_params(params);

  // Make sure the sphere operators have buffers large enough to accommodate this functor's needs
  m_sphere_ops.allocate_buffers(m_tu);
}

template<typename ST>
HyperviscosityFunctorImplST<ST>::
HyperviscosityFunctorImplST (const int num_elems, const SimulationParams &params)
  : m_num_elems(num_elems)
  , m_data (params.hypervis_subcycle,params.hypervis_subcycle_tom,
		        params.nu_ratio1,params.nu_ratio2,params.nu_top,params.nu,
		        params.nu_p,params.nu_s,params.hypervis_scaling)
  , m_hvcoord (Context::singleton().get<HybridVCoord>())
  , m_policy_update_states (Homme::get_default_team_policy<ExecSpace,TagUpdateStates>(m_num_elems))
  , m_policy_first_laplace (Homme::get_default_team_policy<ExecSpace,TagFirstLaplaceHV>(m_num_elems))
  , m_policy_pre_exchange (Homme::get_default_team_policy<ExecSpace, TagHyperPreExchange>(m_num_elems))
  , m_policy_nutop_laplace (Homme::get_default_team_policy<ExecSpace, TagNutopLaplace>(m_num_elems))
  , m_policy_nutop_update_states (Homme::get_default_team_policy<ExecSpace,TagNutopUpdateStates>(m_num_elems))
  , m_tu(m_policy_update_states)
{
  init_params(params);
}

template<typename ST>
void HyperviscosityFunctorImplST<ST>::init_params(const SimulationParams& params)
{
  // Sanity check
  assert(params.params_set);

  if (m_data.nu_top>0) {

    m_nu_scale_top = ExecViewManaged<PT[NUM_LEV]>("nu_scale_top");
    auto h_nu_scale_top = Kokkos::create_mirror_view(m_nu_scale_top);

    const auto etai_h = Kokkos::create_mirror_view(m_hvcoord.etai);
    const auto etam_h = Kokkos::create_mirror_view(m_hvcoord.etam);
    Kokkos::deep_copy(etai_h, m_hvcoord.etai);
    Kokkos::deep_copy(etam_h, m_hvcoord.etam);

    for (int phys_lev=0; phys_lev < NUM_LEV*VECTOR_SIZE; ++phys_lev) {
      const int ilev = phys_lev / VECTOR_SIZE;
      const int ivec = phys_lev % VECTOR_SIZE;

      ST ptop_over_press;

      //prevent padding of nu_scale to get nans to avoid last interface levels
      //of w, phi to be nans, too
      if( phys_lev < NUM_PHYSICAL_LEV ){
        //etai is num_interface_lev, that is, 129 or 73
        //etam is num_lev, so packs
        if ( etai_h(0) == 0.0) {
          ptop_over_press = etam_h(0)[0] / etam_h(ilev)[ivec];
        }else{
          ptop_over_press = etai_h(0) / etam_h(ilev)[ivec];
        }
      }else{
          ptop_over_press = 0.0;
      }

      ST val = 16.0*ptop_over_press*ptop_over_press / (ptop_over_press*ptop_over_press + 1);
      if ( val < 0.15 ) val = 0.0;
      h_nu_scale_top(ilev)[ivec] = val;

      // This is the equivalent of nlev_tom in the F90 code.
      if (val != 0) m_nu_scale_top_ilev_pack_lim = phys_lev + 1;
    }
    Kokkos::deep_copy(m_nu_scale_top, h_nu_scale_top);

    // Convert to pack index.
    m_nu_scale_top_ilev_pack_lim = ((m_nu_scale_top_ilev_pack_lim + VECTOR_SIZE - 1) /
                                    VECTOR_SIZE);
  }

  // Init ElementOps
  m_elem_ops.init(m_hvcoord);

  // Init Equation of state
  m_eos.init(params.theta_hydrostatic_mode,m_hvcoord);

#ifdef HOMMEXX_BFB_TESTING
  m_process_nh_vars = 1;
#else
  m_process_nh_vars = not params.theta_hydrostatic_mode;
#endif

#ifdef HOMMEXX_TEST_HYPERVIS_FAD
  // Make this a runtime check, as we maybe doing ETI also for ST=Real, even if unused
  // Then, don't compile the code that actually inits the Fad.
  EKAT_REQUIRE_MSG((std::is_same_v<ST,DpFadType>),
      "Error! The test for Dp fads requires a Fad scalar type");
  if constexpr (std::is_same_v<ST,DpFadType>) {
    m_data.rel_perturb.fastAccessDx(0) = 1;
    m_data.rel_perturb.val() = params.rel_perturb;
    m_data.nu *= (1 + m_data.rel_perturb);
  }
#endif
}

template<typename ST>
void HyperviscosityFunctorImplST<ST>::
setup(const ElementsGeometry&     geometry,
      const ElementsStateST<ST>&        state,
      const ElementsDerivedStateST<ST>& derived)
{
  m_state = state;
  assert(m_num_elems == m_state.num_elems()); //Sanity check
  m_derived = derived;
  m_geometry = geometry;
  m_sphere_ops = Context::singleton().get<SphereOperatorsST<ST>>();

  // Make sure the sphere operators have buffers large enough to accommodate this functor's needs
  m_sphere_ops.allocate_buffers(m_tu);
}

template<typename ST>
int HyperviscosityFunctorImplST<ST>::requested_buffer_size () const {
  constexpr int size_mid_scalar =   NP*NP*NUM_LEV*VECTOR_SIZE;
  constexpr int size_mid_vector = 2*NP*NP*NUM_LEV*VECTOR_SIZE;
  constexpr int size_int_scalar =   NP*NP*NUM_LEV_P*VECTOR_SIZE;

  // Number of scalar/vector int/mid buffers needed, with size nelems
  const int mid_vectors_nelems = 1;
  const int int_scalars_nelems = 0;
  const int mid_scalars_nelems = 2 + (m_process_nh_vars ? 2 : 0);

  int scl_sz = sizeof(ST) / sizeof(Real);

  const int size = m_num_elems*(mid_scalars_nelems*size_mid_scalar +
                                mid_vectors_nelems*size_mid_vector +
                                int_scalars_nelems*size_int_scalar)*scl_sz;

  return size;
}

template<typename ST>
void HyperviscosityFunctorImplST<ST>::init_buffers (const FunctorsBuffersManager& fbm) {
    EKAT_REQUIRE_MSG(fbm.allocated_size()>=requested_buffer_size(),
        "Error! Buffers size not sufficient.\n");

  constexpr int size_mid_scalar =   NP*NP*NUM_LEV;
  constexpr int size_mid_vector = 2*NP*NP*NUM_LEV;

  auto mem_in = fbm.get_memory();
  PT* mem = reinterpret_cast<PT*>(fbm.get_memory());
  const int nelems = m_geometry.num_elems();

  // Tens quantities (persistent views => nelems)
  m_buffers.dptens = decltype(m_buffers.dptens)(mem,nelems);
  mem += size_mid_scalar*nelems;

  m_buffers.ttens = decltype(m_buffers.ttens)(mem,nelems);
  mem += size_mid_scalar*nelems;

  if (m_process_nh_vars) {
    m_buffers.wtens = decltype(m_buffers.wtens)(mem,nelems);
    mem += size_mid_scalar*nelems;

    m_buffers.phitens = decltype(m_buffers.phitens)(mem,nelems);
    mem += size_mid_scalar*nelems;
  }

  m_buffers.vtens = decltype(m_buffers.vtens)(mem,nelems);
  mem += size_mid_vector*nelems;

  const int used_mem = reinterpret_cast<Real*>(mem)-mem_in;
  if (used_mem < requested_buffer_size()) {
    printf("[HyperviscosityFunctorImplST] Warning! We used less memory than we said we would: %d instead of %d\n",
           used_mem, requested_buffer_size());
  } else if (used_mem > requested_buffer_size()) {
    std::string msg = "[HyperviscosityFunctorImplST] Error! We used more memory than we said we would: "
                    + std::to_string(used_mem) + " instead of "
                    + std::to_string(requested_buffer_size()) + "\n";
    EKAT_ERROR_MSG(msg);
  }
}

template<typename ST>
void HyperviscosityFunctorImplST<ST>::init_boundary_exchanges () {
  auto bm_exchange = Context::singleton().get<MpiBuffersManagerMap>()[MPI_EXCHANGE];
  const auto& sp = Context::singleton().get<SimulationParams>();
  m_be = std::make_shared<BoundaryExchangeST<ST>>();
  m_be_tom = std::make_shared<BoundaryExchangeST<ST>>();
  m_be->m_label = "Hyperviscosity-std";
  m_be_tom->m_label = "Hyperviscosity-TOM";
  std::shared_ptr<BoundaryExchangeST<ST>> bes[] = {m_be, m_be_tom};
  const int nlevs[] = {NUM_LEV, m_nu_scale_top_ilev_pack_lim};
  for (int i = 0; i < 2; ++i) {
    if (i == 1 && m_data.nu_top <= 0) continue;
    auto be = bes[i];
    be->m_diagnostics_level = sp.internal_diagnostics_level;
    const auto nlev = nlevs[i];
    be->set_buffers_manager(bm_exchange);
    if (m_process_nh_vars) {
      be->set_num_fields(0, 0, 6);
    } else {
      be->set_num_fields(0, 0, 4);
    }
    be->register_field(m_buffers.dptens, nlev);
    be->register_field(m_buffers.ttens, nlev);
    if (m_process_nh_vars) {
      be->register_field(m_buffers.wtens, nlev);
      be->register_field(m_buffers.phitens, nlev);
    }
    be->register_field(m_buffers.vtens, 2, 0, nlev);
    be->registration_completed();
  }
}//initBE

template<typename ST>
void HyperviscosityFunctorImplST<ST>::run (const int np1, const Real dt, const Real eta_ave_w)
{
  m_data.np1 = np1;

  m_data.dt = dt;
  if (m_data.hypervis_subcycle > 0) { 
    m_data.dt_hvs = dt/m_data.hypervis_subcycle;
  }else{
    //won't be used
    m_data.dt_hvs = -1.0;
  }
  if (m_data.hypervis_subcycle_tom > 0) { 
    m_data.dt_hvs_tom = dt/m_data.hypervis_subcycle_tom;
  }else{
    //won't be used
    m_data.dt_hvs_tom = -1.0;
  }
  m_data.eta_ave_w = eta_ave_w;

  // Convert vtheta_dp -> theta
  auto state = m_state;
  Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace>(state.num_elems()),
                       KOKKOS_LAMBDA(const TeamMember& team) {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // theta->vtheta
      auto vtheta = Homme::subview(state.m_vtheta_dp,ie,np1,igp,jgp);
      auto dp = Homme::subview(state.m_dp3d,ie,np1,igp,jgp);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,NUM_LEV),
                           [&](const int ilev) {
        vtheta(ilev) /= dp(ilev);
      });
    });
  });
  Kokkos::fence();

  for (int icycle = 0; icycle < m_data.hypervis_subcycle; ++icycle) {
    GPTLstart("hvf-bhwk");
    biharmonic_wk_theta ();
    GPTLstop("hvf-bhwk");

    Kokkos::parallel_for(m_policy_pre_exchange, *this);
    Kokkos::fence();

    // Exchange
    assert (m_be->is_registration_completed());
    GPTLstart("hvf-bexch");
    m_be->exchange();
    GPTLstop("hvf-bexch");

    // Update states
    Kokkos::parallel_for(m_policy_update_states, *this);
    Kokkos::fence();
  } //subcycle

  // Convert theta back to vtheta, and adjust w at surface
  auto geo = m_geometry;
  auto process_nh_vars = m_process_nh_vars;
  Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace>(state.num_elems()),
                       KOKKOS_LAMBDA(const TeamMember& team) {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // theta->vtheta
      auto vtheta = Homme::subview(state.m_vtheta_dp,ie,np1,igp,jgp);
      auto dp = Homme::subview(state.m_dp3d,ie,np1,igp,jgp);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,NUM_LEV),
                           [&](const int ilev) {
        vtheta(ilev) *= dp(ilev);
      });

      // Fix w at surface:
      // Adjust w_i at the surface, since velocity has changed
      if (process_nh_vars) {
        Kokkos::single(Kokkos::PerThread(team),[&](){
          using InfoI = ColInfo<NUM_INTERFACE_LEV>;
          using InfoM = ColInfo<NUM_PHYSICAL_LEV>;
          constexpr int LAST_MID_PACK     = InfoM::LastPack;
          constexpr int LAST_MID_PACK_END = InfoM::LastPackEnd;
          constexpr int LAST_INT_PACK     = InfoI::LastPack;
          constexpr int LAST_INT_PACK_END = InfoI::LastPackEnd;
          constexpr Real g = PhysicalConstants::g;

          const auto& grad_x = geo.m_gradphis(ie,0,igp,jgp);
          const auto& grad_y = geo.m_gradphis(ie,1,igp,jgp);
          const auto& u = state.m_v(ie,np1,0,igp,jgp,LAST_MID_PACK)[LAST_MID_PACK_END];
          const auto& v = state.m_v(ie,np1,1,igp,jgp,LAST_MID_PACK)[LAST_MID_PACK_END];

          auto& w = state.m_w_i(ie,np1,igp,jgp,LAST_INT_PACK)[LAST_INT_PACK_END];

          w = (u*grad_x+v*grad_y) / g;
        });
      }
    });
  });//conversion back to vtheta

  Kokkos::fence();

  // sponge layer 
  if (m_data.nu_top > 0) {
    for (int icycle = 0; icycle < m_data.hypervis_subcycle_tom; ++icycle) {
      // laplace(fields) --> ttens, etc.
      Kokkos::parallel_for(m_policy_nutop_laplace, *this);
      Kokkos::fence();

      // exchange is done on ttens, dptens, vtens, etc.
      assert (m_be->is_registration_completed());
      GPTLstart("hvf-bexch");
      m_be_tom->exchange();
      GPTLstop("hvf-bexch");

      Kokkos::parallel_for(m_policy_nutop_update_states, *this);
      Kokkos::fence();
    }
  } // for sponge layer
} // run()

template<typename ST>
void HyperviscosityFunctorImplST<ST>::biharmonic_wk_theta() const
{
  // For the first laplacian we use a differnt kernel, which uses directly the states
  // at timelevel np1 as inputs, and subtracts the reference states.
  // This way we avoid copying the states to *tens buffers.
  Kokkos::parallel_for(m_policy_first_laplace, *this);
  Kokkos::fence();

  // Exchange
  assert (m_be->is_registration_completed());
  GPTLstart("hvf-bexch");
  m_be->exchange(m_geometry.m_rspheremp);
  GPTLstop("hvf-bexch");

  // Compute second laplacian, tensor or const hv
  const int ne = m_geometry.num_elems();
  if ( m_data.consthv ) {
    auto policy = Homme::get_default_team_policy<ExecSpace,TagSecondLaplaceConstHV>(ne);
    Kokkos::parallel_for(policy, *this);
  }else{
    auto policy = Homme::get_default_team_policy<ExecSpace,TagSecondLaplaceTensorHV>(ne);
    Kokkos::parallel_for(policy, *this);
  }
  Kokkos::fence();
} //biharmonic

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
init_J (const int np1) {
  if (m_adj_dp0.size() == 0) {
    m_adj_dp0    = decltype(m_adj_dp0)   ("HV adjoint dp0",    m_num_elems);
    m_adj_theta0 = decltype(m_adj_theta0)("HV adjoint theta0", m_num_elems);
  }
  auto dp  = ekat::scalarize(m_state.m_dp3d);
  auto vth = ekat::scalarize(m_state.m_vtheta_dp);
  auto dp0 = ekat::scalarize(m_adj_dp0);
  auto th0 = ekat::scalarize(m_adj_theta0);
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0}, {m_num_elems,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(p4_mid, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    const Real dp_ie = dp(ie,np1,ip,jp,k);
    dp0(ie,ip,jp,k) = dp_ie;
    th0(ie,ip,jp,k) = vth(ie,np1,ip,jp,k) / dp_ie;
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
run_JV (const int np1, const StateSnapshot& x, StateSnapshot& y)
{
  EKAT_REQUIRE_MSG(m_data.consthv, "HV run_JV: tensor viscosity (hypervis_scaling!=0) not yet supported.\n");
  EKAT_REQUIRE_MSG(m_data.nu_top<=0, "HV run_JV: sponge layer (nu_top>0) not yet supported.\n");

  linearize_theta_in(x, y);
  for (int icycle=0; icycle<m_data.hypervis_subcycle; ++icycle) {
    run_linear_core(y, true);
  }
  linearize_theta_out(np1, y);
  apply_w_surface_tangent(y);
}

// y = J^T*x, using the same J as run_JV. x and y must be different objects.
template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
run_JtV (const int np1, const StateSnapshot& x, StateSnapshot& y)
{
  EKAT_REQUIRE_MSG(m_data.consthv, "HV run_JtV: tensor viscosity (hypervis_scaling!=0) not yet supported.\n");
  EKAT_REQUIRE_MSG(m_data.nu_top<=0, "HV run_JtV: sponge layer (nu_top>0) not yet supported.\n");

  apply_w_surface_adjoint(x, y);
  linearize_theta_out_adjoint(np1, y);
  for (int icycle=0; icycle<m_data.hypervis_subcycle; ++icycle) {
    run_linear_core(y, false);
  }
  linearize_theta_in_adjoint(y);
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
linearize_theta_in (const StateSnapshot& x, StateSnapshot& y) const {
  Kokkos::deep_copy(y.dp3d, x.dp3d);
  Kokkos::deep_copy(y.v, x.v);
  Kokkos::deep_copy(y.w_i, x.w_i);
  Kokkos::deep_copy(y.phinh_i, x.phinh_i);

  auto dp0  = ekat::scalarize(m_adj_dp0);
  auto th0  = ekat::scalarize(m_adj_theta0);
  auto x_dp = ekat::scalarize(x.dp3d);
  auto x_vt = ekat::scalarize(x.vtheta_dp);
  auto y_th = ekat::scalarize(y.vtheta_dp);
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0}, {m_num_elems,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(p4_mid, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    y_th(ie,ip,jp,k) = (x_vt(ie,ip,jp,k) - th0(ie,ip,jp,k)*x_dp(ie,ip,jp,k)) / dp0(ie,ip,jp,k);
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
linearize_theta_in_adjoint (StateSnapshot& y) const {
  auto dp0  = ekat::scalarize(m_adj_dp0);
  auto th0  = ekat::scalarize(m_adj_theta0);
  auto y_dp = ekat::scalarize(y.dp3d);
  auto y_th = ekat::scalarize(y.vtheta_dp);
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0}, {m_num_elems,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(p4_mid, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    const Real dp_adj = y_dp(ie,ip,jp,k);
    const Real th_adj = y_th(ie,ip,jp,k);
    y_dp(ie,ip,jp,k) = dp_adj - (th0(ie,ip,jp,k)/dp0(ie,ip,jp,k))*th_adj;
    y_th(ie,ip,jp,k) = th_adj / dp0(ie,ip,jp,k);
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
linearize_theta_out (const int np1, StateSnapshot& y) const {
  auto dp_base = ekat::scalarize(m_state.m_dp3d);
  auto vt_base = ekat::scalarize(m_state.m_vtheta_dp);
  auto y_dp = ekat::scalarize(y.dp3d);
  auto y_th = ekat::scalarize(y.vtheta_dp);
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0}, {m_num_elems,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(p4_mid, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    const Real dpf = dp_base(ie,np1,ip,jp,k);
    const Real thf = vt_base(ie,np1,ip,jp,k) / dpf;
    const Real y_theta = y_th(ie,ip,jp,k);
    y_th(ie,ip,jp,k) = y_theta*dpf + thf*y_dp(ie,ip,jp,k);
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
linearize_theta_out_adjoint (const int np1, StateSnapshot& y) const {
  auto dp_base = ekat::scalarize(m_state.m_dp3d);
  auto vt_base = ekat::scalarize(m_state.m_vtheta_dp);
  auto y_dp = ekat::scalarize(y.dp3d);
  auto y_vt = ekat::scalarize(y.vtheta_dp);
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0}, {m_num_elems,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(p4_mid, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    const Real dpf = dp_base(ie,np1,ip,jp,k);
    const Real thf = vt_base(ie,np1,ip,jp,k) / dpf;
    const Real vt_adj = y_vt(ie,ip,jp,k);
    const Real dp_adj = y_dp(ie,ip,jp,k);
    y_vt(ie,ip,jp,k) = dpf * vt_adj;           // -> theta_core_out adjoint
    y_dp(ie,ip,jp,k) = thf * vt_adj + dp_adj;  // -> dp_core_out adjoint
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
apply_w_surface_tangent (StateSnapshot& y) const {
  if (!m_process_nh_vars) return;
  constexpr int last_mid = NUM_PHYSICAL_LEV-1;
  constexpr int last_int = NUM_INTERFACE_LEV-1;
  constexpr Real g = PhysicalConstants::g;
  auto y_v = ekat::scalarize(y.v);
  auto y_w = ekat::scalarize(y.w_i);
  auto gradphis = m_geometry.m_gradphis;
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>;
  auto p3 = md_range_t({0,0,0}, {m_num_elems,NP,NP});
  Kokkos::parallel_for(p3, KOKKOS_LAMBDA (const int ie, const int ip, const int jp) {
    const Real gx = gradphis(ie,0,ip,jp);
    const Real gy = gradphis(ie,1,ip,jp);
    y_w(ie,ip,jp,last_int) = (y_v(ie,0,ip,jp,last_mid)*gx + y_v(ie,1,ip,jp,last_mid)*gy)/g;
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
apply_w_surface_adjoint (const StateSnapshot& x, StateSnapshot& y) const {
  y.deep_copy(x);
  if (!m_process_nh_vars) return;
  constexpr int last_mid = NUM_PHYSICAL_LEV-1;
  constexpr int last_int = NUM_INTERFACE_LEV-1;
  constexpr Real g = PhysicalConstants::g;
  auto y_v = ekat::scalarize(y.v);
  auto y_w = ekat::scalarize(y.w_i);
  auto x_w = ekat::scalarize(x.w_i);
  auto gradphis = m_geometry.m_gradphis;
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>;
  auto p3 = md_range_t({0,0,0}, {m_num_elems,NP,NP});
  Kokkos::parallel_for(p3, KOKKOS_LAMBDA (const int ie, const int ip, const int jp) {
    const Real gx = gradphis(ie,0,ip,jp);
    const Real gy = gradphis(ie,1,ip,jp);
    const Real w_adj = x_w(ie,ip,jp,last_int);
    y_v(ie,0,ip,jp,last_mid) += w_adj*gx/g;
    y_v(ie,1,ip,jp,last_mid) += w_adj*gy/g;
    y_w(ie,ip,jp,last_int) = 0;
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
hv_apply_laplace1 (const StateSnapshot& src, const Real nu_ratio) const {
  const auto policy = Homme::get_default_team_policy<ExecSpace>(m_num_elems);
  const bool nh = m_process_nh_vars;
  auto sphop   = m_sphere_ops;
  auto tu      = m_tu;
  auto dptens  = m_buffers.dptens;
  auto ttens   = m_buffers.ttens;
  auto wtens   = m_buffers.wtens;
  auto phitens = m_buffers.phitens;
  auto vtens   = m_buffers.vtens;
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamMember& team) {
    KernelVariables kv(team, tu);
    sphop.laplace_simple(kv, Homme::subview(src.dp3d,kv.ie), Homme::subview(dptens,kv.ie));
    sphop.laplace_simple(kv, Homme::subview(src.vtheta_dp,kv.ie), Homme::subview(ttens,kv.ie));
    if (nh) {
      sphop.template laplace_simple<NUM_LEV,NUM_LEV_P>(kv, Homme::subview(src.w_i,kv.ie), Homme::subview(wtens,kv.ie));
      sphop.template laplace_simple<NUM_LEV,NUM_LEV_P>(kv, Homme::subview(src.phinh_i,kv.ie), Homme::subview(phitens,kv.ie));
    }
    sphop.vlaplace_sphere_wk_contra(kv, nu_ratio, Homme::subview(src.v,kv.ie), Homme::subview(vtens,kv.ie));
  });
  Kokkos::fence();
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
hv_apply_laplace2 (const Real nu_ratio) const {
  const auto policy = Homme::get_default_team_policy<ExecSpace>(m_num_elems);
  const bool nh = m_process_nh_vars;
  auto sphop   = m_sphere_ops;
  auto tu      = m_tu;
  auto dptens  = m_buffers.dptens;
  auto ttens   = m_buffers.ttens;
  auto wtens   = m_buffers.wtens;
  auto phitens = m_buffers.phitens;
  auto vtens   = m_buffers.vtens;
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamMember& team) {
    KernelVariables kv(team, tu);
    sphop.laplace_simple(kv, Homme::subview(dptens,kv.ie), Homme::subview(dptens,kv.ie));
    sphop.laplace_simple(kv, Homme::subview(ttens,kv.ie), Homme::subview(ttens,kv.ie));
    if (nh) {
      sphop.laplace_simple(kv, Homme::subview(wtens,kv.ie), Homme::subview(wtens,kv.ie));
      sphop.laplace_simple(kv, Homme::subview(phitens,kv.ie), Homme::subview(phitens,kv.ie));
    }
    sphop.vlaplace_sphere_wk_contra(kv, nu_ratio, Homme::subview(vtens,kv.ie), Homme::subview(vtens,kv.ie));
  });
  Kokkos::fence();
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
hv_scale_tens () const {
  const bool nh = m_process_nh_vars;
  const Real nu = m_data.nu, nu_p = m_data.nu_p, nu_s = m_data.nu_s;
  auto dptens  = m_buffers.dptens;
  auto ttens   = m_buffers.ttens;
  auto wtens   = m_buffers.wtens;
  auto phitens = m_buffers.phitens;
  auto vtens   = m_buffers.vtens;
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4 = md_range_t({0,0,0,0}, {m_num_elems,NP,NP,NUM_LEV});
  Kokkos::parallel_for(p4, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    vtens(ie,0,ip,jp,k) *= -nu;
    vtens(ie,1,ip,jp,k) *= -nu;
    ttens(ie,ip,jp,k)   *= -nu;
    dptens(ie,ip,jp,k)  *= -nu_p;
    if (nh) {
      wtens(ie,ip,jp,k)   *= -nu;
      phitens(ie,ip,jp,k) *= -nu_s;
    }
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
hv_add_scaled_tens_to (StateSnapshot& dst) const {
  const bool nh = m_process_nh_vars;
  const Real dt_hvs = m_data.dt_hvs;
  auto rspheremp = m_geometry.m_rspheremp;
  auto dptens  = m_buffers.dptens;
  auto ttens   = m_buffers.ttens;
  auto wtens   = m_buffers.wtens;
  auto phitens = m_buffers.phitens;
  auto vtens   = m_buffers.vtens;
  auto dst_dp  = dst.dp3d;
  auto dst_vth = dst.vtheta_dp;
  auto dst_v   = dst.v;
  auto dst_w   = dst.w_i;
  auto dst_phi = dst.phinh_i;
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4 = md_range_t({0,0,0,0}, {m_num_elems,NP,NP,NUM_LEV});
  Kokkos::parallel_for(p4, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    const auto scale = dt_hvs*rspheremp(ie,ip,jp);
    dst_v(ie,0,ip,jp,k) += scale*vtens(ie,0,ip,jp,k);
    dst_v(ie,1,ip,jp,k) += scale*vtens(ie,1,ip,jp,k);
    dst_vth(ie,ip,jp,k) += scale*ttens(ie,ip,jp,k);
    dst_dp(ie,ip,jp,k)  += scale*dptens(ie,ip,jp,k);
    if (nh) {
      dst_w(ie,ip,jp,k)   += scale*wtens(ie,ip,jp,k);
      dst_phi(ie,ip,jp,k) += scale*phitens(ie,ip,jp,k);
    }
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
hv_scale_state_into_tens (const StateSnapshot& src) const {
  const bool nh = m_process_nh_vars;
  const Real dt_hvs = m_data.dt_hvs;
  auto rspheremp = m_geometry.m_rspheremp;
  auto dptens  = m_buffers.dptens;
  auto ttens   = m_buffers.ttens;
  auto wtens   = m_buffers.wtens;
  auto phitens = m_buffers.phitens;
  auto vtens   = m_buffers.vtens;
  auto src_dp  = src.dp3d;
  auto src_vth = src.vtheta_dp;
  auto src_v   = src.v;
  auto src_w   = src.w_i;
  auto src_phi = src.phinh_i;
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4 = md_range_t({0,0,0,0}, {m_num_elems,NP,NP,NUM_LEV});
  Kokkos::parallel_for(p4, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    const auto scale = dt_hvs*rspheremp(ie,ip,jp);
    vtens(ie,0,ip,jp,k) = scale*src_v(ie,0,ip,jp,k);
    vtens(ie,1,ip,jp,k) = scale*src_v(ie,1,ip,jp,k);
    ttens(ie,ip,jp,k)   = scale*src_vth(ie,ip,jp,k);
    dptens(ie,ip,jp,k)  = scale*src_dp(ie,ip,jp,k);
    if (nh) {
      wtens(ie,ip,jp,k)   = scale*src_w(ie,ip,jp,k);
      phitens(ie,ip,jp,k) = scale*src_phi(ie,ip,jp,k);
    }
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
hv_add_tens_to (StateSnapshot& dst) const {
  const bool nh = m_process_nh_vars;
  auto dptens  = m_buffers.dptens;
  auto ttens   = m_buffers.ttens;
  auto wtens   = m_buffers.wtens;
  auto phitens = m_buffers.phitens;
  auto vtens   = m_buffers.vtens;
  auto dst_dp  = dst.dp3d;
  auto dst_vth = dst.vtheta_dp;
  auto dst_v   = dst.v;
  auto dst_w   = dst.w_i;
  auto dst_phi = dst.phinh_i;
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4 = md_range_t({0,0,0,0}, {m_num_elems,NP,NP,NUM_LEV});
  Kokkos::parallel_for(p4, KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    dst_v(ie,0,ip,jp,k) += vtens(ie,0,ip,jp,k);
    dst_v(ie,1,ip,jp,k) += vtens(ie,1,ip,jp,k);
    dst_vth(ie,ip,jp,k) += ttens(ie,ip,jp,k);
    dst_dp(ie,ip,jp,k)  += dptens(ie,ip,jp,k);
    if (nh) {
      dst_w(ie,ip,jp,k)   += wtens(ie,ip,jp,k);
      dst_phi(ie,ip,jp,k) += phitens(ie,ip,jp,k);
    }
  });
}

template<typename ST>
template<typename MyST>
std::enable_if_t<std::is_same_v<MyST, Real>>
HyperviscosityFunctorImplST<ST>::
run_linear_core (StateSnapshot& y, const bool forward) const {
  if (forward) {
    hv_apply_laplace1(y, m_data.nu_ratio1);          // A1
    m_be->exchange(m_geometry.m_rspheremp);           // E1 (weighted)
    hv_apply_laplace2(m_data.nu_ratio2);              // A2
    hv_scale_tens();                                  // SC1 (-nu family)
    m_be->exchange();                                 // E2 (unweighted)
    hv_add_scaled_tens_to(y);                         // SC2 + residual add
  } else {
    hv_scale_state_into_tens(y);                      // SC2^T = SC2
    m_be->exchange();                                 // E2^T = E2
    hv_scale_tens();                                  // SC1^T = SC1
    hv_apply_laplace2(m_data.nu_ratio2);              // A2^T = A2
    m_be->exchange(m_geometry.m_rspheremp);           // E1^T = E1
    hv_apply_laplace2(m_data.nu_ratio1);              // A1^T = A1 (in place)
    hv_add_tens_to(y);                                // residual add (no extra scale)
  }
}

// Laplace for nu_top
template<typename ST>
KOKKOS_INLINE_FUNCTION
void HyperviscosityFunctorImplST<ST>::operator() (const TagNutopLaplace&, const TeamMember& team) const {
  KernelVariables kv(team, m_tu);

  using MidColumn = decltype(Homme::subview(m_buffers.wtens,0,0,0));

  // Laplacian of layer thickness
  m_sphere_ops.laplace_simple(kv,
                              Homme::subview(m_state.m_dp3d,kv.ie,m_data.np1),
                              Homme::subview(m_buffers.dptens,kv.ie),
                              m_nu_scale_top_ilev_pack_lim);
  // Laplacian of theta
  m_sphere_ops.laplace_simple(kv,
                              Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.np1),
                              Homme::subview(m_buffers.ttens,kv.ie),
                              m_nu_scale_top_ilev_pack_lim);

  if (m_process_nh_vars) {
    // Laplacian of vertical velocity (do not compute last interface)
    m_sphere_ops.template laplace_simple<NUM_LEV,NUM_LEV_P>(kv,
                                                   Homme::subview(m_state.m_w_i,kv.ie,m_data.np1),
                                                   Homme::subview(m_buffers.wtens,kv.ie),
                                                   m_nu_scale_top_ilev_pack_lim);
    // Laplacian of geopotential (do not compute last interface)
    m_sphere_ops.template laplace_simple<NUM_LEV,NUM_LEV_P>(kv,
                                                   Homme::subview(m_state.m_phinh_i,kv.ie,m_data.np1),
                                                   Homme::subview(m_buffers.phitens,kv.ie),
                                                   m_nu_scale_top_ilev_pack_lim);
  }

  // Laplacian of velocity
  m_sphere_ops.vlaplace_sphere_wk_contra(kv, m_data.nu_ratio1,
                                         Homme::subview(m_state.m_v,kv.ie,m_data.np1),
                                         Homme::subview(m_buffers.vtens,kv.ie),
                                         m_nu_scale_top_ilev_pack_lim);

  kv.team_barrier();

  Kokkos::parallel_for(
    Kokkos::TeamThreadRange(kv.team,NP*NP),
    [&] (const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      const auto utens  = Homme::subview(m_buffers.vtens,kv.ie,0,igp,jgp);
      const auto vtens  = Homme::subview(m_buffers.vtens,kv.ie,1,igp,jgp);
      const auto ttens  = Homme::subview(m_buffers.ttens,kv.ie,igp,jgp);
      const auto dptens = Homme::subview(m_buffers.dptens,kv.ie,igp,jgp);
     
      MidColumn wtens, phitens;
      if (m_process_nh_vars) {
        wtens   = Homme::subview(m_buffers.wtens,kv.ie,igp,jgp);
        phitens = Homme::subview(m_buffers.phitens,kv.ie,igp,jgp);
      }

      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, m_nu_scale_top_ilev_pack_lim),
        [&] (const int ilev) {
          
          const auto xf = m_data.dt_hvs_tom  * m_nu_scale_top(ilev) * m_data.nu_top;
          utens(ilev)  *= xf;
          vtens(ilev)  *= xf;
          ttens(ilev)  *= xf;
          dptens(ilev) *= xf;

          if (m_process_nh_vars) {
            wtens(ilev)   *= xf;
            phitens(ilev) *= xf;
          }

        }); // threadvectorrange
    }); // teamthreadrange
} // TagNutopLaplace

template<typename ST>
KOKKOS_INLINE_FUNCTION
void HyperviscosityFunctorImplST<ST>::operator() (const TagNutopUpdateStates&, const TeamMember& team) const {
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

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, m_nu_scale_top_ilev_pack_lim),
                         [&](const int ilev) {
      utens(ilev)   *= rspheremp;
      vtens(ilev)   *= rspheremp;
      ttens(ilev)   *= rspheremp;
      dptens(ilev)  *= rspheremp;
      u(ilev)      += utens(ilev);
      v(ilev)      += vtens(ilev);
      vtheta(ilev) += ttens(ilev);
      dp(ilev)     += dptens(ilev);

      if (m_process_nh_vars) {
        wtens(ilev)   *= rspheremp;
        phitens(ilev) *= rspheremp;
        w(ilev)     += wtens(ilev);
        phi_i(ilev) += phitens(ilev);
      }
    }); // threadvectorrange
  }); // threadteamrange
} // tagUpdateStates2

} // namespace Homme
