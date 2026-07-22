/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_CAAR_FUNCTOR_IMPL_HPP
#define HOMMEXX_CAAR_FUNCTOR_IMPL_HPP

#include "Types.hpp"
#include "Elements.hpp"
#include "ColumnOps.hpp"
#include "Context.hpp"
#include "EquationOfState.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "LimiterFunctor.hpp"
#include "ReferenceElement.hpp"
#include "RKStageData.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "Tracers.hpp"
#include "kokkos_utils.hpp"

#include "mpi/BoundaryExchange.hpp"
#include "mpi/MpiBuffersManager.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/ViewUtils.hpp"

#include "profiling.hpp"
#include "ErrorDefs.hpp"

#include <ekat_pack_kokkos.hpp>

#include <assert.h>

namespace Homme {

template<typename ST>
struct CaarFunctorImplST {
  using PT = PackType<ST>;

  struct Buffers {
    static constexpr int num_3d_scalar_mid_buf = 10;
    static constexpr int num_3d_vector_mid_buf =  5;
    static constexpr int num_3d_scalar_int_buf =  6;
    static constexpr int num_3d_vector_int_buf =  3;

    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   temp;

    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   pnh;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   pi;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   exner;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   div_vdp;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   phi;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   omega_p;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   vort;

    ExecViewUnmanaged<PT* [2][NP][NP][NUM_LEV]  >   grad_exner;
    ExecViewUnmanaged<PT* [2][NP][NP][NUM_LEV]  >   mgrad;
    ExecViewUnmanaged<PT* [2][NP][NP][NUM_LEV]  >   grad_tmp;
    ExecViewUnmanaged<PT* [2][NP][NP][NUM_LEV]  >   vdp;

    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV_P]>   dp_i;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV_P]>   vtheta_i;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV_P]>   dpnh_dp_i;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV_P]>   eta_dot_dpdn;

    ExecViewUnmanaged<PT* [2][NP][NP][NUM_LEV_P]>   v_i;
    ExecViewUnmanaged<PT* [2][NP][NP][NUM_LEV_P]>   grad_phinh_i;
    ExecViewUnmanaged<PT* [2][NP][NP][NUM_LEV_P]>   grad_w_i;

    ExecViewUnmanaged<PT* [2][NP][NP][NUM_LEV]  >   v_tens;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   theta_tens;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV]  >   dp_tens;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV_P]>   w_tens;
    ExecViewUnmanaged<PT*    [NP][NP][NUM_LEV_P]>   phi_tens;
  };

  using deriv_type = ReferenceElement::deriv_type;

  RKStageData                 m_data;

  // The 'm_data.scale2' coeff used for the calculation of w_tens and phi_tens must be replaced
  // with 'm_data.scale1' on the surface (k=NUM_INTERFACE_LEV). To allow pack-level operations
  // in the code, we store a pack containing [[scale2 scale2 ...]  scale1 [garbage] ], to be used
  // at pack NUM_LEV_P

  PT                      m_scale2g_last_int_pack;

  const int           m_num_elems;
  const int           m_rsplit;
  const bool          m_theta_hydrostatic_mode;
  const AdvectionForm m_theta_advection_form;
  const bool          m_pgrad_correction;

  HybridVCoord                m_hvcoord;
  ElementsStateST<ST>         m_state;
  ElementsDerivedStateST<ST>  m_derived;
  ElementsGeometry            m_geometry;
  EquationOfState<>           m_eos;
  Buffers                     m_buffers;
  deriv_type                  m_deriv;

  SphereOperatorsST<ST>       m_sphere_ops;

  struct TagPreExchange {};
  struct TagPostExchange {};

  // Policies
#ifndef NDEBUG
  template<typename Tag>
  using TeamPolicyType = Kokkos::TeamPolicy<ExecSpace,Kokkos::LaunchBounds<512,1>,Tag>;
#else
  template<typename Tag>
  using TeamPolicyType = Kokkos::TeamPolicy<ExecSpace,Tag>;
#endif

  TeamPolicyType<TagPreExchange>   m_policy_pre;

  Kokkos::RangePolicy<ExecSpace, TagPostExchange> m_policy_post;

  TeamUtils<ExecSpace> m_tu;

  Kokkos::Array<std::shared_ptr<BoundaryExchangeST<ST>>, NUM_TIME_LEVELS> m_bes;

  CaarFunctorImplST(const ElementsST<ST> &elements, const TracersST<ST>&/* tracers */,
                    const ReferenceElement &ref_FE, const HybridVCoord &hvcoord,
                    const SphereOperatorsST<ST> &sphere_ops, const SimulationParams& params)
      : m_num_elems(elements.num_elems())
      , m_rsplit(params.rsplit)
      , m_theta_hydrostatic_mode(params.theta_hydrostatic_mode)
      , m_theta_advection_form(params.theta_adv_form)
      , m_pgrad_correction(params.pgrad_correction)
      , m_hvcoord(hvcoord)
      , m_state(elements.m_state)
      , m_derived(elements.m_derived)
      , m_geometry(elements.m_geometry)
      , m_deriv(ref_FE.get_deriv())
      , m_sphere_ops(sphere_ops)
      , m_policy_pre (Homme::get_default_team_policy<ExecSpace,TagPreExchange>(m_num_elems))
      , m_policy_post (0,m_num_elems*NP*NP)
      , m_tu(m_policy_pre)
  {
    // Initialize equation of state
    m_eos.init(params.theta_hydrostatic_mode,m_hvcoord);

    // Make sure the buffers in sph op are large enough for this functor's needs
    m_sphere_ops.allocate_buffers(m_tu);
  }

  CaarFunctorImplST(const int num_elems, const SimulationParams& params)
      : m_num_elems(num_elems)
      , m_rsplit(params.rsplit)
      , m_theta_hydrostatic_mode(params.theta_hydrostatic_mode)
      , m_theta_advection_form(params.theta_adv_form)
      , m_pgrad_correction(params.pgrad_correction)
      , m_policy_pre (Homme::get_default_team_policy<ExecSpace,TagPreExchange>(m_num_elems))
      , m_policy_post (0,num_elems*NP*NP)
      , m_tu(m_policy_pre)
  {}

  void setup (const ElementsST<ST> &elements, const TracersST<ST> &/*tracers*/,
              const ReferenceElement &ref_FE, const HybridVCoord &hvcoord,
              const SphereOperatorsST<ST> &sphere_ops)
  {
    assert(m_num_elems == elements.num_elems()); // Sanity check
    m_hvcoord = hvcoord;
    m_state = elements.m_state;
    m_derived = elements.m_derived;
    m_geometry = elements.m_geometry;
    m_deriv = ref_FE.get_deriv();
    m_sphere_ops = sphere_ops;

    // Initialize equation of state
    m_eos.init(m_theta_hydrostatic_mode,m_hvcoord);

    // Make sure the buffers in sph op are large enough for this functor's needs
    m_sphere_ops.allocate_buffers(m_tu);
  }

  int requested_buffer_size () const {
    // Ask the buffers manager to allocate enough buffers to satisfy Caar's needs
    const int nslots = m_tu.get_num_ws_slots();

    int num_scalar_mid_buf = Buffers::num_3d_scalar_mid_buf;
    int num_scalar_int_buf = Buffers::num_3d_scalar_int_buf;
    int num_vector_mid_buf = Buffers::num_3d_vector_mid_buf;
    int num_vector_int_buf = Buffers::num_3d_vector_int_buf;

    // Depending on rsplit/hydro-mode, we may remove some
    // buffers that are not needed from the counters above.
    if (m_theta_hydrostatic_mode) {
      // pi=pnh, and no wtens/phitens
      num_scalar_mid_buf -= 1;
      num_scalar_int_buf -= 3;

      // No grad_w_i/v_i
      num_vector_int_buf -= 2;
    }
    if (m_rsplit>0) {
      // No theta_i/eta_dot_dpdn
      num_scalar_int_buf -=2;
      if (m_theta_hydrostatic_mode) {
        // No dp_i
        num_scalar_int_buf -= 1;
      }
    }

    int scl_sz = sizeof(ST) / sizeof(Real);

    return (num_scalar_mid_buf  *NP*NP*NUM_LEV  *VECTOR_SIZE*nslots +
            num_scalar_int_buf  *NP*NP*NUM_LEV_P*VECTOR_SIZE*nslots +
            num_vector_mid_buf*2*NP*NP*NUM_LEV  *VECTOR_SIZE*nslots +
            num_vector_int_buf*2*NP*NP*NUM_LEV_P*VECTOR_SIZE*nslots) * scl_sz;
  }

  void init_buffers (const FunctorsBuffersManager& fbm) {
    EKAT_REQUIRE_MSG(fbm.allocated_size()>=requested_buffer_size(), "Error! Buffers size not sufficient.\n");

    PT* mem = reinterpret_cast<PT*>(fbm.get_memory());
    const int nslots = m_tu.get_num_ws_slots();

    // Midpoints scalars
    m_buffers.pnh        = decltype(m_buffers.pnh       )(mem,nslots);
    mem += m_buffers.pnh.size();
    if (m_theta_hydrostatic_mode) {
      // pi = pnh
      m_buffers.pi = m_buffers.pnh;
    } else {
      m_buffers.pi       = decltype(m_buffers.pi        )(mem,nslots);
      mem += m_buffers.pi.size();
    }

    m_buffers.temp       = decltype(m_buffers.temp      )(mem,nslots);
    mem += m_buffers.temp.size();
    m_buffers.exner      = decltype(m_buffers.exner     )(mem,nslots);
    mem += m_buffers.exner.size();
    m_buffers.phi        = decltype(m_buffers.phi       )(mem,nslots);
    mem += m_buffers.phi.size();
    m_buffers.div_vdp    = decltype(m_buffers.div_vdp   )(mem,nslots);
    mem += m_buffers.div_vdp.size();
    m_buffers.omega_p    = decltype(m_buffers.omega_p   )(mem,nslots);
    mem += m_buffers.omega_p.size();
    m_buffers.vort       = decltype(m_buffers.vort)(mem,nslots);
    mem += m_buffers.vort.size();
    m_buffers.theta_tens = decltype(m_buffers.theta_tens)(mem,nslots);
    mem += m_buffers.theta_tens.size();
    m_buffers.dp_tens    = decltype(m_buffers.dp_tens   )(mem,nslots);
    mem += m_buffers.dp_tens.size();

    // Midpoints vectors
    m_buffers.grad_exner = decltype(m_buffers.grad_exner)(mem,nslots);
    mem += m_buffers.grad_exner.size();
    m_buffers.mgrad = decltype(m_buffers.mgrad)(mem,nslots);
    mem += m_buffers.mgrad.size();
    m_buffers.grad_tmp = decltype(m_buffers.grad_tmp)(mem,nslots);
    mem += m_buffers.grad_tmp.size();

    m_buffers.vdp      = decltype(m_buffers.vdp     )(mem,nslots);
    mem += m_buffers.vdp.size();
    m_buffers.v_tens   = decltype(m_buffers.v_tens  )(mem,nslots);
    mem += m_buffers.v_tens.size();

    // Interface scalars
    if (!m_theta_hydrostatic_mode || m_rsplit==0) {
      m_buffers.dp_i = decltype(m_buffers.dp_i)(mem,nslots);
      mem += m_buffers.dp_i.size();
    }

    if (!m_theta_hydrostatic_mode) {
      m_buffers.dpnh_dp_i = decltype(m_buffers.dpnh_dp_i)(mem,nslots);
      mem += m_buffers.dpnh_dp_i.size();
    }

    if (m_rsplit==0) {
      m_buffers.eta_dot_dpdn = decltype(m_buffers.eta_dot_dpdn)(mem,nslots);
      mem += m_buffers.eta_dot_dpdn.size();
      m_buffers.vtheta_i     = decltype(m_buffers.vtheta_i    )(mem,nslots);
      mem += m_buffers.vtheta_i.size();
    }

    if (!m_theta_hydrostatic_mode) {
      m_buffers.phi_tens     = decltype(m_buffers.phi_tens    )(mem,nslots);
      mem += m_buffers.phi_tens.size();
      m_buffers.w_tens       = decltype(m_buffers.w_tens      )(mem,nslots);
      mem += m_buffers.w_tens.size();
    }

    // Interface vectors
    if (!m_theta_hydrostatic_mode) {
      m_buffers.v_i          = decltype(m_buffers.v_i         )(mem,nslots);
      mem += m_buffers.v_i.size();
      m_buffers.grad_w_i     = decltype(m_buffers.grad_w_i    )(mem,nslots);
      mem += m_buffers.grad_w_i.size();
    }
    m_buffers.grad_phinh_i = decltype(m_buffers.grad_phinh_i)(mem,nslots);
    mem += m_buffers.grad_phinh_i.size();

    assert ((reinterpret_cast<Real*>(mem) - fbm.get_memory())==requested_buffer_size());
  }

  void init_boundary_exchanges (const std::shared_ptr<MpiBuffersManager>& bm_exchange) {
    const auto& sp = Context::singleton().get<SimulationParams>();
    for (int tl=0; tl<NUM_TIME_LEVELS; ++tl) {
      m_bes[tl] = std::make_shared<BoundaryExchangeST<ST>>();
      auto& be = *m_bes[tl];
      be.m_label = std::string("CAAR-") + std::to_string(tl);
      be.m_diagnostics_level = sp.internal_diagnostics_level;
      be.set_buffers_manager(bm_exchange);
      if (m_theta_hydrostatic_mode) {
        be.set_num_fields(0,0,4);
      } else {
        be.set_num_fields(0,0,4,2);
      }
      be.register_field(m_state.m_v,tl,2,0);
      be.register_field(m_state.m_vtheta_dp,1,tl);
      be.register_field(m_state.m_dp3d,1,tl);
      if (!m_theta_hydrostatic_mode) {
        // Note: phinh_i at the surface (last level) is constant, so it doesn't *need* bex.
        //       If bex(constant)=constant, we might just do it. This would not eliminate
        //       the need for halo-exchange of interface-based quantities though, since
        //       we would still need to exchange w_i.
        be.register_field(m_state.m_w_i,1,tl);
        be.register_field(m_state.m_phinh_i,1,tl);
      }
      be.registration_completed();
    }
  }

  void set_rk_stage_data (const RKStageData& data) {
    m_data = data;

    // Set m_scale2 to m_data.scale2 everywhere, except on last interface,
    // where we set it to m_data.scale1. While at it, we already multiply by g.
    constexpr auto g = PhysicalConstants::g;

    m_scale2g_last_int_pack = m_data.scale2*g;
    m_scale2g_last_int_pack[ColInfo<NUM_INTERFACE_LEV>::LastPackEnd] = m_data.scale1*g;
  }

  void run_pre_exchange (const RKStageData& data)
  {
    set_rk_stage_data(data);

    GPTLstart("caar compute");
    int nerr;
    Kokkos::parallel_reduce("caar loop pre-boundary exchange", m_policy_pre, *this, nerr);
    Kokkos::fence();
    GPTLstop("caar compute");
    if (nerr > 0)
      m_state.check_print_abort_on_bad_elems("CaarFunctorImpl::run TagPreExchange", data.n0);
  }

  void run (const RKStageData& data)
  {
    auto& limiter  = Context::singleton().get<LimiterFunctorST<ST>>();

    set_rk_stage_data(data);

    GPTLstart("caar compute");
    int nerr;
    Kokkos::parallel_reduce("caar loop pre-boundary exchange", m_policy_pre, *this, nerr);
    Kokkos::fence();
    GPTLstop("caar compute");
    if (nerr > 0)
      m_state.check_print_abort_on_bad_elems("CaarFunctorImpl::run TagPreExchange", data.n0);

    GPTLstart("caar_bexchV");
    m_bes[data.np1]->exchange(m_geometry.m_rspheremp);
    Kokkos::fence();
    GPTLstop("caar_bexchV");

    if (!m_theta_hydrostatic_mode) {
      GPTLstart("caar compute");
      Kokkos::parallel_for("caar loop post-boundary exchange", m_policy_post, *this);
      Kokkos::fence();
      GPTLstop("caar compute");
    }

    limiter.run(data.np1);
  }

#ifdef HOMMEXX_ENABLE_FAD_TYPES
  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST,DxFadTypeCaar>>
  init_J (const RKStageData& data);

  // Init J derivatives. Fad entries are initialized to diagonal, using a compressed-column approach
  // The column-compressed Fad, relies on the fact that levels more than 2 above or 1 below
  // will not interact with the current level (while we have full NPxNP coupling in the horiz direction).
  // Moreover, for each gauss point, we don't need the derivs of ALL vars at all 4 levels of the
  // stencil. In fact, the state at k-th level (be that midpoint or interface) depends only on
  // [u_prev, u_curr, v_prev, v_curr,
  //  vth_prev, vth_curr, vth_next,
  //  dp_prev, dp_curr, dp_next,
  //  phi_prev, phi_curr, phi_next, phi_next2,
  //  w_curr, w_next]
  // where _prev refers to lev k-1, _curr to lev k, _next to lev k+1, and _next2 to lev k+2.
  // So the stencil_sz is 16. To compress columns, we interpret the derivs as follows. Let
  //  - kmN_M := (k-N) % M
  //  - k_M   := k % M
  //  - kpN_M := (k+N) % M
  // and for state X let X_blah=X(blah). Then the stencil is
  // [u_km1_2, u_k_2, v_km1_2, v_k_2,
  //  vth_km1_3, vth_k_3, vth_kp1_3,
  //  dp_km1_3, dp_k_3, dp_kp1_3,
  //  phi_km1_3, phi_k_4, phi_kp1_4, phi_kp2_4,
  //  w_k_2, w_kp1_2]
  // One would think that, since mod arith on neg numbers doesn't work well, we should do
  //  kmN_M := (k+M-N) % M
  // However, if k<N, we will NOT use those deriv entries, so we don't need to be careful with mod arith.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST,DxFadTypeCaar>>
  init_J (const RKStageData& data)
  {
    constexpr int stencil_sz = 16;

    int offset_u   = 0;
    int offset_v   = 2;
    int offset_vth = 4;
    int offset_dp  = 7;
    int offset_phi = 10;
    int offset_w   = 14;

    using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
    auto p4_mid = md_range_t({0,0,0,0},{m_num_elems,NP,NP,NUM_PHYSICAL_LEV});
    auto p4_int = md_range_t({0,0,0,0},{m_num_elems,NP,NP,NUM_INTERFACE_LEV});

    auto dvdx_v = ekat::scalarize(m_state.m_v);
    auto dvthdx_v = ekat::scalarize(m_state.m_vtheta_dp);
    auto ddpdx_v = ekat::scalarize(m_state.m_dp3d);
    auto dphidx_v = ekat::scalarize(m_state.m_phinh_i);
    auto dwdx_v = ekat::scalarize(m_state.m_w_i);

    int n0 = data.n0;
    auto init_dx_mid = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
      int pt_offset = ip*NP*stencil_sz + jp*stencil_sz;

      auto& dudx   = dvdx_v  (ie,n0,0,ip,jp,k);
      auto& dvdx   = dvdx_v  (ie,n0,1,ip,jp,k);
      auto& dvthdx = dvthdx_v(ie,n0,  ip,jp,k);
      auto& ddpdx  = ddpdx_v (ie,n0,  ip,jp,k);

      dudx.zero();
      dvdx.zero();
      dvthdx.zero();
      ddpdx.zero();

      dudx.fastAccessDx   (pt_offset + offset_u   + k%2) = 1;
      dvdx.fastAccessDx   (pt_offset + offset_v   + k%2) = 1;
      dvthdx.fastAccessDx (pt_offset + offset_vth + k%3) = 1;
      ddpdx.fastAccessDx  (pt_offset + offset_dp  + k%3) = 1;
    };
    auto init_dx_int = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
      int pt_offset = ip*NP*stencil_sz + jp*stencil_sz;

      auto& dphidx = dphidx_v(ie,n0,ip,jp,k);
      auto& dwdx   = dwdx_v  (ie,n0,ip,jp,k);

      dphidx.zero();
      dwdx.zero();

      dphidx.fastAccessDx(pt_offset + offset_phi + k%4) = 1;
      dwdx.fastAccessDx  (pt_offset + offset_w   + k%2) = 1;
    };

    Kokkos::parallel_for(p4_mid,init_dx_mid);
    Kokkos::parallel_for(p4_int,init_dx_int);
  }

  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST,DxFadTypeCaar>>
  run_JV (const RKStageData& data, const StateSnapshot& x, StateSnapshot& y) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST,DxFadTypeCaar>>
  run_JV (const RKStageData& data, const StateSnapshot& x, StateSnapshot& y)
  {
    constexpr int stencil_sz = 16;

    int offset_u   = 0;
    int offset_v   = 2;
    int offset_vth = 4;
    int offset_dp  = 7;
    int offset_phi = 10;
    int offset_w   = 14;

    using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
    auto p4_mid = md_range_t({0,0,0,0},{m_num_elems,NP,NP,NUM_PHYSICAL_LEV});
    auto p4_int = md_range_t({0,0,0,0},{m_num_elems,NP,NP,NUM_INTERFACE_LEV});

    auto dvdx_v = ekat::scalarize(m_state.m_v);
    auto dvthdx_v = ekat::scalarize(m_state.m_vtheta_dp);
    auto ddpdx_v = ekat::scalarize(m_state.m_dp3d);
    auto dphidx_v = ekat::scalarize(m_state.m_phinh_i);
    auto dwdx_v = ekat::scalarize(m_state.m_w_i);

    // Compute Y = dxnew/dxold * X
    int np1 = data.np1;

    auto x_V = ekat::scalarize(x.v);
    auto x_vth = ekat::scalarize(x.vtheta_dp);
    auto x_dp = ekat::scalarize(x.dp3d);
    auto x_phi = ekat::scalarize(x.phinh_i);
    auto x_w = ekat::scalarize(x.w_i);

    auto y_V = ekat::scalarize(y.v);
    auto y_vth = ekat::scalarize(y.vtheta_dp);
    auto y_dp = ekat::scalarize(y.dp3d);
    auto y_phi = ekat::scalarize(y.phinh_i);
    auto y_w = ekat::scalarize(y.w_i);

    constexpr int last_mid = NUM_PHYSICAL_LEV-1;
    auto prod_rule_mid = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {

      const auto& x_u_ie   = Homme::subview(x_V,ie,0);
      const auto& x_v_ie   = Homme::subview(x_V,ie,1);
      const auto& x_vth_ie = Homme::subview(x_vth,ie);
      const auto& x_dp_ie  = Homme::subview(x_dp,ie);
      const auto& x_phi_ie = Homme::subview(x_phi,ie);
      const auto& x_w_ie   = Homme::subview(x_w,ie);

      auto& y_u_val   = y_V(ie,0,ipt,jpt,k);
      auto& y_v_val   = y_V(ie,1,ipt,jpt,k);
      auto& y_vth_val = y_vth(ie,ipt,jpt,k);
      auto& y_dp_val  = y_dp(ie,ipt,jpt,k);

      // Jacobians
      const auto& Ju   = dvdx_v(ie,np1,0,ipt,jpt,k).dx();
      const auto& Jv   = dvdx_v(ie,np1,1,ipt,jpt,k).dx();
      const auto& Jvth = dvthdx_v(ie,np1,ipt,jpt,k).dx();
      const auto& Jdp  = ddpdx_v(ie,np1,ipt,jpt,k).dx();
      const auto& Jphi = dphidx_v(ie,np1,ipt,jpt,k).dx();
      const auto& Jw   = dwdx_v(ie,np1,ipt,jpt,k).dx();

      y_u_val = 0;
      y_v_val = 0;
      y_dp_val = 0;
      y_vth_val = 0;

      // offsets of each var in the deriv vector
      int u_prev    = offset_u   + (k-1) % 2;
      int u_curr    = offset_u   +  k    % 2;
      int v_prev    = offset_v   + (k-1) % 2;
      int v_curr    = offset_v   +  k    % 2;
      int vth_prev  = offset_vth + (k-1) % 3;
      int vth_curr  = offset_vth +  k    % 3;
      int vth_next  = offset_vth + (k+1) % 3;
      int dp_prev   = offset_dp  + (k-1) % 3;
      int dp_curr   = offset_dp  +  k    % 3;
      int dp_next   = offset_dp  + (k+1) % 3;
      int phi_prev  = offset_phi + (k-1) % 4;
      int phi_curr  = offset_phi +  k    % 4;
      int phi_next  = offset_phi + (k+1) % 4;
      int phi_next2 = offset_phi + (k+2) % 4;
      int w_curr    = offset_w   +  k    % 2;
      int w_next    = offset_w   + (k+1) % 2;

      for (int mpt=0; mpt<NP; ++mpt) {
        for (int npt=0; npt<NP; ++npt) {
          int pt_offset = mpt*NP*stencil_sz + npt*stencil_sz;

          auto pt_Ju   = Ju   + pt_offset;
          auto pt_Jv   = Jv   + pt_offset;
          auto pt_Jvth = Jvth + pt_offset;
          auto pt_Jdp  = Jdp  + pt_offset;

          // Dependencies on k quantities
          y_u_val += pt_Ju[u_curr]   * x_u_ie  (mpt,npt,k)  // du(k)/du(k)
                   + pt_Ju[v_curr]   * x_v_ie  (mpt,npt,k)  // du(k)/dv(k)
                   + pt_Ju[vth_curr] * x_vth_ie(mpt,npt,k)  // du(k)/dvth(k)
                   + pt_Ju[dp_curr]  * x_dp_ie (mpt,npt,k)  // du(k)/ddp(k)
                   + pt_Ju[phi_curr] * x_phi_ie(mpt,npt,k)  // du(k)/dphi(k)
                   + pt_Ju[w_curr]   * x_w_ie  (mpt,npt,k); // du(k)/dw(k)

          y_v_val += pt_Jv[u_curr]   * x_u_ie  (mpt,npt,k)  // dv(k)/du(k)
                   + pt_Jv[v_curr]   * x_v_ie  (mpt,npt,k)  // dv(k)/dv(k)
                   + pt_Jv[vth_curr] * x_vth_ie(mpt,npt,k)  // dv(k)/dvth(k)
                   + pt_Jv[dp_curr]  * x_dp_ie (mpt,npt,k)  // dv(k)/ddp(k)
                   + pt_Jv[phi_curr] * x_phi_ie(mpt,npt,k)  // dv(k)/dphi(k)
                   + pt_Jv[w_curr]   * x_w_ie  (mpt,npt,k); // dv(k)/dw(k)

          y_vth_val += pt_Jvth[u_curr]   * x_u_ie  (mpt,npt,k)  // dvth(k)/du(k)
                     + pt_Jvth[v_curr]   * x_v_ie  (mpt,npt,k)  // dvth(k)/dv(k)
                     + pt_Jvth[vth_curr] * x_vth_ie(mpt,npt,k)  // dvth(k)/dvth(k)
                     + pt_Jvth[dp_curr]  * x_dp_ie (mpt,npt,k); // dvth(k)/ddp(k)

          y_dp_val += pt_Jdp[u_curr]  * x_u_ie  (mpt,npt,k)  // ddp(k)/du(k)
                    + pt_Jdp[v_curr]  * x_v_ie  (mpt,npt,k)  // ddp(k)/dv(k)
                    + pt_Jdp[dp_curr] * x_dp_ie (mpt,npt,k); // ddp(k)/dp(k)

          // Dependencies on k-1 quantities
          if (k>0) {
            y_u_val += pt_Ju[vth_prev] * x_vth_ie(mpt,npt,k-1)  // du(k)/dvth(k-1)
                     + pt_Ju[dp_prev]  * x_dp_ie (mpt,npt,k-1)  // du(k)/ddp(k-1)
                     + pt_Ju[phi_prev] * x_phi_ie(mpt,npt,k-1); // du(k)/dphi(k-1)

            y_v_val += pt_Jv[vth_prev] * x_vth_ie(mpt,npt,k-1)  // dv(k)/dvth(k-1)
                     + pt_Jv[dp_prev]  * x_dp_ie (mpt,npt,k-1)  // dv(k)/ddp(k-1)
                     + pt_Jv[phi_prev] * x_phi_ie(mpt,npt,k-1); // dv(k)/dphi(k-1)
          }

          // Dependencies on k+1 quantities
          y_u_val += pt_Ju[phi_next] * x_phi_ie(mpt,npt,k+1)  // du(k)/dphi(k+1)
                   + pt_Ju[w_next]   * x_w_ie  (mpt,npt,k+1); // du(k)/dw(k+1)

          y_v_val += pt_Jv[phi_next] * x_phi_ie(mpt,npt,k+1)  // dv(k)/dphi(k+1)
                   + pt_Jv[w_next]   * x_w_ie  (mpt,npt,k+1); // dv(k)/dw(k+1)
          if (k<last_mid) {
            y_u_val += pt_Ju[vth_next]  * x_vth_ie(mpt,npt,k+1)  // du(k)/dvth(k+1)
                     + pt_Ju[dp_next]   * x_dp_ie (mpt,npt,k+1)  // du(k)/ddp(k+1)
                     + pt_Ju[phi_next2] * x_phi_ie(mpt,npt,k+2); // du(k)/dphi(k+2)

            y_v_val += pt_Jv[vth_next]  * x_vth_ie(mpt,npt,k+1)  // dv(k)/dvth(k+1)
                     + pt_Jv[dp_next]   * x_dp_ie (mpt,npt,k+1)  // dv(k)/ddp(k+1)
                     + pt_Jv[phi_next2] * x_phi_ie(mpt,npt,k+2); // dv(k)/dphi(k+2)
          }
        }
      }
    };

    constexpr int last_int = NUM_INTERFACE_LEV-1;
    auto prod_rule_int = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
      const auto& x_u_ie   = Homme::subview(x_V,ie,0);
      const auto& x_v_ie   = Homme::subview(x_V,ie,1);
      const auto& x_vth_ie = Homme::subview(x_vth,ie);
      const auto& x_dp_ie  = Homme::subview(x_dp,ie);
      const auto& x_phi_ie = Homme::subview(x_phi,ie);
      const auto& x_w_ie   = Homme::subview(x_w,ie);

      auto& y_phi_val = y_phi(ie,ipt,jpt,k);
      auto& y_w_val   = y_w(ie,ipt,jpt,k);

      // Jacobians
      const auto& Jphi = dphidx_v(ie,np1,ipt,jpt,k).dx();
      const auto& Jw   = dwdx_v(ie,np1,ipt,jpt,k).dx();

      y_phi_val = 0;
      y_w_val = 0;

      // offsets of each var in the deriv vector
      int u_prev    = offset_u   + (k-1) % 2;
      int u_curr    = offset_u   +  k    % 2;
      int v_prev    = offset_v   + (k-1) % 2;
      int v_curr    = offset_v   +  k    % 2;
      int vth_prev  = offset_vth + (k-1) % 3;
      int vth_curr  = offset_vth +  k    % 3;
      int dp_prev   = offset_dp  + (k-1) % 3;
      int dp_curr   = offset_dp  +  k    % 3;
      int phi_prev  = offset_phi + (k-1) % 4;
      int phi_curr  = offset_phi +  k    % 4;
      int phi_next  = offset_phi + (k+1) % 4;
      int w_curr    = offset_w   +  k    % 2;

      for (int mpt=0; mpt<NP; ++mpt) {
        for (int npt=0; npt<NP; ++npt) {
          int pt_offset = mpt*NP*stencil_sz + npt*stencil_sz;

          auto pt_Jphi = Jphi + pt_offset;
          auto pt_Jw   = Jw   + pt_offset;

          // Dependencies on k quantities
          y_phi_val += pt_Jphi[u_curr]   * x_u_ie  (mpt,npt,k)  // dphi(k)/du(k)
                     + pt_Jphi[v_curr]   * x_v_ie  (mpt,npt,k)  // dphi(k)/dv(k)
                     + pt_Jphi[dp_curr]  * x_dp_ie (mpt,npt,k)  // dphi(k)/ddp(k)
                     + pt_Jphi[phi_curr] * x_phi_ie(mpt,npt,k)  // dphi(k)/dphi(k)
                     + pt_Jphi[w_curr]   * x_w_ie  (mpt,npt,k); // dphi(k)/dw(k)

          y_w_val += pt_Jw[u_curr]   * x_u_ie  (mpt,npt,k)  // dw(k)/du(k)
                   + pt_Jw[v_curr]   * x_v_ie  (mpt,npt,k)  // dw(k)/dv(k)
                   + pt_Jw[vth_curr] * x_vth_ie(mpt,npt,k)  // dw(k)/dvth(k)
                   + pt_Jw[dp_curr]  * x_dp_ie (mpt,npt,k)  // dw(k)/ddp(k)
                   + pt_Jw[phi_curr] * x_phi_ie(mpt,npt,k)  // dw(k)/dphi(k)
                   + pt_Jw[w_curr]   * x_w_ie  (mpt,npt,k); // dw(k)/dw(k)

          // Dependencies on k-1 quantities
          if (k>0) {
            y_phi_val += pt_Jphi[u_prev]  * x_u_ie  (mpt,npt,k-1)  // dphi(k)/du(k-1)
                       + pt_Jphi[v_prev]  * x_v_ie  (mpt,npt,k-1)  // dphi(k)/dv(k-1)
                       + pt_Jphi[dp_prev] * x_dp_ie (mpt,npt,k-1); // dphi(k)/ddp(k-1)

            y_w_val += pt_Jw[u_prev]   * x_u_ie  (mpt,npt,k-1)  // dw(k)/du(k-1)
                     + pt_Jw[v_prev]   * x_v_ie  (mpt,npt,k-1)  // dw(k)/dv(k-1)
                     + pt_Jw[vth_prev] * x_vth_ie(mpt,npt,k-1)  // dw(k)/dvth(k-1)
                     + pt_Jw[dp_prev]  * x_dp_ie (mpt,npt,k-1)  // dw(k)/ddp(k-1)
                     + pt_Jw[phi_prev] * x_phi_ie(mpt,npt,k-1); // dw(k)/dphi(k-1)
          }
          // Dependency on k+1 quantities
          if (k<last_int) {
            y_w_val += pt_Jw[phi_next] * x_phi_ie (mpt,npt,k+1); // dw(k)/dphi(k+1)
          }
        }
      }
    };

    Kokkos::parallel_for(p4_mid,prod_rule_mid);
    Kokkos::parallel_for(p4_int,prod_rule_int);
  }

  // Init J derivatives. Fad entries are initialized to diagonal. No column-compression used
  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST,DxFadTypeCaar>>
  init_J_full (const RKStageData& data) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST,DxFadTypeCaar>>
  init_J_full (const RKStageData& data)
  {
    auto dvdx_v = ekat::scalarize(m_state.m_v);
    auto dvthdx_v = ekat::scalarize(m_state.m_vtheta_dp);
    auto ddpdx_v = ekat::scalarize(m_state.m_dp3d);
    auto dwdx_v = ekat::scalarize(m_state.m_w_i);
    auto dphidx_v = ekat::scalarize(m_state.m_phinh_i);

    int n0 = data.n0;

    const int num_fad = Sacado::StaticSize<DxFadTypeCaar>::value;
    for (int ie=0; ie<m_num_elems; ++ie) {
      int fad_idx = 0;
      for (int igp=0; igp<NP; ++igp) {
        for (int jgp=0; jgp<NP; ++jgp) {
          for (int lvl=0; lvl<NUM_PHYSICAL_LEV; ++lvl) {
            dvdx_v(ie,n0,0,igp,jgp,lvl).zero();
            dvdx_v(ie,n0,1,igp,jgp,lvl).zero();
            dvthdx_v(ie,n0,igp,jgp,lvl).zero();
            ddpdx_v(ie,n0,igp,jgp,lvl).zero();

            dvdx_v(ie,n0,0,igp,jgp,lvl).diff(fad_idx++, num_fad);
            dvdx_v(ie,n0,1,igp,jgp,lvl).diff(fad_idx++, num_fad);
            dvthdx_v(ie,n0,igp,jgp,lvl).diff(fad_idx++, num_fad);
            ddpdx_v(ie,n0,igp,jgp,lvl).diff(fad_idx++, num_fad);
          }
          for (int lvl=0; lvl<NUM_INTERFACE_LEV; ++lvl) {
            dwdx_v(ie,n0,igp,jgp,lvl).zero();
            dphidx_v(ie,n0,igp,jgp,lvl).zero();

            dwdx_v(ie,n0,igp,jgp,lvl).diff(fad_idx++, num_fad);
            dphidx_v(ie,n0,igp,jgp,lvl).diff(fad_idx++, num_fad);
          }
        }
      }
      if (ie == 0)
        std::cout << "total needed fad size is " << fad_idx << std::endl
                  << "NP = " << NP << std::endl
                  << "NUM_PHYSICAL_LEV = " << NUM_PHYSICAL_LEV << std::endl
                  << "NUM_INTERFACE_LEV = " << NUM_INTERFACE_LEV << std::endl;
      if (fad_idx > num_fad) {
        std::cout << "fad_idx == " << fad_idx << " but num_fad == " << num_fad << std::endl;
      }
      assert(fad_idx <= num_fad);
    }
  }

  // A version of run_JV that computes the full element Jacobian instead of using column compression to help debugging
  // Note, to run this you need to set DxFadTypeCaar correctly based on the number of levels.  The formula is:
  //      NP*NP*(NUM_PHYSICAL_LEV*4 + NUM_INTERFACE_LEV*2)
  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST,DxFadTypeCaar>>
  run_JV_full (const RKStageData& data, const StateSnapshot& x, StateSnapshot& y) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST,DxFadTypeCaar>>
  run_JV_full (const RKStageData& data, const StateSnapshot& x, StateSnapshot& y)
  {
    // First, init d/dx derivs
    auto dvdx_v = ekat::scalarize(m_state.m_v);
    auto dvthdx_v = ekat::scalarize(m_state.m_vtheta_dp);
    auto ddpdx_v = ekat::scalarize(m_state.m_dp3d);
    auto dwdx_v = ekat::scalarize(m_state.m_w_i);
    auto dphidx_v = ekat::scalarize(m_state.m_phinh_i);

    int n0 = data.n0;

    const int num_fad = Sacado::StaticSize<DxFadTypeCaar>::value;

    // Then compute dxnew/dp = dxnew/dxold * dxold/dp
    int np1 = data.np1;

    auto x_V = ekat::scalarize(x.v);
    auto x_vth = ekat::scalarize(x.vtheta_dp);
    auto x_dp = ekat::scalarize(x.dp3d);
    auto x_w = ekat::scalarize(x.w_i);
    auto x_phi = ekat::scalarize(x.phinh_i);

    auto y_V = ekat::scalarize(y.v);
    auto y_vth = ekat::scalarize(y.vtheta_dp);
    auto y_dp = ekat::scalarize(y.dp3d);
    auto y_w = ekat::scalarize(y.w_i);
    auto y_phi = ekat::scalarize(y.phinh_i);
    // std::ofstream uf("u.txt");
    // std::ofstream vf("v.txt");
    // std::ofstream vthf("vth.txt");
    // std::ofstream dpf("dp.txt");
    // std::ofstream phif("phi.txt");
    // std::ofstream wf("w.txt");

    // const int elem_to_write = 0;

    for (int ie=0; ie<m_num_elems; ++ie) {
      for (int igp=0; igp<NP; ++igp) {
        for (int jgp=0; jgp<NP; ++jgp) {
          for (int lvl=0; lvl<NUM_PHYSICAL_LEV; ++lvl) {

            // // Write Jacobian matrix to file
            // if (ie == elem_to_write) {
            //   for (int k=0; k<num_fad; ++k) {
            //     uf << dvdx_v(ie,np1,0,igp,jgp,lvl).dx(k) << (k == num_fad-1 ? "\n" : " ");
            //     vf << dvdx_v(ie,np1,1,igp,jgp,lvl).dx(k) << (k == num_fad-1 ? "\n" : " ");
            //     vthf << dvthdx_v(ie,np1,igp,jgp,lvl).dx(k) << (k == num_fad-1 ? "\n" : " ");
            //     dpf << ddpdx_v(ie,np1,igp,jgp,lvl).dx(k) << (k == num_fad-1 ? "\n" : " ");
            //   }
            // }

            // Zero Fad components
            y_V(ie,0,igp,jgp,lvl) = 0;
            y_V(ie,1,igp,jgp,lvl) = 0;
            y_vth(ie,igp,jgp,lvl) = 0;
            y_dp(ie,igp,jgp,lvl) = 0;

            // Compute mat-vec one row at a time
            int fad_idx = 0;

            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_V(ie,0,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,0,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,0,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,0,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,1,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,0,igp,jgp,lvl).dx(fad_idx++) * x_vth(ie,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,0,igp,jgp,lvl).dx(fad_idx++) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_V(ie,0,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,0,igp,jgp,lvl).dx(fad_idx++) * x_w(ie,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,0,igp,jgp,lvl).dx(fad_idx++) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            fad_idx = 0;
            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_V(ie,1,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,0,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,1,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,igp,jgp,lvl).dx(fad_idx++) * x_vth(ie,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,igp,jgp,lvl).dx(fad_idx++) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_V(ie,1,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,1,igp,jgp,lvl).dx(fad_idx++) * x_w(ie,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,igp,jgp,lvl).dx(fad_idx++) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            fad_idx = 0;
            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_vth(ie,igp,jgp,lvl) +=
                    dvthdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,0,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,1,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_vth(ie,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_vth(ie,igp,jgp,lvl) +=
                    dvthdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_w(ie,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            fad_idx = 0;
            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_dp(ie,igp,jgp,lvl) +=
                    ddpdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,0,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,1,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_vth(ie,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_dp(ie,igp,jgp,lvl) +=
                    ddpdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_w(ie,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }
          } // lvl

          for (int lvl=0; lvl<NUM_INTERFACE_LEV; ++lvl) {

            // // Write Jacobian matrix to file
            // if (ie == elem_to_write) {
            //   for (int k=0; k<num_fad; ++k) {
            //     phif << dphidx_v(ie,np1,igp,jgp,lvl).dx(k) << (k == num_fad-1 ? "\n" : " ");
            //     wf << dwdx_v(ie,np1,igp,jgp,lvl).dx(k) << (k == num_fad-1 ? "\n" : " ");
            //   }
            // }

            // Zero Fad components
            y_w(ie,np1,igp,jgp,lvl) = 0;
            y_phi(ie,np1,igp,jgp,lvl) = 0;

            // Compute mat-vec one row at a time
            int fad_idx = 0;

            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_w(ie,igp,jgp,lvl) +=
                    dwdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,0,sigp,sjgp,slvl) +
                    dwdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,1,sigp,sjgp,slvl) +
                    dwdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_vth(ie,sigp,sjgp,slvl) +
                    dwdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_w(ie,igp,jgp,lvl) +=
                    dwdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_w(ie,sigp,sjgp,slvl) +
                    dwdx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            fad_idx = 0;
            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_phi(ie,igp,jgp,lvl) +=
                    dphidx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,0,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_V(ie,1,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_vth(ie,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_phi(ie,igp,jgp,lvl) +=
                    dphidx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_w(ie,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,igp,jgp,lvl).dx(fad_idx++) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }
          } // lvl

        } // j
      } // i
    } // e

    // uf.close();
    // vf.close();
    // vthf.close();
    // dpf.close();
    // phif.close();
    // wf.close();
  }

  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST,DxFadTypeCaar>>
  run_JtV (const RKStageData& data, const StateSnapshot& x, StateSnapshot& y) = delete;

  // Computes the transpose Jacobian-vector product: y = J^T * x,
  // where J = d(x_new)/d(x_old) is the same Jacobian used in run_JV.
  // Uses the identical column-compressed stencil (stencil_sz=16), but reverses
  // the accumulation: for each input point (mpt,npt,m) we sum over all Jacobian
  // rows (ipt,jpt,k) whose stencil includes that input point.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST,DxFadTypeCaar>>
  run_JtV (const RKStageData& data, const StateSnapshot& x, StateSnapshot& y)
  {
    // The same compressed-column consideration as in run_JV holds, and everything looks "the same".
    // There is one important distinction. The indices (i,j,k) for y must now match the indices
    // in the deriv, rather than the indices of the var beeing differentiated. That's b/c
    //
    //  [J *v]_i = J_ij * v_j = d(x_i)/d(x_j) * v_j
    //  [J'*v]_i = J_ji * v_j = d(x_j)/d(x_i) * v_j
    //
    // For the gauss points, this is easy to do, as all GPs are coupled with all GPs.
    // For the levels, since we are doing compression, we need to be careful: while in J*v
    // we considered v_j at the levels that can affect x_i, in J'*v we have to consider
    // v_j at the levels that x_i can affect.
    constexpr int stencil_sz = 16;

    int offset_u   = 0;
    int offset_v   = 2;
    int offset_vth = 4;
    int offset_dp  = 7;
    int offset_phi = 10;
    int offset_w   = 14;

    using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
    auto p4_mid = md_range_t({0,0,0,0},{m_num_elems,NP,NP,NUM_PHYSICAL_LEV});
    auto p4_int = md_range_t({0,0,0,0},{m_num_elems,NP,NP,NUM_INTERFACE_LEV});

    auto dvdx_v = ekat::scalarize(m_state.m_v);
    auto dvthdx_v = ekat::scalarize(m_state.m_vtheta_dp);
    auto ddpdx_v = ekat::scalarize(m_state.m_dp3d);
    auto dphidx_v = ekat::scalarize(m_state.m_phinh_i);
    auto dwdx_v = ekat::scalarize(m_state.m_w_i);

    // Compute Y = d(state_new)/d(state_old) * X
    int np1 = data.np1;

    auto x_V = ekat::scalarize(x.v);
    auto x_vth = ekat::scalarize(x.vtheta_dp);
    auto x_dp = ekat::scalarize(x.dp3d);
    auto x_phi = ekat::scalarize(x.phinh_i);
    auto x_w = ekat::scalarize(x.w_i);

    auto y_V = ekat::scalarize(y.v);
    auto y_vth = ekat::scalarize(y.vtheta_dp);
    auto y_dp = ekat::scalarize(y.dp3d);
    auto y_phi = ekat::scalarize(y.phinh_i);
    auto y_w = ekat::scalarize(y.w_i);

    constexpr int last_mid = NUM_PHYSICAL_LEV-1;
    auto jtv_mid = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
      const auto& x_u_ie   = Homme::subview(x_V,ie,0);
      const auto& x_v_ie   = Homme::subview(x_V,ie,1);
      const auto& x_vth_ie = Homme::subview(x_vth,ie);
      const auto& x_dp_ie  = Homme::subview(x_dp,ie);
      const auto& x_phi_ie = Homme::subview(x_phi,ie);
      const auto& x_w_ie   = Homme::subview(x_w,ie);

      auto& y_u_val   = y_V(ie,0,ipt,jpt,k);
      auto& y_v_val   = y_V(ie,1,ipt,jpt,k);
      auto& y_vth_val = y_vth(ie,ipt,jpt,k);
      auto& y_dp_val  = y_dp(ie,ipt,jpt,k);

      // Jacobians
      const auto& Ju   = Homme::subview(dvdx_v,ie,np1,0);
      const auto& Jv   = Homme::subview(dvdx_v,ie,np1,1);
      const auto& Jvth = Homme::subview(dvthdx_v,ie,np1);
      const auto& Jdp  = Homme::subview(ddpdx_v,ie,np1);
      const auto& Jphi = Homme::subview(dphidx_v,ie,np1);
      const auto& Jw   = Homme::subview(dwdx_v,ie,np1);

      y_u_val = 0;
      y_v_val = 0;
      y_dp_val = 0;
      y_vth_val = 0;

      // offsets of each var in the deriv vector
      // NOTE: to do (X-m) mod n, do (X-m+n) mod n, since the mod of a neg number is not what you think
      int pt_offset = ipt*NP*stencil_sz + jpt*stencil_sz;

      int u_curr    = pt_offset + offset_u   + k % 2;
      int v_curr    = pt_offset + offset_v   + k % 2;
      int vth_curr  = pt_offset + offset_vth + k % 3;
      int dp_curr   = pt_offset + offset_dp  + k % 3;
      int phi_curr  = pt_offset + offset_phi + k % 4;
      int w_curr    = pt_offset + offset_w   + k % 2;

      for (int mpt=0; mpt<NP; ++mpt) {
        for (int npt=0; npt<NP; ++npt) {
          auto pt_Ju   = Homme::subview(Ju   ,mpt,npt);
          auto pt_Jv   = Homme::subview(Jv   ,mpt,npt);
          auto pt_Jvth = Homme::subview(Jvth ,mpt,npt);
          auto pt_Jdp  = Homme::subview(Jdp  ,mpt,npt);
          auto pt_Jphi = Homme::subview(Jphi ,mpt,npt);
          auto pt_Jw   = Homme::subview(Jw   ,mpt,npt);

          // Impacts on k quantities
          y_u_val += pt_Ju(k).dx(u_curr)   * x_u_ie  (mpt,npt,k)  // du(k)/du(k)
                   + pt_Jv(k).dx(u_curr)   * x_v_ie  (mpt,npt,k)  // dv(k)/du(k)
                   + pt_Jvth(k).dx(u_curr) * x_vth_ie(mpt,npt,k)  // dvth(k)/du(k)
                   + pt_Jdp(k).dx(u_curr)  * x_dp_ie (mpt,npt,k)  // ddp(k)/du(k)
                   + pt_Jphi(k).dx(u_curr) * x_phi_ie(mpt,npt,k)  // dphi(k)/du(k)
                   + pt_Jw(k).dx(u_curr)   * x_w_ie  (mpt,npt,k); // dw(k)/du(k)

          y_v_val += pt_Ju(k).dx(v_curr)   * x_u_ie  (mpt,npt,k)  // du(k)/du(k)
                   + pt_Jv(k).dx(v_curr)   * x_v_ie  (mpt,npt,k)  // dv(k)/du(k)
                   + pt_Jvth(k).dx(v_curr) * x_vth_ie(mpt,npt,k)  // dvth(k)/du(k)
                   + pt_Jdp(k).dx(v_curr)  * x_dp_ie (mpt,npt,k)  // ddp(k)/du(k)
                   + pt_Jphi(k).dx(v_curr) * x_phi_ie(mpt,npt,k)  // dphi(k)/du(k)
                   + pt_Jw(k).dx(v_curr)   * x_w_ie  (mpt,npt,k); // dw(k)/du(k)

          y_vth_val += pt_Ju(k).dx(vth_curr)   * x_u_ie  (mpt,npt,k)  // du(k)/dvth(k)
                    +  pt_Jv(k).dx(vth_curr)   * x_v_ie  (mpt,npt,k)  // dv(k)/dvth(k)
                    +  pt_Jvth(k).dx(vth_curr) * x_vth_ie(mpt,npt,k)  // dvth(k)/dvth(k)
                    +  pt_Jw(k).dx(vth_curr)   * x_w_ie  (mpt,npt,k); // dw(k)/dvth(k)

          y_dp_val += pt_Ju(k).dx(dp_curr)   * x_u_ie  (mpt,npt,k)  // du(k)/ddp(k)
                   +  pt_Jv(k).dx(dp_curr)   * x_v_ie  (mpt,npt,k)  // dv(k)/ddp(k)
                   +  pt_Jvth(k).dx(dp_curr) * x_vth_ie(mpt,npt,k)  // dvth(k)/ddp(k)
                   +  pt_Jdp(k).dx(dp_curr)  * x_dp_ie (mpt,npt,k)  // ddp(k)/ddp(k)
                   +  pt_Jphi(k).dx(dp_curr) * x_phi_ie(mpt,npt,k)  // dphi(k)/ddp(k)
                   +  pt_Jw(k).dx(dp_curr)   * x_w_ie  (mpt,npt,k); // dw(k)/ddp(k)

          // Impacts on k+1 quantities
          y_u_val += pt_Jphi(k+1).dx(u_curr) * x_phi_ie(mpt,npt,k+1)  // dphi(k+1)/du(k)
                   + pt_Jw(k+1).dx(u_curr)   * x_w_ie  (mpt,npt,k+1); // dw(k+1)/du(k)

          y_v_val += pt_Jphi(k+1).dx(v_curr) * x_phi_ie(mpt,npt,k+1)  // dphi(k+1)/dv(k)
                   + pt_Jw(k+1).dx(v_curr)   * x_w_ie  (mpt,npt,k+1); // dw(k+1)/dv(k)

          y_vth_val += pt_Jw(k+1).dx(vth_curr) * x_w_ie(mpt,npt,k+1); // dw(k+1)/dvth(k)

          y_dp_val += pt_Jphi(k+1).dx(dp_curr) * x_phi_ie(mpt,npt,k+1)  // dphi(k+1)/ddp(k)
                    + pt_Jw(k+1).dx(dp_curr)   * x_w_ie  (mpt,npt,k+1); // dw(k+1)/ddp(k)

          if (k<last_mid) {
            y_vth_val += pt_Ju(k+1).dx(vth_curr) * x_u_ie(mpt,npt,k+1)  // du(k+1)/dvth(k)
                       + pt_Jv(k+1).dx(vth_curr) * x_v_ie(mpt,npt,k+1); // dv(k+1)/dvth(k)

            y_dp_val += pt_Ju(k+1).dx(dp_curr) * x_u_ie(mpt,npt,k+1)  // du(k+1)/ddp(k)
                      + pt_Jv(k+1).dx(dp_curr) * x_v_ie(mpt,npt,k+1); // dv(k+1)/ddp(k)
          }

          // Impacts on k-1 quantities
          if (k>0) {
            y_vth_val += pt_Ju(k-1).dx(vth_curr) * x_u_ie(mpt,npt,k-1)  // du(k-1)/dvth(k)
                       + pt_Jv(k-1).dx(vth_curr) * x_v_ie(mpt,npt,k-1); // dv(k-1)/dvth(k)

            y_dp_val += pt_Ju(k-1).dx(dp_curr) * x_u_ie(mpt,npt,k-1)  // du(k-1)/ddp(k)
                      + pt_Jv(k-1).dx(dp_curr) * x_v_ie(mpt,npt,k-1); // dv(k-1)/ddp(k)
          }
        }
      }
    };

    constexpr int last_int = NUM_INTERFACE_LEV-1;
    auto jtv_int = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
      const auto& x_u_ie   = Homme::subview(x_V,ie,0);
      const auto& x_v_ie   = Homme::subview(x_V,ie,1);
      const auto& x_vth_ie = Homme::subview(x_vth,ie);
      const auto& x_dp_ie  = Homme::subview(x_dp,ie);
      const auto& x_phi_ie = Homme::subview(x_phi,ie);
      const auto& x_w_ie   = Homme::subview(x_w,ie);

      auto& y_phi_val = y_phi(ie,ipt,jpt,k);
      auto& y_w_val   = y_w(ie,ipt,jpt,k);

      // Jacobians
      const auto& Ju   = Homme::subview(dvdx_v,ie,np1,0);
      const auto& Jv   = Homme::subview(dvdx_v,ie,np1,1);
      const auto& Jvth = Homme::subview(dvthdx_v,ie,np1);
      const auto& Jdp  = Homme::subview(ddpdx_v,ie,np1);
      const auto& Jphi = Homme::subview(dphidx_v,ie,np1);
      const auto& Jw   = Homme::subview(dwdx_v,ie,np1);

      y_phi_val = 0;
      y_w_val = 0;

      int pt_offset = ipt*NP*stencil_sz + jpt*stencil_sz;

      int u_curr    = pt_offset + offset_u   + k % 2;
      int v_curr    = pt_offset + offset_v   + k % 2;
      int vth_curr  = pt_offset + offset_vth + k % 3;
      int dp_curr   = pt_offset + offset_dp  + k % 3;
      int phi_curr  = pt_offset + offset_phi + k % 4;
      int w_curr    = pt_offset + offset_w   + k % 2;

      for (int mpt=0; mpt<NP; ++mpt) {
        for (int npt=0; npt<NP; ++npt) {
          auto pt_Ju   = Homme::subview(Ju   ,mpt,npt);
          auto pt_Jv   = Homme::subview(Jv   ,mpt,npt);
          auto pt_Jphi = Homme::subview(Jphi ,mpt,npt);
          auto pt_Jw   = Homme::subview(Jw   ,mpt,npt);

          // Impacts on k interfaces
          y_phi_val += pt_Jphi(k).dx(phi_curr) * x_phi_ie(mpt,npt,k)  // dphi(k)/dphi(k)
                     + pt_Jw(k).dx(phi_curr)   * x_w_ie  (mpt,npt,k); // dw(k)/dphi(k)

          y_w_val += pt_Jphi(k).dx(w_curr) * x_phi_ie(mpt,npt,k)  // dphi(k)/dw(k)
                   + pt_Jw(k).dx(w_curr)   * x_w_ie  (mpt,npt,k); // dw(k)/dw(k)

          // Impacts on k+1 interfaces and k midpoints
          if (k<last_int) {
            y_phi_val += pt_Ju(k).dx(phi_curr)   * x_u_ie(mpt,npt,k)  // du(k)/dphi(k)
                       + pt_Jv(k).dx(phi_curr)   * x_v_ie(mpt,npt,k)  // dv(k)/dphi(k)
                       + pt_Jw(k+1).dx(phi_curr) * x_w_ie(mpt,npt,k+1); // dw(k+1)/dphi(k)

            y_w_val += pt_Ju(k).dx(w_curr) * x_u_ie(mpt,npt,k)  // du(k)/dw(k)
                     + pt_Jv(k).dx(w_curr) * x_v_ie(mpt,npt,k); // dv(k)/dw(k)

            if (k<last_mid) {
              y_phi_val += pt_Ju(k+1).dx(phi_curr) * x_u_ie(mpt,npt,k+1)  // du(k+1)/dphi(k)
                         + pt_Jv(k+1).dx(phi_curr) * x_v_ie(mpt,npt,k+1); // dv(k+1)/dphi(k)
            }
          }

          // Impacts on k-1 quantities
          if (k>0) {
            y_phi_val += pt_Ju(k-1).dx(phi_curr) * x_u_ie(mpt,npt,k-1)  // du(k-1)/dphi(k)
                       + pt_Jv(k-1).dx(phi_curr) * x_v_ie(mpt,npt,k-1)  // dv(k-1)/dphi(k)
                       + pt_Jw(k-1).dx(phi_curr) * x_w_ie(mpt,npt,k-1); // dw(k-1)/dphi(k)

            y_w_val += pt_Ju(k-1).dx(w_curr) * x_u_ie(mpt,npt,k-1)  // du(k-1)/dw(k)
                     + pt_Jv(k-1).dx(w_curr) * x_v_ie(mpt,npt,k-1); // dv(k-1)/dw(k)

            // Impacts on k-2 quantities
            if (k>1) {
              y_phi_val += pt_Ju(k-2).dx(phi_curr) * x_u_ie(mpt,npt,k-2)  // du(k-2)/dphi(k)
                         + pt_Jv(k-2).dx(phi_curr) * x_v_ie(mpt,npt,k-2); // dv(k-2)/dphi(k)
            }
          }
        }
      }
    };

    Kokkos::parallel_for(p4_mid,jtv_mid);
    Kokkos::parallel_for(p4_int,jtv_int);
  }

  // A version of run_JVt that computes the full element Jacobian instead of using column compression to help debugging
  // Note, to run this you need to set DxFadTypeCaar correctly based on the number of levels.  The formula is:
  //      NP*NP*(NUM_PHYSICAL_LEV*4 + NUM_INTERFACE_LEV*2)
  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST,DxFadTypeCaar>>
  run_JtV_full (const RKStageData& data, const StateSnapshot& x, StateSnapshot& y) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST,DxFadTypeCaar>>
  run_JtV_full (const RKStageData& data, const StateSnapshot& x, StateSnapshot& y)
  {
    // First, init d/dx derivs
    auto dvdx_v = ekat::scalarize(m_state.m_v);
    auto dvthdx_v = ekat::scalarize(m_state.m_vtheta_dp);
    auto ddpdx_v = ekat::scalarize(m_state.m_dp3d);
    auto dwdx_v = ekat::scalarize(m_state.m_w_i);
    auto dphidx_v = ekat::scalarize(m_state.m_phinh_i);

    int n0 = data.n0;

    const int num_fad = Sacado::StaticSize<DxFadTypeCaar>::value;

    // Then compute dxnew/dp = dxnew/dxold * dxold/dp
    int np1 = data.np1;

    auto x_V = ekat::scalarize(x.v);
    auto x_vth = ekat::scalarize(x.vtheta_dp);
    auto x_dp = ekat::scalarize(x.dp3d);
    auto x_w = ekat::scalarize(x.w_i);
    auto x_phi = ekat::scalarize(x.phinh_i);

    auto y_V = ekat::scalarize(y.v);
    auto y_vth = ekat::scalarize(y.vtheta_dp);
    auto y_dp = ekat::scalarize(y.dp3d);
    auto y_w = ekat::scalarize(y.w_i);
    auto y_phi = ekat::scalarize(y.phinh_i);

    for (int ie=0; ie<m_num_elems; ++ie) {
      int fad_idx = 0;

      for (int igp=0; igp<NP; ++igp) {
        for (int jgp=0; jgp<NP; ++jgp) {
          for (int lvl=0; lvl<NUM_PHYSICAL_LEV; ++lvl) {

            // Zero Fad components
            y_V(ie,0,igp,jgp,lvl) = 0;
            y_V(ie,1,igp,jgp,lvl) = 0;
            y_vth(ie,igp,jgp,lvl) = 0;
            y_dp(ie,igp,jgp,lvl) = 0;

            // Compute mat-trans-vec one column at a time
            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_V(ie,0,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,0,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,0,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,1,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_vth(ie,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_V(ie,0,igp,jgp,lvl) +=
                    dwdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_w(ie,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            ++fad_idx;

            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_V(ie,1,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,0,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,0,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,1,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_vth(ie,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_V(ie,1,igp,jgp,lvl) +=
                    dwdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_w(ie,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            ++fad_idx;

            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_vth(ie,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,0,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,0,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,1,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_vth(ie,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_vth(ie,igp,jgp,lvl) +=
                    dwdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_w(ie,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            ++fad_idx;

            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_dp(ie,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,0,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,0,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,1,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_vth(ie,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_dp(ie,igp,jgp,lvl) +=
                    dwdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_w(ie,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            ++fad_idx;

          } // lvl

          for (int lvl=0; lvl<NUM_INTERFACE_LEV; ++lvl) {

            // Zero Fad components
            y_w(ie,igp,jgp,lvl) = 0;
            y_phi(ie,igp,jgp,lvl) = 0;

            // Compute mat-trans-vec one column at a time
            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_w(ie,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,0,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,0,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,1,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_vth(ie,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_w(ie,igp,jgp,lvl) +=
                    dwdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_w(ie,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            ++fad_idx;

            for (int sigp=0; sigp<NP; ++sigp) {
              for (int sjgp=0; sjgp<NP; ++sjgp) {
                for (int slvl=0; slvl<NUM_PHYSICAL_LEV; ++slvl) {
                  y_phi(ie,igp,jgp,lvl) +=
                    dvdx_v(ie,np1,0,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,0,sigp,sjgp,slvl) +
                    dvdx_v(ie,np1,1,sigp,sjgp,slvl).dx(fad_idx) * x_V(ie,1,sigp,sjgp,slvl) +
                    dvthdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_vth(ie,sigp,sjgp,slvl) +
                    ddpdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_dp(ie,sigp,sjgp,slvl);
                }
                for (int slvl=0; slvl<NUM_INTERFACE_LEV; ++slvl) {
                  y_phi(ie,igp,jgp,lvl) +=
                    dwdx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_w(ie,sigp,sjgp,slvl) +
                    dphidx_v(ie,np1,sigp,sjgp,slvl).dx(fad_idx) * x_phi(ie,sigp,sjgp,slvl);
                }
              }
            }

            ++fad_idx;

          } // lvl
        } // j
      } // i

      assert(fad_idx <= num_fad);
    } // e
  }
#endif //HOMMEXX_ENABLE_FAD_TYPES

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagPreExchange&, const TeamMember &team, int& nerr) const {
    // In this body, we use '====' to separate sync epochs (delimited by barriers)
    // Note: make sure the same temp is not used within each epoch!

    KernelVariables kv(team, m_tu);

    // =========== EPOCH 1 =========== //
    compute_div_vdp(kv);

    // =========== EPOCH 2 =========== //
    kv.team_barrier();

    // Computes pi, omega, and phi.
    const bool ok = compute_scan_quantities(kv);
    if ( ! ok) nerr = 1;

    if (m_rsplit==0 || !m_theta_hydrostatic_mode) {
      // ============ EPOCH 2.1 =========== //
      kv.team_barrier();
      compute_interface_quantities(kv);
    }

    if (m_rsplit==0) {
      // ============= EPOCH 2.2 ============ //
      kv.team_barrier();
      compute_vertical_advection(kv);
    }

    // ============= EPOCH 3 ============== //
    kv.team_barrier();
    compute_accumulated_quantities(kv);

    // Compute update quantities
    if (!m_theta_hydrostatic_mode) {
      compute_w_and_phi_tens (kv);
    }

    compute_dp_and_theta_tens (kv);

    // ============= EPOCH 4 =========== //
    // compute_v_tens reuses some buffers used by compute_dp_and_theta_tens
    kv.team_barrier();
    compute_v_tens (kv);

    // Update states
    if (!m_theta_hydrostatic_mode) {
      compute_w_and_phi_np1(kv);
    }
    compute_dp3d_and_theta_np1(kv);

    // ============= EPOCH 5 =========== //
    // v_tens has been computed after last barrier. Need to make sure it's done
    kv.team_barrier();
    compute_v_np1(kv);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagPostExchange&, const int idx) const {
    // For g
    using namespace PhysicalConstants;

    using InfoM = ColInfo<NUM_PHYSICAL_LEV>;
    using InfoI = ColInfo<NUM_INTERFACE_LEV>;

    constexpr int LAST_MID_PACK     = InfoM::LastPack;
    constexpr int LAST_MID_PACK_END = InfoM::LastPackEnd;
    constexpr int LAST_INT_PACK     = InfoI::LastPack;
    constexpr int LAST_INT_PACK_END = InfoI::LastPackEnd;

    // Note: make sure you run this only in non-hydro mode
    // KernelVariables kv(team);
    const int ie  = idx / (NP*NP);
    const int igp = (idx / NP) % NP;
    const int jgp =  idx % NP;

    auto& u = m_state.m_v(ie,m_data.np1,0,igp,jgp,LAST_MID_PACK)[LAST_MID_PACK_END];
    auto& v = m_state.m_v(ie,m_data.np1,1,igp,jgp,LAST_MID_PACK)[LAST_MID_PACK_END];
    auto& w = m_state.m_w_i(ie,m_data.np1,igp,jgp,LAST_INT_PACK)[LAST_INT_PACK_END];
    const auto& phis_x = m_geometry.m_gradphis(ie,0,igp,jgp);
    const auto& phis_y = m_geometry.m_gradphis(ie,1,igp,jgp);

    // Compute dpnh_dp_i on surface
    ST dpnh_dp_i = 1 + ( ( (u*phis_x + v*phis_y)/g - w) /
                           (g + (phis_x*phis_x+phis_y*phis_y)/(2*g) ) ) / m_data.dt;

    // Update w_i on bottom interface
    // Update v on bottom level
    w += m_data.scale1*m_data.dt*g*(dpnh_dp_i-1.0);
    u -= m_data.scale1*m_data.dt*(dpnh_dp_i-1.0)*phis_x/2.0;
    v -= m_data.scale1*m_data.dt*(dpnh_dp_i-1.0)*phis_y/2.0;

    // TODO: you need to modify the BoundaryExchange class a bit, cause as of today
    //       it exchanges *all* vertical levels. For phi, we don't want/need to
    //       exchange the last level, since phi=phis at surface.
    //       So to make sure we're not messing up, set phi back to phis on last interface
    // Note: this is *independent* of whether NUM_LEV==NUM_LEV_P or not.
    auto& phi_surf = m_state.m_phinh_i(ie,m_data.np1,igp,jgp,LAST_INT_PACK)[LAST_INT_PACK_END];
    phi_surf = m_geometry.m_phis(ie,igp,jgp);

#if defined(ENERGY_DIAGNOSTICS) && !defined(NDEBUG)
    // Check w bc
    if (fabs( (u*phis_x+v*phis_y)/g - w ) > 1e-10) {
      printf("[CAAR] WARNING! w b.c. not satisfied at (ie,igp,jgp) = (%d,%d,%d):\n"
             "         w:              %3.15f\n"
             "         v*grad(phis)/g: %3.15f\n"
             "         diff:           %3.15f\n",
             ie,igp,jgp,w,(u*phis_x+v*phis_y)/g,fabs( (u*phis_x+v*phis_y)/g - w ));
    }

    auto phi = ekat::scalarize(Homme::subview(m_state.m_phinh_i,ie,m_data.np1,igp,jgp));
    for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
      if ( (phi(k)-phi(k+1)) < g ) {
        printf("[CAAR] WARNING! delta z < 1m, at (ie,igp,jgp,k) = (%d,%d,%d,%d):\n"
               "         phi(k):   %3.15f\n"
               "         phi(k+1): %3.15f\n",
               ie,igp,jgp,k,phi(k),phi(k+1));
      }
    }
#endif
  }

  KOKKOS_INLINE_FUNCTION
  void compute_div_vdp(KernelVariables &kv) const {
    // Compute vdp
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto u = Homme::subview(m_state.m_v,kv.ie,m_data.n0,0,igp,jgp);
      auto v = Homme::subview(m_state.m_v,kv.ie,m_data.n0,1,igp,jgp);
      auto dp3d = Homme::subview(m_state.m_dp3d,kv.ie,m_data.n0,igp,jgp);
      auto udp = Homme::subview(m_buffers.vdp,kv.team_idx,0,igp,jgp);
      auto vdp = Homme::subview(m_buffers.vdp,kv.team_idx,1,igp,jgp);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&] (const int& ilev) {
        udp(ilev) = u(ilev)*dp3d(ilev);
        vdp(ilev) = v(ilev)*dp3d(ilev);
      });
    });
    kv.team_barrier();

    // Compute div(vdp)
    m_sphere_ops.divergence_sphere(kv,
        Homme::subview(m_buffers.vdp, kv.team_idx),
        Homme::subview(m_buffers.div_vdp, kv.team_idx));
  }

  KOKKOS_INLINE_FUNCTION
  bool compute_scan_quantities (KernelVariables &kv) const {
    bool ok = true;

    kv.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // At interfaces, pi_i(k+1) = pi_i(k)+dp(k), with pi_i(0)=hyai(0)*ps0;
      // then, pi(k) = (pi_i(k) + pi_i(k+1))/2. Hence, accumulate midpoints to interfaces,
      // then do a interface->midpoints average.
      // For omega, we have omega_i(k+1) = omega_i(k)+div_vdp(k), with omega_i(0)=0;
      // then, omega(k) = v*grad(pi) - average(omega_i). You can see how we could
      // avoid computing omega_i, and set the accumulation step to be
      // omega(k+1) = omega(k)+(div_vdp(k)+div_vdp(k-1))/2, with omega(0)=div_vdp(0)/2.
      // However, avoiding computing omega_i would lose BFB agreement
      // with the original F90 implementation. So for now, keep
      // the two step calculation for omega (interface, then midpoints)
      // TODO; skip calculation of omega_i
      // Note: pi_i and omega_i are not needed after computing pi and omega,
      //       so simply grab unused buffers
      auto dp      = Homme::subview(m_state.m_dp3d,kv.ie,m_data.n0,igp,jgp);
      auto div_vdp = Homme::subview(m_buffers.div_vdp,kv.team_idx,igp,jgp);
      auto pi      = Homme::subview(m_buffers.pi,kv.team_idx,igp,jgp);
      auto omega_i = Homme::subview(m_buffers.grad_phinh_i,kv.team_idx,0,igp,jgp);
      auto pi_i    = Homme::subview(m_buffers.grad_phinh_i,kv.team_idx,1,igp,jgp);

      Kokkos::single(Kokkos::PerThread(kv.team),[&]() {
        pi_i(0)[0] = m_hvcoord.ps0*m_hvcoord.hybrid_ai0;
      });
      kv.team_barrier(); // necessary to avoid race in column_scan_mid_to_int

      ColumnOps::column_scan_mid_to_int<true>(kv,dp,pi_i);

      ColumnOps::compute_midpoint_values(kv,pi_i,pi);

      Kokkos::single(Kokkos::PerThread(kv.team),[&]() {
        omega_i(0)[0] = 0.0;
      });
      kv.team_barrier(); // necessary to avoid race in column_scan_mid_to_int

      ColumnOps::column_scan_mid_to_int<true>(kv,div_vdp,omega_i);
      // Average omega_i to midpoints, and change sign, since later
      //   omega=v*grad(pi)-average(omega_i)
      auto omega = Homme::subview(m_buffers.omega_p,kv.team_idx,igp,jgp);
      ColumnOps::compute_midpoint_values<CombineMode::Scale>(kv,omega_i,omega,-1.0);
    });
    kv.team_barrier();

    // Compute grad(pi)
    m_sphere_ops.gradient_sphere(kv,Homme::subview(m_buffers.pi,kv.team_idx),
                                    Homme::subview(m_buffers.grad_tmp,kv.team_idx));
    kv.team_barrier();

    // Update omega with v*grad(p)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto omega = Homme::subview(m_buffers.omega_p,kv.team_idx,igp,jgp);
      auto v = Homme::subview(m_state.m_v,kv.ie,m_data.n0);
      auto grad_pi = Homme::subview(m_buffers.grad_tmp,kv.team_idx);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        omega(ilev) += (v(0,igp,jgp,ilev)*grad_pi(0,igp,jgp,ilev) +
                        v(1,igp,jgp,ilev)*grad_pi(1,igp,jgp,ilev));
      });

      // Use EOS to compute pnh/exner or exner/phi (depending on whether it's hydro mode).
      if (m_theta_hydrostatic_mode) {
        // Recall that, in this case, pnh aliases pi, and pi was already computed
        m_eos.compute_exner(kv,Homme::subview(m_buffers.pnh,kv.team_idx,igp,jgp),
                               Homme::subview(m_buffers.exner,kv.team_idx,igp,jgp));

        m_eos.compute_phi_i(kv, m_geometry.m_phis(kv.ie,igp,jgp),
                            Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.n0,igp,jgp),
                            Homme::subview(m_buffers.exner,kv.team_idx,igp,jgp),
                            Homme::subview(m_buffers.pnh,kv.team_idx,igp,jgp),
                            Homme::subview(m_state.m_phinh_i,kv.ie,m_data.n0,igp,jgp));
      } else {
        const bool ok1 =
        m_eos.compute_pnh_and_exner(kv,
                                    Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.n0,igp,jgp),
                                    Homme::subview(m_state.m_phinh_i,kv.ie,m_data.n0,igp,jgp),
                                    Homme::subview(m_buffers.pnh,kv.team_idx,igp,jgp),
                                    Homme::subview(m_buffers.exner,kv.team_idx,igp,jgp));
        if ( ! ok1) ok = false;
      }

      // Compute phi at midpoints
      if (m_rsplit==0) {
        ColumnOps::compute_midpoint_values(kv,Homme::subview(m_state.m_phinh_i,kv.ie,m_data.n0,igp,jgp),
                                              Homme::subview(m_buffers.phi,kv.team_idx,igp,jgp));
      }
    });
    kv.team_barrier();
    return ok;
  }

  KOKKOS_INLINE_FUNCTION
  void compute_interface_quantities(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto dp   = Homme::subview(m_state.m_dp3d,kv.ie,m_data.n0,igp,jgp);
      auto dp_i = Homme::subview(m_buffers.dp_i,kv.team_idx,igp,jgp);

      // Compute interface dp
      ColumnOps::compute_interface_values(kv.team,dp,dp_i);

      if (!m_theta_hydrostatic_mode) {
        auto u    = Homme::subview(m_state.m_v,kv.ie,m_data.n0,0,igp,jgp);
        auto v    = Homme::subview(m_state.m_v,kv.ie,m_data.n0,1,igp,jgp);

        // Compute interface horiz velocity
        auto u_i  = Homme::subview(m_buffers.v_i,kv.team_idx,0,igp,jgp);
        auto v_i  = Homme::subview(m_buffers.v_i,kv.team_idx,1,igp,jgp);
        ColumnOps::compute_interface_values(kv.team,dp,dp_i,u,u_i);
        ColumnOps::compute_interface_values(kv.team,dp,dp_i,v,v_i);

        // grad_phinh_i is yet to be computed, so the buffer is available
        auto dpnh_dp_i = Homme::subview(m_buffers.dpnh_dp_i,kv.team_idx,igp,jgp);

        m_eos.compute_dpnh_dp_i(kv,Homme::subview(m_buffers.pnh,kv.team_idx,igp,jgp),
                                   dp_i,
                                   dpnh_dp_i);
      }

      if (m_rsplit==0) {
        // Shorter names, to keep a call to ColumnOps a bit shorter
        using CM = CombineMode;

        // Compute interface vtheta_i, with an energy preserving scheme
        auto vtheta_i = Homme::subview(m_buffers.vtheta_i,kv.team_idx,igp,jgp);
        auto dexner_i = Homme::subview(m_buffers.grad_phinh_i,kv.team_idx,0,igp,jgp);
        auto exner = Homme::subview(m_buffers.exner,kv.team_idx,igp,jgp);
        auto phi = Homme::subview(m_buffers.phi,kv.team_idx,igp,jgp);

        // vtheta_i(k) = -dpnh_dp_i(k)*(dphi(k) / dexner(k)) / Cp
        // with dX = X(k+1)-X(k)
        ColumnOps::compute_interface_delta(kv,phi,vtheta_i);

        // Since bcVal is the last input, if you need bcVal != 0.0,
        // you also need to pass alpha and beta as well (1.0 and 0.0 resp.).
        // Here, bcVal is 1.0 (to avoid nan/inf with division by 0)
        ColumnOps::compute_interface_delta<CM::Divide>(kv,exner,vtheta_i,1.0,0.0,1.0);

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                             [&](const int ilev) {
          vtheta_i(ilev) /= -PhysicalConstants::cp;
          if (!m_theta_hydrostatic_mode) {
            vtheta_i(ilev) *= m_buffers.dpnh_dp_i(kv.team_idx,igp,jgp,ilev);
          }
        });
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_vertical_advection(KernelVariables &kv) const {
    // Compute vertical advection terms:
    //  - eta_dot_dpdn
    //  - v_vadv
    //  - theta_vadv
    //  - phi_vadv_i
    //  - w_vadv_i
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      compute_eta_dot_dpn (kv,igp,jgp);
      compute_v_vadv      (kv,igp,jgp);
      compute_vtheta_vadv (kv,igp,jgp);
      if (!m_theta_hydrostatic_mode) {
        compute_w_vadv      (kv,igp,jgp);
        compute_phi_vadv    (kv,igp,jgp);
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_eta_dot_dpn (KernelVariables& kv, const int& igp, const int& jgp) const {

    auto div_vdp = ekat::scalarize(Homme::subview(m_buffers.div_vdp,kv.team_idx,igp,jgp));
    auto eta_dot_dpdn = ekat::scalarize(Homme::subview(m_buffers.eta_dot_dpdn,kv.team_idx,igp,jgp));

    // Integrate -vdp
    Dispatch<ExecSpace>::parallel_scan(kv.team,NUM_PHYSICAL_LEV,
                          [&](const int ilev, ST& accumulator, const bool last) {
      accumulator += div_vdp(ilev);
      if (last) {
        eta_dot_dpdn(ilev+1) = -accumulator;
      }
    });

    // Get the last entry, which is the sum over the whole column
    auto eta_dot_dpdn_last = -eta_dot_dpdn(NUM_INTERFACE_LEV-1);

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,1,NUM_PHYSICAL_LEV),
                         [&](const int ilev) {
      eta_dot_dpdn(ilev) += m_hvcoord.hybrid_bi(ilev)*eta_dot_dpdn_last;
    });

    // Set the boundary conditions for eta_dot_dpdn
    Kokkos::single(Kokkos::PerThread(kv.team),[&](){
      eta_dot_dpdn(0) = eta_dot_dpdn(NUM_INTERFACE_LEV-1) = 0;
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_v_vadv (KernelVariables& kv, const int& igp, const int& jgp) const {
    // Note: save v_vadv temp by stuffing directly in vtens

    auto u  = ekat::scalarize(Homme::subview(m_state.m_v,kv.ie,m_data.n0,0,igp,jgp));
    auto v  = ekat::scalarize(Homme::subview(m_state.m_v,kv.ie,m_data.n0,1,igp,jgp));
    auto dp = ekat::scalarize(Homme::subview(m_state.m_dp3d,kv.ie,m_data.n0,igp,jgp));
    auto eta_dot_dpdn = ekat::scalarize(Homme::subview(m_buffers.eta_dot_dpdn,kv.team_idx,igp,jgp));
    auto u_vadv = ekat::scalarize(Homme::subview(m_buffers.v_tens,kv.team_idx,0,igp,jgp));
    auto v_vadv = ekat::scalarize(Homme::subview(m_buffers.v_tens,kv.team_idx,1,igp,jgp));

    // TODO: vectorize this code.
    const ST facp_1 = 0.5*eta_dot_dpdn(1)/dp(0);
    u_vadv(0) = facp_1*(u(1)-u(0));
    v_vadv(0) = facp_1*(v(1)-v(0));

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,1,NUM_PHYSICAL_LEV-1),
                         [&](const int k) {
      const ST facp = 0.5*eta_dot_dpdn(k+1)/dp(k);
      const ST facm = 0.5*eta_dot_dpdn(k)/dp(k);
      u_vadv(k) = facp*(u(k+1)-u(k)) + facm*(u(k)-u(k-1));
      v_vadv(k) = facp*(v(k+1)-v(k)) + facm*(v(k)-v(k-1));
    });

    constexpr int last = NUM_PHYSICAL_LEV-1;
    const ST facm_N = 0.5*eta_dot_dpdn(last)/dp(last);
    u_vadv(last) = facm_N*(u(last)-u(last-1));
    v_vadv(last) = facm_N*(v(last)-v(last-1));
  }

  KOKKOS_INLINE_FUNCTION
  void compute_vtheta_vadv (KernelVariables& kv, const int& igp, const int& jgp) const {
    auto eta_dot_dpdn = Homme::subview(m_buffers.eta_dot_dpdn,kv.team_idx,igp,jgp);
    auto vtheta_i = Homme::subview(m_buffers.vtheta_i,kv.team_idx,igp,jgp);
    // No point in going till NUM_LEV_P, since vtheta_i=0 at the bottom
    // Also, this is the last time we need vtheta_i, so we can overwrite it.

    auto provider = [&](const int ilev)->PT{
      return eta_dot_dpdn(ilev)*vtheta_i(ilev);
    };

    // Compute theta_vadv, store directly in theta_tens
    auto theta_vadv = Homme::subview(m_buffers.theta_tens,kv.team_idx,igp,jgp);
    ColumnOps::compute_midpoint_delta(kv,provider,theta_vadv);
  }

  KOKKOS_INLINE_FUNCTION
  void compute_w_vadv (KernelVariables& kv, const int& igp, const int& jgp) const {
    // w_vadv = average(temp)/dp3d_i.
    // temp = average(eta_dot)*delta(w_i)
    // Note: store directly in w_tens
    auto temp = Homme::subview(m_buffers.temp,kv.team_idx,igp,jgp);
    auto w_i = Homme::subview(m_state.m_w_i,kv.ie,m_data.n0,igp,jgp);
    auto dp_i = Homme::subview(m_buffers.dp_i,kv.team_idx,igp,jgp);
    auto eta_dot_dpdn = Homme::subview(m_buffers.eta_dot_dpdn,kv.team_idx,igp,jgp);
    auto w_vadv = Homme::subview(m_buffers.w_tens,kv.team_idx,igp,jgp);

    ColumnOps::compute_midpoint_delta(kv,w_i,temp);
    ColumnOps::compute_midpoint_values<CombineMode::Multiply>(kv,eta_dot_dpdn,temp);

    ColumnOps::compute_interface_values(kv,temp,w_vadv);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV_P),
                         [&](const int ilev) {
      w_vadv(ilev) /= dp_i(ilev);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_phi_vadv (KernelVariables& kv, const int& igp, const int& jgp) const {
    // phi_vadv = eta_dot*delta(phi)/dp3d_i.
    // Note: store directly into phi_tens
    auto phi = Homme::subview(m_buffers.phi,kv.team_idx,igp,jgp);
    auto phi_vadv = Homme::subview(m_buffers.phi_tens,kv.team_idx,igp,jgp);
    auto eta_dot_dpdn = Homme::subview(m_buffers.eta_dot_dpdn,kv.team_idx,igp,jgp);
    auto dp_i = Homme::subview(m_buffers.dp_i,kv.team_idx,igp,jgp);

    ColumnOps::compute_interface_delta(kv,phi,phi_vadv);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                         [&](const int ilev) {
      phi_vadv(ilev) *= eta_dot_dpdn(ilev);
      phi_vadv(ilev) /= dp_i(ilev);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_accumulated_quantities(KernelVariables &kv) const {
    // Compute omega = v*grad(p) + average(omega_i)
    // Accmuulate: dereived.omega_p += eta_ave_w*omega
    // Accmuulate: dereived.eta_dot_dpdn += eta_ave_w*eta_dot_dpdn
    // Accumulate: vn0 += eta_ave_w*v*dp
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // If rsplit>0, m_buffers.eta_dot_dpdn is identically 0, so skip this step
      if (m_rsplit==0) {
        // Accumulate eta_dot_dpdn
        // Note: m_buffers.eta_dot_dpdn is 0 at top/bottom, so we can just accumulate NUM_LEV packs
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                             [&](const int ilev) {
          m_derived.m_eta_dot_dpdn(kv.ie,igp,jgp,ilev) +=
                m_data.eta_ave_w*m_buffers.eta_dot_dpdn(kv.team_idx,igp,jgp,ilev);
        });
      }

      // Compute omega = v*grad(p)+average(omega_i) and accumulate
      // Accumulate v*dp in vn0
      // Note: so far omega already contains average(omega_i), since we didn't even
      //       bother storing omega_i, and computed directly the average
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        m_derived.m_omega_p(kv.ie,igp,jgp,ilev) +=
              m_data.eta_ave_w*m_buffers.omega_p(kv.team_idx,igp,jgp,ilev);

        m_derived.m_vn0(kv.ie,0,igp,jgp,ilev) += m_data.eta_ave_w*m_buffers.vdp(kv.team_idx,0,igp,jgp,ilev);
        m_derived.m_vn0(kv.ie,1,igp,jgp,ilev) += m_data.eta_ave_w*m_buffers.vdp(kv.team_idx,1,igp,jgp,ilev);
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_w_and_phi_tens (KernelVariables& kv) const {
    using namespace PhysicalConstants;
    // Compute grad(phinh_i)
    // Compute v*grad(w_i)
    // Compute w_tens = scale1*(-w_vadv_i - v*grad(w_i)) - scale2*g*(1-dpnh_dp_i)
    // Compute phi_tens = scale1*(-phi_vadv_i - v*grad(phinh_i)) + scale2*g*w_i
    auto grad_w_i = Homme::subview(m_buffers.grad_w_i,kv.team_idx);
    auto grad_phinh_i = Homme::subview(m_buffers.grad_phinh_i,kv.team_idx);
    m_sphere_ops.gradient_sphere(kv,Homme::subview(m_state.m_phinh_i,kv.ie,m_data.n0),
                                    grad_phinh_i);
    kv.team_barrier();
    m_sphere_ops.gradient_sphere(kv,Homme::subview(m_state.m_w_i,kv.ie,m_data.n0),
                                    grad_w_i);
    kv.team_barrier();

    auto v_i = Homme::subview(m_buffers.v_i,kv.team_idx);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto w_tens = Homme::subview(m_buffers.w_tens,kv.team_idx,igp,jgp);
      auto phi_tens = Homme::subview(m_buffers.phi_tens,kv.team_idx,igp,jgp);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV_P),
                           [&](const int ilev) {

        // Note: if rsplit=0, phi_tens/w_tens already contains phi_vadv/w_vadv,
        //       otherwise, just garbage from previous team

        // Compute w_tens
        PT v_grad = v_i(0,igp,jgp,ilev)*grad_w_i(0,igp,jgp,ilev)
                  + v_i(1,igp,jgp,ilev)*grad_w_i(1,igp,jgp,ilev);
        if (m_rsplit==0) {
          w_tens(ilev) += v_grad;
        } else {
          w_tens(ilev)  = v_grad;
        }
        w_tens(ilev) *= -m_data.scale1;
        w_tens(ilev) += (m_buffers.dpnh_dp_i(kv.team_idx,igp,jgp,ilev)-1) *
                        (ilev==(NUM_LEV_P-1) ? m_scale2g_last_int_pack : PT(m_data.scale2*g));

        // Compute phi_tens.
        v_grad = v_i(0,igp,jgp,ilev)*grad_phinh_i(0,igp,jgp,ilev)
               + v_i(1,igp,jgp,ilev)*grad_phinh_i(1,igp,jgp,ilev);
        if (m_rsplit==0) {
          phi_tens(ilev) += v_grad;
        } else {
          phi_tens(ilev) =  v_grad;
        }
        phi_tens(ilev) *= -m_data.scale1;
        phi_tens(ilev) += m_state.m_w_i(kv.ie,m_data.n0,igp,jgp,ilev) *
                          (ilev==NUM_LEV_P ? m_scale2g_last_int_pack : PT(m_data.scale2*g));

        if (m_data.scale1!=m_data.scale2) {
           // add imex phi_h splitting
           // use approximate phi_h = hybi*phis
           // could also use true hydrostatic pressure, but this requires extra DSS in dirk()
           phi_tens(ilev) +=  (m_data.scale1-m_data.scale2) *
                (v_i(0,igp,jgp,ilev)*m_geometry.m_gradphis(kv.ie,0,igp,jgp) +
                 v_i(1,igp,jgp,ilev)*m_geometry.m_gradphis(kv.ie,1,igp,jgp) ) * m_hvcoord.hybrid_bi_packed(ilev);
        }
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_w_and_phi_np1(KernelVariables &kv) const {
    // Update w_i(np1) = spheremp*(scale3*w_i(nm1) + dt*w_tens)
    // Update phi_i(np1) = spheremp*(scale3*phi_i(nm1) + dt*phi_tens)

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto spheremp = m_geometry.m_spheremp(kv.ie,igp,jgp);

      auto w_tens = Homme::subview(m_buffers.w_tens,kv.team_idx,igp,jgp);
      auto phi_tens = Homme::subview(m_buffers.phi_tens,kv.team_idx,igp,jgp);

      auto w_np1 = Homme::subview(m_state.m_w_i,kv.ie,m_data.np1,igp,jgp);
      auto phi_np1 = Homme::subview(m_state.m_phinh_i,kv.ie,m_data.np1,igp,jgp);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {

        // Add w_tens to w_nm1 and multiply by spheremp
        w_tens(ilev) *= m_data.dt*spheremp;
        w_np1(ilev) = m_state.m_w_i(kv.ie,m_data.nm1,igp,jgp,ilev);
        w_np1(ilev) *= m_data.scale3*spheremp;
        w_np1(ilev) += w_tens(ilev);

        // Add phi_tens to phi_nm1 and multiply by spheremp
        // temp *= m_data.dt*spheremp;
        phi_tens(ilev) *= m_data.dt*spheremp;
        phi_np1(ilev) = m_state.m_phinh_i(kv.ie,m_data.nm1,igp,jgp,ilev);
        phi_np1(ilev) *= m_data.scale3*spheremp;
        phi_np1(ilev) += phi_tens(ilev);
      });

      // Last interface only for w, not phi (since phi=phis there)
      Kokkos::single(Kokkos::PerThread(kv.team),[&](){
        using Info = ColInfo<NUM_INTERFACE_LEV>;
        constexpr int ilev = Info::LastPack;
        constexpr int ivec = Info::LastPackEnd;

        // Note: we only have 1 physical entry in the pack,
        // so we may as well operate on the Real level
        if (NUM_LEV==NUM_LEV_P) {
          // We processed last interface too, so fix phi.
          phi_np1(ilev)[ivec] = m_geometry.m_phis(kv.ie,igp,jgp);
        } else {
          // We didn't do anything on last interface, so update w
          w_tens(ilev)[ivec] *= m_data.dt*spheremp;
          w_np1(ilev)[ivec]  = m_state.m_w_i(kv.ie,m_data.nm1,igp,jgp,ilev)[ivec];
          w_np1(ilev)[ivec] *= m_data.scale3*spheremp;
          w_np1(ilev)[ivec] += w_tens(ilev)[ivec];
        }
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_dp_and_theta_tens (KernelVariables &kv) const {
    // Compute dp_tens=scale1*(div(vdp) + delta(eta_dot_dpdn))
    // Compute theta_tens=scale1*(-theta_vadv-div(v*vtheta_dp)
    // Note: div(v*vtheta_dp) can be computed in conservative or
    //       non concervative form (div(ab) vs a*div(b)+grad(a)*b)

    // Use a couple of lambdas to compute input of diff operators on the fly
    auto v         = Homme::subview(m_state.m_v,kv.ie,m_data.n0);
    auto vtheta_dp = Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.n0);
    auto dp        = Homme::subview(m_state.m_dp3d,kv.ie,m_data.n0);

    auto vtheta = [&](const int igp,const int jgp,const int ilev)->PT {
      return vtheta_dp(igp,jgp,ilev) / dp(igp,jgp,ilev);
    };

    auto v_vtheta_dp = [&](const int icomp, const int igp, const int jgp, const int ilev)->PT {
      return v(icomp,igp,jgp,ilev) * vtheta_dp(igp,jgp,ilev);
    };

    if (m_theta_advection_form==AdvectionForm::Conservative) {
      if (m_rsplit==0) {
        using CM = CombineMode;
        // If you want a CombineMode different than Replace, unfortunately you have to specify
        // all the template args, since the CombineMode is the last one...
        m_sphere_ops.template divergence_sphere_cm<CM::Add>(kv,v_vtheta_dp,
                                          Homme::subview(m_buffers.theta_tens,kv.team_idx));
      } else {
        m_sphere_ops.divergence_sphere(kv,v_vtheta_dp,
                                          Homme::subview(m_buffers.theta_tens,kv.team_idx));
      }
    } else {
      m_sphere_ops.gradient_sphere(kv,vtheta,
                                      Homme::subview(m_buffers.grad_tmp,kv.team_idx));
    }
    kv.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto dp_tens = Homme::subview(m_buffers.dp_tens,kv.team_idx,igp,jgp);
      auto theta_tens = Homme::subview(m_buffers.theta_tens,kv.team_idx,igp,jgp);

      auto div_vdp = Homme::subview(m_buffers.div_vdp,kv.team_idx,igp,jgp);

      if (m_rsplit==0) {
        auto eta_dot_dpdn = Homme::subview(m_buffers.eta_dot_dpdn,kv.team_idx,igp,jgp);
        ColumnOps::compute_midpoint_delta(kv,eta_dot_dpdn,dp_tens);
      }

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        // Compute dp_tens
        if (m_rsplit>0) {
          dp_tens(ilev)  = div_vdp(ilev);
        } else {
          dp_tens(ilev) += div_vdp(ilev);
        }

        // Compute theta_tens
        // NOTE: if the condition is false, then theta_tens already contains div(v*theta*dp) already
        if (m_theta_advection_form==AdvectionForm::NonConservative) {
          // We need a temp, since, if rsplit=0, theta_tens is already storing theta_vadv

          PT temp = div_vdp(ilev)*vtheta(igp,jgp,ilev);
          temp += m_buffers.grad_tmp(kv.team_idx,0,igp,jgp,ilev)*m_buffers.vdp(kv.team_idx,0,igp,jgp,ilev);
          temp += m_buffers.grad_tmp(kv.team_idx,1,igp,jgp,ilev)*m_buffers.vdp(kv.team_idx,1,igp,jgp,ilev);
          if (m_rsplit>0) {
            theta_tens(ilev) = temp;
          } else {
            theta_tens(ilev) += temp;
          }
        }
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_dp3d_and_theta_np1(KernelVariables &kv) const {
    // Update dp3d(np1) = spheremp*(scale3*dp3d(nm1) + dt*dp_tens)
    // Update vtheta_dp(np1) = spheremp*(scale3*vtheta_dp(nm1) + dt*theta_tens)

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      const auto& spheremp = m_geometry.m_spheremp(kv.ie,igp,jgp);

      auto dp_tens = Homme::subview(m_buffers.dp_tens,kv.team_idx,igp,jgp);
      auto theta_tens = Homme::subview(m_buffers.theta_tens,kv.team_idx,igp,jgp);
      auto dp_np1 = Homme::subview(m_state.m_dp3d,kv.ie,m_data.np1,igp,jgp);
      auto vtheta_np1 = Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.np1,igp,jgp);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {

        // Add dp_tens to dp_nm1 and multiply by spheremp
        dp_tens(ilev) *= m_data.scale1*m_data.dt*spheremp;
        dp_np1(ilev)   = m_data.scale3*spheremp*m_state.m_dp3d(kv.ie,m_data.nm1,igp,jgp,ilev);
        dp_np1(ilev)  -= dp_tens(ilev);

        // Add theta_tens to vtheta_nm1 and multiply by spheremp
        theta_tens(ilev) *= -m_data.scale1*m_data.dt*spheremp;
        vtheta_np1(ilev)  = m_state.m_vtheta_dp(kv.ie,m_data.nm1,igp,jgp,ilev);
        vtheta_np1(ilev) *= m_data.scale3*spheremp;
        vtheta_np1(ilev) += theta_tens(ilev);
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_v_tens (KernelVariables &kv) const {
    // Not necessarily in this order,
    //  - Compute vort=vorticity(v)
    //  - Compute wvor = grad[average(w_i^2/2)] - average[w_i*grad(w_i)]
    //  - Compute gradKE = grad(v*v/2)
    //  - Compute gradExner = grad(exner)
    //  - Compute mgrad = average[dpnh_dp_i*gradphinh_i]
    //              + cp*T0*(grad(log(exner))-grad(exner)/exner) (pgrad_correction, if applicable)
    //  - Compute v_tens =
    //           scale1*(-v_vadv + v2*(fcor+vort)-gradKE -mgrad  -cp*vtheta*gradExner - (mgrad + wvor))
    //           scale1*(-v_vadv - v1*(fcor+vort)-gradKE -mgrad  -cp*vtheta*gradExner - (mgrad + wvor))

    auto vort  = Homme::subview(m_buffers.vort,kv.team_idx);
    auto wvor = Homme::subview(m_buffers.vdp,kv.team_idx);
    auto grad_exner = Homme::subview(m_buffers.grad_exner,kv.team_idx);
    auto mgrad = Homme::subview(m_buffers.mgrad,kv.team_idx);
    auto grad_tmp = Homme::subview(m_buffers.grad_tmp,kv.team_idx);

    // Compute vorticity(v)
    m_sphere_ops.vorticity_sphere(kv, Homme::subview(m_state.m_v,kv.ie,m_data.n0),
                                      vort);

    if (m_theta_hydrostatic_mode) {
      // In nh mode, gradphinh has already been computed, but in hydro mode
      // we skip the whole compute_w_and_phi_tens call
      m_sphere_ops.gradient_sphere(kv,Homme::subview(m_state.m_phinh_i,kv.ie,m_data.n0),
                                      Homme::subview(m_buffers.grad_phinh_i,kv.team_idx));
      kv.team_barrier();
    } else {
      // Compute average(w^2/2)
      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;
        auto w_i = Homme::subview(m_state.m_w_i,kv.ie,m_data.n0,igp,jgp);

        // Use lambda as a provider for w_i*w_i
        const auto w_sq = [&w_i] (const int ilev) {
          return w_i(ilev)*w_i(ilev);
        };

        // Rescale by 2 (we are averaging kinetic energy w_i*w_i/2, but the provider has no '/2')
        ColumnOps::compute_midpoint_values<CombineMode::Scale>(kv, w_sq, Homme::subview(m_buffers.temp,kv.team_idx,igp,jgp),0.5);
      });
      kv.team_barrier();

      // Compute grad(average(w^2/2)). Store in wvor.
      m_sphere_ops.gradient_sphere(kv, Homme::subview(m_buffers.temp,kv.team_idx),
                                       wvor);

      // Compute grad(w)
      m_sphere_ops.gradient_sphere(kv, Homme::subview(m_state.m_w_i,kv.ie,m_data.n0),
                                       Homme::subview(m_buffers.v_i,kv.team_idx));
      kv.team_barrier();
    }

    // Compute grad(exner)
    // Note: exner = (pi/p0)^k, therefore grad(exner) = k*(exner/pi)*grad(pi).
    //       So you *could* avoid computing this grad, at the price of some arithmetic op.
    m_sphere_ops.gradient_sphere(kv, Homme::subview(m_buffers.exner,kv.team_idx),
                                     grad_exner);

    // If pgrad_correction=1, we need the gradient sphere
    // for log(exner) below. Store into m_buffers.grad_tmp.
    if (m_pgrad_correction) {
      const auto exner = Homme::subview(m_buffers.exner,kv.team_idx);
      const auto log_exner = [&exner](const int igp, const int jgp, const int ilev)->PT {
        return log(exner(igp, jgp, ilev));
      };
      m_sphere_ops.gradient_sphere(kv, log_exner,
                                       grad_tmp);
    }
    kv.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto wvor_x = Homme::subview(wvor,0,igp,jgp);
      auto wvor_y = Homme::subview(wvor,1,igp,jgp);

      if (!m_theta_hydrostatic_mode) {
        // Compute wvor = grad(average(w^2/2)) - average(w*grad(w))
        // Note: vtens is already storing grad(avg(w^2/2))
        auto gradw_x = Homme::subview(m_buffers.v_i,kv.team_idx,0,igp,jgp);
        auto gradw_y = Homme::subview(m_buffers.v_i,kv.team_idx,1,igp,jgp);
        auto w_i = Homme::subview(m_state.m_w_i,kv.ie,m_data.n0,igp,jgp);

        const auto w_gradw_x = [&gradw_x,&w_i] (const int ilev) {
          return gradw_x(ilev)*w_i(ilev);
        };
        const auto w_gradw_y = [&gradw_y,&w_i] (const int ilev) {
          return gradw_y(ilev)*w_i(ilev);
        };

        ColumnOps::compute_midpoint_values<CombineMode::ScaleAdd>(kv,
                          w_gradw_x, wvor_x, -1.0);
        ColumnOps::compute_midpoint_values<CombineMode::ScaleAdd>(kv,
                          w_gradw_y, wvor_y, -1.0);
      } else {
        // wvor is not used if theta_hydrostatic_mode=1. Set to zero
        // here to avoid adding in uninitialized values into v_tens.
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                             [&](const int ilev) {
          wvor_x(ilev) = 0;
          wvor_y(ilev) = 0;
        });
      }

      auto mgrad_x = Homme::subview(mgrad,0,igp,jgp);
      auto mgrad_y = Homme::subview(mgrad,1,igp,jgp);

      // Compute mgrad = average(dpnh_dp_i*grad(phinh_i))
      const auto phinh_i_x = Homme::subview(m_buffers.grad_phinh_i,kv.team_idx,0,igp,jgp);
      const auto phinh_i_y = Homme::subview(m_buffers.grad_phinh_i,kv.team_idx,1,igp,jgp);
      if (m_theta_hydrostatic_mode) {
        ColumnOps::compute_midpoint_values(kv,phinh_i_x,mgrad_x);
        ColumnOps::compute_midpoint_values(kv,phinh_i_y,mgrad_y);
      } else {
        const auto dpnh_dp_i = Homme::subview(m_buffers.dpnh_dp_i,kv.team_idx,igp,jgp);
        const auto prod_x = [&phinh_i_x,&dpnh_dp_i](const int ilev)->PT {
          return phinh_i_x(ilev)*dpnh_dp_i(ilev);
        };
        const auto prod_y = [&phinh_i_y,&dpnh_dp_i](const int ilev)->PT {
          return phinh_i_y(ilev)*dpnh_dp_i(ilev);
        };

        ColumnOps::compute_midpoint_values(kv,prod_x,mgrad_x);
        ColumnOps::compute_midpoint_values(kv,prod_y,mgrad_y);
      }

      // Apply pgrad_correction: mgrad += cp*T0*(grad(log(exner))-grad(exner)/exner) (if applicable)
      if (m_pgrad_correction) {
        using namespace PhysicalConstants;
        constexpr Real T0 = Tref - Tref_lapse_rate*Tref*cp/g;

        const auto grad_tmp_i_x   = Homme::subview(grad_tmp,0,igp,jgp);
        const auto grad_tmp_i_y   = Homme::subview(grad_tmp,1,igp,jgp);
        const auto grad_exner_i_x = Homme::subview(grad_exner,0,igp,jgp);
        const auto grad_exner_i_y = Homme::subview(grad_exner,1,igp,jgp);
        const auto exner_i        = Homme::subview(m_buffers.exner,kv.team_idx,igp,jgp);

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                             [&](const int ilev) {
          mgrad_x(ilev) += cp*T0*(grad_tmp_i_x(ilev) - grad_exner_i_x(ilev)/exner_i(ilev));
          mgrad_y(ilev) += cp*T0*(grad_tmp_i_y(ilev) - grad_exner_i_y(ilev)/exner_i(ilev));
        });
      }

      // Compute KE. Also, add fcor to vort
      auto u  = Homme::subview(m_state.m_v,kv.ie,m_data.n0,0,igp,jgp);
      auto v  = Homme::subview(m_state.m_v,kv.ie,m_data.n0,1,igp,jgp);
      auto KE = Homme::subview(m_buffers.temp,kv.team_idx,igp,jgp);
      const auto& fcor = m_geometry.m_fcor(kv.ie,igp,jgp);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        KE(ilev)  = u(ilev)*u(ilev);
        KE(ilev) += v(ilev)*v(ilev);
        KE(ilev) /= 2.0;

        vort(igp,jgp,ilev) += fcor;
      });
    });
    kv.team_barrier();

    // Compute grad(KE), and put it directly in v_tens
    if (m_rsplit==0) {
      // v_tens already contains v_vadv. Need to sum into it.
      m_sphere_ops.gradient_sphere_update(kv, Homme::subview(m_buffers.temp,kv.team_idx),
                                              Homme::subview(m_buffers.v_tens,kv.team_idx));
    } else {
      // v_tens contains garbage from previos team. Overwrite it.
      m_sphere_ops.gradient_sphere(kv, Homme::subview(m_buffers.temp,kv.team_idx),
                                       Homme::subview(m_buffers.v_tens,kv.team_idx));
    }

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // Assemble vtens (both components)
      // Note: right now, vtens already contains gradKE+v_vadv
      //       (or just gradKE if rsplit>0)
      auto u_tens = Homme::subview(m_buffers.v_tens,kv.team_idx,0,igp,jgp);
      auto v_tens = Homme::subview(m_buffers.v_tens,kv.team_idx,1,igp,jgp);
      auto vort = Homme::subview(m_buffers.vort,kv.team_idx,igp,jgp);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        // grad(exner)*vtheta*cp
        PT cp_vtheta = PhysicalConstants::cp *
                           (m_state.m_vtheta_dp(kv.ie,m_data.n0,igp,jgp,ilev) /
                            m_state.m_dp3d(kv.ie,m_data.n0,igp,jgp,ilev));

        u_tens(ilev) += cp_vtheta*grad_exner(0,igp,jgp,ilev);
        v_tens(ilev) += cp_vtheta*grad_exner(1,igp,jgp,ilev);

        u_tens(ilev) += (mgrad(0,igp,jgp,ilev) + wvor(0,igp,jgp,ilev));
        v_tens(ilev) += (mgrad(1,igp,jgp,ilev) + wvor(1,igp,jgp,ilev));

        // Add +/- v_j*(vort+fcor), where v_j=v for u_tens and v_j=u for v_tens
        u_tens(ilev) -= m_state.m_v(kv.ie,m_data.n0,1,igp,jgp,ilev)*vort(ilev);
        v_tens(ilev) += m_state.m_v(kv.ie,m_data.n0,0,igp,jgp,ilev)*vort(ilev);
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_v_np1(KernelVariables &kv) const {
    // Update v(np1) = spheremp*(scale3*v(nm1) + dt*v_tens)
    // NOTE: quite a few buffers no longer needed in this call of operator() are going to be reused here.

    auto v_tens = Homme::subview(m_buffers.v_tens,kv.team_idx);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      const auto& spheremp = m_geometry.m_spheremp(kv.ie,igp,jgp);

      // Add vtens to v_nm1 and multiply by spheremp
      auto u_np1 = Homme::subview(m_state.m_v,kv.ie,m_data.np1,0,igp,jgp);
      auto v_np1 = Homme::subview(m_state.m_v,kv.ie,m_data.np1,1,igp,jgp);
      auto u_tens = Homme::subview(m_buffers.v_tens,kv.team_idx,0,igp,jgp);
      auto v_tens = Homme::subview(m_buffers.v_tens,kv.team_idx,1,igp,jgp);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {

        // Add v_tens to v_nm1 and multiply by spheremp
        u_np1(ilev) = m_state.m_v(kv.ie,m_data.nm1,0,igp,jgp,ilev);
        v_np1(ilev) = m_state.m_v(kv.ie,m_data.nm1,1,igp,jgp,ilev);

        u_tens(ilev) *= -m_data.scale1*m_data.dt*spheremp;
        v_tens(ilev) *= -m_data.scale1*m_data.dt*spheremp;

        u_np1(ilev) *= m_data.scale3*spheremp;
        v_np1(ilev) *= m_data.scale3*spheremp;

        u_np1(ilev) += u_tens(ilev);
        v_np1(ilev) += v_tens(ilev);
      });
    });
  }

};

using CaarFunctorImpl = CaarFunctorImplST<ScalarValue>;

} // Namespace Homme

#endif // HOMMEXX_CAAR_FUNCTOR_IMPL_HPP
