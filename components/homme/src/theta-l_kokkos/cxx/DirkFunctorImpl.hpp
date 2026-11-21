/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_DIRK_FUNCTOR_IMPL_HPP
#define HOMMEXX_DIRK_FUNCTOR_IMPL_HPP

#include "Types.hpp"
#include "EquationOfState.hpp"
#include "FunctorsBuffersManager.hpp"
#include "Elements.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "PhysicalConstants.hpp"
#include "ElementOps.hpp"
#include "profiling.hpp"
#include "ErrorDefs.hpp"
#include "utilities/MathUtils.hpp"
#include "utilities/scream_tridiag.hpp"

#include <ekat_assert.hpp>

#include <cassert>
#include <type_traits>

namespace Homme {

template<typename ST, typename Constants = PhysicalConstantsProvider>
struct DirkFunctorImplST {
  using PT = PackType<ST>;

  static constexpr int  packn = VECTOR_SIZE;
  static constexpr int  scaln = NP*NP;
  static constexpr int  npack = (scaln + packn - 1)/packn;
  static constexpr int  max_num_lev_pack = NUM_LEV_P;
  static constexpr int  num_lev_aligned = max_num_lev_pack*packn;
  static constexpr int  num_phys_lev = NUM_PHYSICAL_LEV;
  static constexpr int  num_work = 12;
  static constexpr bool calc_initial_guess_in_newton_kernel = false;

  static constexpr int
#ifdef HOMMEXX_BFB_TESTING
    default_bfb_solver = 1;
#else
    default_bfb_solver = 0;
#endif

  static_assert(num_lev_aligned >= 3,
                "We use wrk(0:2,:) and so need num_lev_aligned >= 3");

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MT = typename TeamPolicy::member_type;

  using Work
    = Kokkos::View<PT*[num_work][num_lev_aligned][npack],
                   Kokkos::LayoutRight, ExecSpace>;
  using WorkSlot
    = Kokkos::View<PT           [num_lev_aligned][npack],
                   Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  using ConstWorkSlot
    = Kokkos::View<const PT     [num_lev_aligned][npack],
                   Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  using LinearSystem
    = Kokkos::View<PT*[4][num_phys_lev][npack],
                   Kokkos::LayoutRight, ExecSpace>;
  using LinearSystemSlot
    = Kokkos::View<PT    [num_phys_lev][npack],
                   Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  KOKKOS_INLINE_FUNCTION
  static WorkSlot get_work_slot (const Work& w, const int& wi, const int& si) {
    using Kokkos::subview;
    using Kokkos::ALL;
    const auto a = ALL();
    return subview(w, wi, si, a, a);
  }

  KOKKOS_INLINE_FUNCTION
  static LinearSystemSlot get_ls_slot (const LinearSystem& w, const int& wi,
                                       const int& si) {
    using Kokkos::subview;
    using Kokkos::ALL;
    const auto a = ALL();
    return subview(w, wi, si, a, a);
  }

  Work m_work;
  LinearSystem m_ls;
  TeamPolicy m_policy, m_ig_policy;
  TeamUtils<ExecSpace> m_tu, m_tu_ig;
  int nslot;

#ifdef HOMMEXX_ENABLE_FAD_TYPES
  // Whether this is the functor of the adjoint of the DIRK stages (see run_adjoint)
  static constexpr bool is_adj_fad = std::is_same_v<ST, DirkAdjFadType>;
#else
  static constexpr bool is_adj_fad = false;
#endif

  // Scratch space for run_adjoint: the Fad-typed state of one element, in Hxx format, for 2 time
  // levels (buffer 0: the np1 state; buffer 1: the n0 and nm1 states, one at a time).
  // Only allocated if is_adj_fad. Dims: (team slot, buffer, [component,] ip, jp, level).
  using AdjViewV = Kokkos::View<PT*[2][2][NP][NP][NUM_LEV  ], Kokkos::LayoutRight, ExecSpace>;
  using AdjViewM = Kokkos::View<PT*[2]   [NP][NP][NUM_LEV  ], Kokkos::LayoutRight, ExecSpace>;
  using AdjViewI = Kokkos::View<PT*[2]   [NP][NP][NUM_LEV_P], Kokkos::LayoutRight, ExecSpace>;
  AdjViewV m_adj_v;
  AdjViewM m_adj_vth, m_adj_dp;
  AdjViewI m_adj_w, m_adj_phi;
  int m_dx_tl = -1;  // time level seeded in init_J; used by run_JtV for identity block
  bool m_verbose = true; // Set to false to suppress some warnings in unit tests
  EquationOfState<Constants> m_eos;

  DirkFunctorImplST (const int nelem)
    : m_policy(1,1,1), m_ig_policy(1,1,1), m_tu(m_policy), m_tu_ig(m_ig_policy) // throwaway settings
  {
    init(nelem);
  }

  void init (const int nelem) {
    if (OnGpu<ExecSpace>::value) {
      ThreadPreferences tp;
      tp.max_threads_usable = NUM_PHYSICAL_LEV;
      tp.max_vectors_usable = NP*NP;
      tp.prefer_threads = false;
      tp.prefer_larger_team = true;
      const auto p = DefaultThreadsDistribution<ExecSpace>
        ::team_num_threads_vectors(nelem, tp);
      const auto
        nhwthr = p.first*p.second,
        nvec = std::min(NP*NP, nhwthr),
        nthr = nhwthr/nvec;
      m_policy = TeamPolicy(nelem, nthr, nvec);
    } else {
      ThreadPreferences tp;
      tp.max_threads_usable = NUM_PHYSICAL_LEV;
      tp.max_vectors_usable = 1;
      tp.prefer_threads = true;
      const auto p = DefaultThreadsDistribution<ExecSpace>
        ::team_num_threads_vectors(nelem, tp);
      m_policy = TeamPolicy(nelem, p.first, 1);
    }
    m_tu = TeamUtils<ExecSpace>(m_policy);
    nslot = std::min(nelem, m_tu.get_num_ws_slots());
    if constexpr (is_adj_fad) {
      m_adj_v   = AdjViewV  ("dirk_adj_v"  , nslot);
      m_adj_vth = AdjViewM  ("dirk_adj_vth", nslot);
      m_adj_dp  = AdjViewM  ("dirk_adj_dp" , nslot);
      m_adj_w   = AdjViewI  ("dirk_adj_w"  , nslot);
      m_adj_phi = AdjViewI  ("dirk_adj_phi", nslot);
    }
    m_ig_policy = Homme::get_default_team_policy<ExecSpace>(nelem);
    m_tu_ig = TeamUtils<ExecSpace>(m_ig_policy);
  }

  int requested_buffer_size () const {
    // FunctorsBuffersManager wants the size in terms of sizeof(Real).
    return (Work::shmem_size(nslot) + LinearSystem::shmem_size(nslot))/sizeof(Real);
  }

  void init_buffers (const FunctorsBuffersManager& fbm) {
    PT* mem = reinterpret_cast<PT*>(fbm.get_memory());
    m_work = Work(mem, nslot);
    mem += Work::shmem_size(nslot)/sizeof(PT);
    m_ls = LinearSystem(mem, nslot);
  }

  void run (int nm1, Real alphadt_nm1, int n0, Real alphadt_n0, int np1, Real dt2,
            const ElementsST<ST>& e, const HybridVCoord& hvcoord,
            const bool bfb_solver = default_bfb_solver) {
    if ( ! calc_initial_guess_in_newton_kernel) {
      run_initial_guess(np1, e, hvcoord);
      Kokkos::fence();
    }

    run_newton(nm1, alphadt_nm1, n0, alphadt_n0, np1, dt2, e, hvcoord, bfb_solver);
    Kokkos::fence();
  }

#ifdef HOMMEXX_ENABLE_FAD_TYPES
  // Offsets into the per-column FAD derivative vector for init_J / run_JV.
  // The FAD type DxFadTypeDirk has size NUM_PHYSICAL_LEV*4 + NUM_INTERFACE_LEV*2.
  // For each column (ip,jp) the state variables are encoded as:
  //   [u(0..NUM_PHYSICAL_LEV-1), v(0..NUM_PHYSICAL_LEV-1),
  //    vtheta_dp(0..NUM_PHYSICAL_LEV-1), dp3d(0..NUM_PHYSICAL_LEV-1),
  //    w_i(0..NUM_INTERFACE_LEV-1), phinh_i(0..NUM_INTERFACE_LEV-1)]
  static constexpr int fad_offset_u   = 0;
  static constexpr int fad_offset_v   = NUM_PHYSICAL_LEV;
  static constexpr int fad_offset_vth = 2*NUM_PHYSICAL_LEV;
  static constexpr int fad_offset_dp  = 3*NUM_PHYSICAL_LEV;
  static constexpr int fad_offset_w   = 4*NUM_PHYSICAL_LEV;
  static constexpr int fad_offset_phi = 4*NUM_PHYSICAL_LEV + NUM_INTERFACE_LEV;

  // Initialize the d/dx derivatives of the state so that each state variable
  // gets a unique FAD index encoding its level. Since DIRK has no horizontal
  // (GaussPoint) coupling, each column (ip,jp) is independent and uses the
  // same per-level FAD indices. The input dx_tl specifies with respect of
  // which state time level we are computing derivs.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, DxFadTypeDirk>>
  init_J (const int dx_tl, const ElementsStateST<ST>& state) {
    using md_range_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>;
    const int nelem = state.num_elems();
    auto p5_mid = md_range_t({0,0,0,0,0}, {nelem, NUM_TIME_LEVELS, NP, NP, NUM_PHYSICAL_LEV});
    auto p5_int = md_range_t({0,0,0,0,0}, {nelem, NUM_TIME_LEVELS, NP, NP, NUM_INTERFACE_LEV});

    auto dvdx_v   = ekat::scalarize(state.m_v);
    auto dvthdx_v = ekat::scalarize(state.m_vtheta_dp);
    auto ddpdx_v  = ekat::scalarize(state.m_dp3d);
    auto dwdx_v   = ekat::scalarize(state.m_w_i);
    auto dphidx_v = ekat::scalarize(state.m_phinh_i);

    // Local copies of offsets for lambda capture (KOKKOS_LAMBDA uses [=])
    const int offset_u   = fad_offset_u;
    const int offset_v   = fad_offset_v;
    const int offset_vth = fad_offset_vth;
    const int offset_dp  = fad_offset_dp;
    const int offset_w   = fad_offset_w;
    const int offset_phi = fad_offset_phi;

    auto init_dx_mid = KOKKOS_LAMBDA (int ie, int tl, int ip, int jp, int k) {
      auto& dudx   = dvdx_v  (ie, tl, 0, ip, jp, k);
      auto& dvdx   = dvdx_v  (ie, tl, 1, ip, jp, k);
      auto& dvthdx = dvthdx_v(ie, tl,    ip, jp, k);
      auto& ddpdx  = ddpdx_v (ie, tl,    ip, jp, k);

      auto coeff = tl==dx_tl ? 1 : 0;
      dudx.zero();   dudx.fastAccessDx  (offset_u   + k) = coeff;
      dvdx.zero();   dvdx.fastAccessDx  (offset_v   + k) = coeff;
      dvthdx.zero(); dvthdx.fastAccessDx(offset_vth + k) = coeff;
      ddpdx.zero();  ddpdx.fastAccessDx (offset_dp  + k) = coeff;
    };

    auto init_dx_int = KOKKOS_LAMBDA (int ie, int tl, int ip, int jp, int k) {
      auto& dwdx   = dwdx_v  (ie, tl, ip, jp, k);
      auto& dphidx = dphidx_v(ie, tl, ip, jp, k);

      auto coeff = tl==dx_tl ? 1 : 0;
      dwdx.zero();   dwdx.fastAccessDx  (offset_w   + k) = coeff;
      dphidx.zero(); dphidx.fastAccessDx(offset_phi + k) = coeff;
    };

    Kokkos::parallel_for(p5_mid, init_dx_mid);
    Kokkos::parallel_for(p5_int, init_dx_int);
    m_dx_tl = dx_tl;
  }

  // Compute the product J*V where J = d(state(np1)) / d(state(itl)) (where itl
  // was set in init_J).
  // DIRK only updates w_i and phinh_i at np1; for v, vtheta_dp, dp3d the
  // Jacobian is the identity (DIRK leaves them unchanged).
  // Preconditions: init_J(itl,e) and run(...) must have been called first.
  // The arg state_dx should contain dU(np1)/dU(itl), where itl is the slice
  // we specified during init_J
  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, DxFadTypeDirk>>
  run_JV (const int /*np1*/, const ElementsStateST<ST>& /*state_dx*/,
          const StateSnapshot& /* x */, StateSnapshot& /* y */) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, DxFadTypeDirk>>
  run_JV (const int np1, const ElementsStateST<ST>& state_dx,
          const StateSnapshot& x, StateSnapshot& y)
  {
    // Extract the Jacobian-vector product using the product rule.
    // dw_v(ie,np1,ip,jp,k).dx(j) = d(w_np1(ie,ip,jp,k)) / d(input_j)
    // dphi_v(ie,np1,ip,jp,k).dx(j) = d(phi_np1(ie,ip,jp,k)) / d(input_j)
    const int nelem = state_dx.num_elems();

    auto dw_v   = ekat::scalarize(state_dx.m_w_i);
    auto dphi_v = ekat::scalarize(state_dx.m_phinh_i);

    // Input
    auto x_V   = ekat::scalarize(x.v);
    auto x_vth = ekat::scalarize(x.vtheta_dp);
    auto x_dp  = ekat::scalarize(x.dp3d);
    auto x_w   = ekat::scalarize(x.w_i);
    auto x_phi = ekat::scalarize(x.phinh_i);

    // Output
    auto y_V   = ekat::scalarize(y.v);
    auto y_vth = ekat::scalarize(y.vtheta_dp);
    auto y_dp  = ekat::scalarize(y.dp3d);
    auto y_w   = ekat::scalarize(y.w_i);
    auto y_phi = ekat::scalarize(y.phinh_i);

    // Local copies of offsets for lambda capture
    const int offset_u   = fad_offset_u;
    const int offset_v   = fad_offset_v;
    const int offset_vth = fad_offset_vth;
    const int offset_dp  = fad_offset_dp;
    const int offset_w   = fad_offset_w;
    const int offset_phi = fad_offset_phi;

    using md_range_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
    auto p4_int = md_range_t({0,0,0,0}, {nelem, NP, NP, NUM_INTERFACE_LEV});
    auto p4_mid = md_range_t({0,0,0,0}, {nelem, NP, NP, NUM_PHYSICAL_LEV});

    // Compute J*V for w_i and phinh_i at np1 (non-trivial DIRK blocks)
    auto prod_rule_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      const auto& Jw   = dw_v  (ie, np1, ip, jp, k).dx();
      const auto& Jphi = dphi_v(ie, np1, ip, jp, k).dx();

      Real y_w_new   = 0;
      Real y_phi_new = 0;

      // Contributions from midpoint variables (u, v, vtheta_dp, dp3d)
      for (int k2 = 0; k2 < NUM_PHYSICAL_LEV; ++k2) {
        y_w_new   += Jw  [offset_u   + k2] * x_V  (ie, 0, ip, jp, k2)
                   + Jw  [offset_v   + k2] * x_V  (ie, 1, ip, jp, k2)
                   + Jw  [offset_vth + k2] * x_vth(ie,    ip, jp, k2)
                   + Jw  [offset_dp  + k2] * x_dp (ie,    ip, jp, k2);
        y_phi_new += Jphi[offset_u   + k2] * x_V  (ie, 0, ip, jp, k2)
                   + Jphi[offset_v   + k2] * x_V  (ie, 1, ip, jp, k2)
                   + Jphi[offset_vth + k2] * x_vth(ie,    ip, jp, k2)
                   + Jphi[offset_dp  + k2] * x_dp (ie,    ip, jp, k2);
      }
      // Contributions from interface variables (w_i, phinh_i)
      for (int k2 = 0; k2 < NUM_INTERFACE_LEV; ++k2) {
        y_w_new   += Jw  [offset_w   + k2] * x_w  (ie, ip, jp, k2)
                   + Jw  [offset_phi + k2] * x_phi(ie, ip, jp, k2);
        y_phi_new += Jphi[offset_w   + k2] * x_w  (ie, ip, jp, k2)
                   + Jphi[offset_phi + k2] * x_phi(ie, ip, jp, k2);
      }

      y_w  (ie, ip, jp, k) = y_w_new;
      y_phi(ie, ip, jp, k) = y_phi_new;
    };

    // Identity blocks: DIRK does not modify v, vtheta_dp, or dp3d
    auto prod_rule_mid_id = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      y_V  (ie, 0, ip, jp, k) = x_V  (ie, 0, ip, jp, k);
      y_V  (ie, 1, ip, jp, k) = x_V  (ie, 1, ip, jp, k);
      y_vth(ie,    ip, jp, k) = x_vth(ie,    ip, jp, k);
      y_dp (ie,    ip, jp, k) = x_dp (ie,    ip, jp, k);
    };

    Kokkos::parallel_for(p4_int, prod_rule_int);
    Kokkos::parallel_for(p4_mid, prod_rule_mid_id);
  }

  // Compute the product J^T*V where J = d(state(np1)) / d(state(itl)) (where itl
  // was set in init_J).
  // DIRK only updates w_i and phinh_i; for v, vtheta_dp, dp3d the
  // Jacobian is the identity (DIRK leaves them unchanged), so J^T also
  // contributes an identity block for those variables.
  // Preconditions: init_J(itl,e) and run(...) must have been called first.
  // The arg state_dx should contain dU(np1)/dU(itl), where itl is the slice
  // we specified during init_J
  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, DxFadTypeDirk>>
  run_JtV (const int /*np1*/, const ElementsStateST<ST>& /*state_dx*/,
           const StateSnapshot& /*x*/, StateSnapshot& /*y*/) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, DxFadTypeDirk>>
  run_JtV (const int np1, const ElementsStateST<ST>& state_dx,
           const StateSnapshot& x, StateSnapshot& y)
  {
    // J has the following block structure (rows = np1 outputs, cols = itl inputs):
    //   - Identity block for u, v, vtheta_dp, dp3d (DIRK does not modify them)
    //   - Full blocks for w_i and phinh_i (computed by the Newton solve)
    //
    // J^T therefore:
    //   - For interface inputs (w, phi): sum Jacobian rows of w/phi at np1
    //   - For midpoint inputs (u, v, vth, dp): identity + cross-contributions
    //     from the w/phi rows of J (since J[w_np1(k), u_itl(k2)] != 0 in general)
    const int nelem = state_dx.num_elems();

    auto dw_v   = ekat::scalarize(state_dx.m_w_i);
    auto dphi_v = ekat::scalarize(state_dx.m_phinh_i);

    // Input
    auto x_V   = ekat::scalarize(x.v);
    auto x_vth = ekat::scalarize(x.vtheta_dp);
    auto x_dp  = ekat::scalarize(x.dp3d);
    auto x_w   = ekat::scalarize(x.w_i);
    auto x_phi = ekat::scalarize(x.phinh_i);

    // Output
    auto y_V   = ekat::scalarize(y.v);
    auto y_vth = ekat::scalarize(y.vtheta_dp);
    auto y_dp  = ekat::scalarize(y.dp3d);
    auto y_w   = ekat::scalarize(y.w_i);
    auto y_phi = ekat::scalarize(y.phinh_i);

    // Local copies of offsets for lambda capture
    const int offset_u   = fad_offset_u;
    const int offset_v   = fad_offset_v;
    const int offset_vth = fad_offset_vth;
    const int offset_dp  = fad_offset_dp;
    const int offset_w   = fad_offset_w;
    const int offset_phi = fad_offset_phi;

    using md_range_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
    auto p4_int = md_range_t({0,0,0,0}, {nelem, NP, NP, NUM_INTERFACE_LEV});
    auto p4_mid = md_range_t({0,0,0,0}, {nelem, NP, NP, NUM_PHYSICAL_LEV});

    // Compute (J^T * l)_w(k2) and (J^T * l)_phi(k2) for interface inputs k2.
    // Each output sums contributions from all interface levels k of the w and phi rows of J.
    // J^T[w_itl(k2), w_np1(k)]   = J[w_np1(k), w_itl(k2)]   = dw_v(k).dx(offset_w + k2)
    // J^T[w_itl(k2), phi_np1(k)] = J[phi_np1(k), w_itl(k2)] = dphi_v(k).dx(offset_w + k2)
    // (and similarly for phi input)
    auto jtv_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k2) {
      Real y_w_new   = 0;
      Real y_phi_new = 0;

      for (int k = 0; k < NUM_INTERFACE_LEV; ++k) {
        const Real x_w_k   = x_w  (ie, ip, jp, k);
        const Real x_phi_k = x_phi(ie, ip, jp, k);

        y_w_new   += dw_v  (ie, np1, ip, jp, k).dx(offset_w   + k2) * x_w_k
                   + dphi_v(ie, np1, ip, jp, k).dx(offset_w   + k2) * x_phi_k;
        y_phi_new += dw_v  (ie, np1, ip, jp, k).dx(offset_phi + k2) * x_w_k
                   + dphi_v(ie, np1, ip, jp, k).dx(offset_phi + k2) * x_phi_k;
      }

      y_w  (ie, ip, jp, k2) = y_w_new;
      y_phi(ie, ip, jp, k2) = y_phi_new;
    };

    // Compute (J^T * x)_u(k2), _v(k2), _vth(k2), _dp(k2) for midpoint inputs k2.
    // When dx_tl==np1 (primary DIRK input), DIRK leaves v/vtheta/dp unchanged,
    // so J has an identity block and J^T adds x back. When dx_tl!=np1 (background
    // slot), d(v_np1)/d(v_bg)=0 (no identity contribution).
    const bool add_identity = (m_dx_tl == np1);
    auto jtv_mid = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k2) {
      Real contrib_u   = 0;
      Real contrib_v   = 0;
      Real contrib_vth = 0;
      Real contrib_dp  = 0;

      for (int k = 0; k < NUM_INTERFACE_LEV; ++k) {
        const Real x_w_k   = x_w  (ie, ip, jp, k);
        const Real x_phi_k = x_phi(ie, ip, jp, k);

        contrib_u   += dw_v  (ie, np1, ip, jp, k).dx(offset_u   + k2) * x_w_k
                     + dphi_v(ie, np1, ip, jp, k).dx(offset_u   + k2) * x_phi_k;
        contrib_v   += dw_v  (ie, np1, ip, jp, k).dx(offset_v   + k2) * x_w_k
                     + dphi_v(ie, np1, ip, jp, k).dx(offset_v   + k2) * x_phi_k;
        contrib_vth += dw_v  (ie, np1, ip, jp, k).dx(offset_vth + k2) * x_w_k
                     + dphi_v(ie, np1, ip, jp, k).dx(offset_vth + k2) * x_phi_k;
        contrib_dp  += dw_v  (ie, np1, ip, jp, k).dx(offset_dp  + k2) * x_w_k
                     + dphi_v(ie, np1, ip, jp, k).dx(offset_dp  + k2) * x_phi_k;
      }

      // Identity block: only when the seeded slot equals the solve slot (np1)
      const Real id_u   = add_identity ? x_V  (ie, 0, ip, jp, k2) : 0;
      const Real id_v   = add_identity ? x_V  (ie, 1, ip, jp, k2) : 0;
      const Real id_vth = add_identity ? x_vth(ie,    ip, jp, k2) : 0;
      const Real id_dp  = add_identity ? x_dp (ie,    ip, jp, k2) : 0;
      y_V  (ie, 0, ip, jp, k2) = id_u   + contrib_u;
      y_V  (ie, 1, ip, jp, k2) = id_v   + contrib_v;
      y_vth(ie,    ip, jp, k2) = id_vth + contrib_vth;
      y_dp (ie,    ip, jp, k2) = id_dp  + contrib_dp;
    };

    Kokkos::parallel_for(p4_int, jtv_int);
    Kokkos::parallel_for(p4_mid, jtv_mid);
  }
#endif // HOMMEXX_ENABLE_FAD_TYPES

#ifdef HOMMEXX_ENABLE_FAD_TYPES
  // ---- Adjoint of a DIRK stage (implicit function theorem) ---------------------------------------
  //
  // A stage solves r(w*; p) = 0 for w* (see calc_w0_phi0 for r), where p are the inputs of the stage
  // (the state at np1 before the stage and, if alphadt_n0/alphadt_nm1 are not 0, the states at n0 and
  // nm1), and then sets phi = phi0(p) + dt2*g*w*. At the converged solution, dw*/dp = -(dr/dw)^{-1} dr/dp,
  // so, for the seed mu on the output of the stage, J^T*mu has these pieces (levels 0..nlev-1):
  //   omega = mu_w + dt2*g*mu_phi,    z = (dr/dw)^{-T} omega,
  //   (J^T*mu)_p = -z^T dr/dp + mu_phi^T dphi0/dp,
  // plus an identity block for v, vtheta_dp, dp3d (the stage does not change them), and for w_i at the
  // surface (the stage does not solve for it). dr/dw is the tridiagonal matrix of calc_jacobian.
  //
  // The vector -z^T dr/dp + mu_phi^T dphi0/dp is the derivative w.r.t. p of sum_k [z_k*x_k + mu_phi_k*phi0_k],
  // where x=-r and z, mu_phi, and w* are held fixed. So, we evaluate, with the same code as the forward
  // solve, w0, phi0, and x, in a Fad type that carries the derivatives w.r.t. the state of the time level
  // we want the product for, with a given z. The rows k of x and phi0 only depend on a few levels of
  // each variable (the stencil of the equations). So, instead of one derivative per level, the levels l
  // of a variable are 'colored': l goes in the slot base + l%m, with m the number of levels a row depends
  // on. Then the derivative w.r.t. level l is the sum, over the rows k that depend on l, of the
  // derivative of t_k = z_k*x_k + mu_phi_k*phi0_k w.r.t. the slot of l.
  //
  // Row k depends on the levels [k-a, k+b] of each variable (this is a bound, shared by the 3 time levels):
  //   w_i: [k-1,k+1],  vtheta_dp: [k-1,k],  dp3d and v: [k-2,k+1],  phinh_i: [k-1,k+1]
  // (D_k involves pnh_{k-1}, pnh_k, which depend on vtheta_dp_m and dphi_m for m=k-1,k, and thus on
  // phi0_j for j=k-1..k+1, which involves phi_j, w_j, and the vertical velocity induced by the topography
  // gwh_j, which depends on dp and v at the levels j-1, j).
  struct AdjWin { int a, b, m, base; };
  static constexpr AdjWin adj_win_w   = {1,1,3, 0};
  static constexpr AdjWin adj_win_vth = {1,0,2, 3};
  static constexpr AdjWin adj_win_dp  = {2,1,4, 5};
  static constexpr AdjWin adj_win_v1  = {2,1,4, 9};
  static constexpr AdjWin adj_win_v2  = {2,1,4,13};
  static constexpr AdjWin adj_win_phi = {1,1,3,17};

  // Copy a Real field (Hxx format, midpoints or interfaces) into a Fad one, and, if seed, seed the
  // derivative of the level l in the slot base + l%m. Must be called by the whole team.
  template <typename SrcView, typename DstView>
  KOKKOS_INLINE_FUNCTION
  static void adj_copy_seed (const KernelVariables& kv, const SrcView& src, const DstView& dst,
                             const int nlev_field, const bool seed, const AdjWin& win) {
    const auto f = [&] (const int idx) {
      const int gi = idx / NP, gj = idx % NP;
      const auto g = [&] (const int pk) {
        for (int s = 0; s < VECTOR_SIZE; ++s) {
          const int l = pk*VECTOR_SIZE + s;
          auto& d = dst(gi,gj,pk)[s];
          d.zero();
          d.val() = src(gi,gj,pk)[s];
          if (seed && l < nlev_field)
            d.fastAccessDx(win.base + l % win.m) = 1;
        }
      };
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, dst.extent_int(2)), g);
    };
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP*NP), f);
  }

  // For each level l of a variable, and for each column, call store(gi,gj,l,value), with value
  // the sum, over the rows k of t that depend on l, of the derivative of t_k w.r.t. the slot of l.
  // Must be called by the whole team.
  template <typename Store>
  KOKKOS_INLINE_FUNCTION
  static void adj_uncolor (const KernelVariables& kv, const LinearSystemSlot& t, const AdjWin& win,
                           const int nlev_field, const int nrows, const Store& store) {
    const auto f = [&] (const int l) {
      const int klo = l - win.b > 0 ? l - win.b : 0;
      const int khi = l + win.a < nrows - 1 ? l + win.a : nrows - 1;
      const int slot = win.base + l % win.m;
      const auto g = [&] (const int i) {
        for (int s = 0; s < packn; ++s) {
          const int idx = packn*i + s;
          if (idx >= scaln) break;
          Real val = 0;
          for (int k = klo; k <= khi; ++k)
            val += t(k,i)[s].fastAccessDx(slot);
          store(idx / NP, idx % NP, l, val);
        }
      };
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, npack), g);
    };
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, nlev_field), f);
  }

  // J^T*mu for one DIRK stage, which took s_pre as the np1 state and gave s_post (only w_i is used).
  // J_np1 = d(state after)/d(s_pre) goes to out_pre, and, if not null, J_n0 and J_nm1 (the derivs w.r.t.
  // the states s_n0 and s_nm1, which the stage depends on iff alphadt_n0 and alphadt_nm1 are not 0)
  // go to out_n0 and out_nm1. s_n0 (s_nm1) is not used, and can be null, if alphadt_n0 (alphadt_nm1) is 0.
  // All the outputs are fully overwritten. mu and out_pre may be the same object.
  // This is the adjoint of run (with the same arguments), but the Newton iterations are not replayed:
  // the converged solution is read from s_post. See above for the math. Only for ST=DirkAdjFadType.
  void run_adjoint (const Real alphadt_nm1, const Real alphadt_n0, const Real dt2,
                    const ElementsGeometry& geo, const HybridVCoord& hvcoord,
                    const StateSnapshot* s_nm1, const StateSnapshot* s_n0,
                    const StateSnapshot& s_pre, const StateSnapshot& s_post,
                    const StateSnapshot& mu,
                    StateSnapshot& out_pre, StateSnapshot* out_n0, StateSnapshot* out_nm1,
                    const bool bfb_solver = default_bfb_solver) {
    static_assert(is_adj_fad, "run_adjoint is only available for ST=DirkAdjFadType");
    EKAT_REQUIRE_MSG (alphadt_nm1==0 or s_nm1!=nullptr,
        "[DirkFunctorImpl::run_adjoint] Error! The nm1 state is needed if alphadt_nm1!=0.\n");
    EKAT_REQUIRE_MSG (alphadt_n0==0 or s_n0!=nullptr,
        "[DirkFunctorImpl::run_adjoint] Error! The n0 state is needed if alphadt_n0!=0.\n");
    GPTLstart("dirk_stage_adjoint");

    // If a state is not needed, pass in another valid one: it won't be read
    const StateSnapshot& snm1 = s_nm1 ? *s_nm1 : s_pre;
    const StateSnapshot& sn0  = s_n0  ? *s_n0  : s_pre;

    // The np1 output goes last, since out_pre may alias mu (which all the passes read)
    if (out_n0)  run_adjoint_pass(n0_tl,  alphadt_nm1, alphadt_n0, dt2, geo, hvcoord, snm1, sn0, s_pre, s_post, mu, *out_n0,  bfb_solver);
    if (out_nm1) run_adjoint_pass(nm1_tl, alphadt_nm1, alphadt_n0, dt2, geo, hvcoord, snm1, sn0, s_pre, s_post, mu, *out_nm1, bfb_solver);
    run_adjoint_pass(np1_tl, alphadt_nm1, alphadt_n0, dt2, geo, hvcoord, snm1, sn0, s_pre, s_post, mu, out_pre, bfb_solver);

    GPTLstop("dirk_stage_adjoint");
  }

  // The time levels, as seen by calc_w0_phi0 in run_adjoint
  static constexpr int nm1_tl = 0, n0_tl = 1, np1_tl = 2;

  // J^T*mu for the derivs w.r.t. the state at time level seed_tl (see run_adjoint)
  void run_adjoint_pass (const int seed_tl, const Real alphadt_nm1, const Real alphadt_n0, const Real dt2,
                         const ElementsGeometry& geo, const HybridVCoord& hvcoord,
                         const StateSnapshot& s_nm1, const StateSnapshot& s_n0,
                         const StateSnapshot& s_pre, const StateSnapshot& s_post,
                         const StateSnapshot& mu, StateSnapshot& out, const bool bfb_solver) {
    using Kokkos::subview;
    using Kokkos::parallel_reduce;
    const auto a = Kokkos::ALL();
    const bool is_np1 = seed_tl == np1_tl;

    const auto work = m_work;
    const auto ls = m_ls;
    const auto tu = m_tu;
    const auto policy = m_policy;
    const auto adj_v = m_adj_v, adj_vth = m_adj_vth, adj_dp = m_adj_dp, adj_w = m_adj_w, adj_phi = m_adj_phi;
    const auto eos = m_eos;
    const auto constants = eos.m_constants;
    const auto gradphis = geo.m_gradphis;
    const auto hybi = hvcoord.hybrid_bi;

    // Shallow copies, to capture in the kernel
    const StateSnapshot nm1s = s_nm1, n0s = s_n0, pre = s_pre, post = s_post, muc = mu;
    const auto o_v   = ekat::scalarize(out.v);
    const auto o_vth = ekat::scalarize(out.vtheta_dp);
    const auto o_dp  = ekat::scalarize(out.dp3d);
    const auto o_w   = ekat::scalarize(out.w_i);
    const auto o_phi = ekat::scalarize(out.phinh_i);
    const auto m_v   = ekat::scalarize(muc.v);
    const auto m_vth = ekat::scalarize(muc.vtheta_dp);
    const auto m_dp  = ekat::scalarize(muc.dp3d);
    const auto m_w   = ekat::scalarize(muc.w_i);

    const AdjWin win_w = adj_win_w, win_vth = adj_win_vth, win_dp = adj_win_dp,
                 win_v1 = adj_win_v1, win_v2 = adj_win_v2, win_phi = adj_win_phi;

    // Tolerance on the residual of the stage equations at the solution that we read from s_post
    const Real res_tol = 1e-5;

    const auto toplevel = KOKKOS_LAMBDA (const MT& team, int& nerr) {
      KernelVariables kv(team, tu);
      const auto ie = kv.ie;
      const int slot = kv.team_idx;
      const int nlev = num_phys_lev;
      const int nvec = npack;
      const auto grav = constants.g();

      // The slots are the same as in run_newton
      const auto
      phi_n0    = get_work_slot(work, slot,  0),
      phi_np1   = get_work_slot(work, slot,  1),
      dphi      = get_work_slot(work, slot,  2),
      w_n0      = get_work_slot(work, slot,  3),
      w_np1     = get_work_slot(work, slot,  4),
      dpnh_dp_i = get_work_slot(work, slot,  5),
      gwh_i     = get_work_slot(work, slot,  6),
      dphi_n0   = get_work_slot(work, slot,  6), // reuse gwh_i
      vtheta_dp = get_work_slot(work, slot,  7),
      dp3d      = get_work_slot(work, slot,  8),
      pnh       = get_work_slot(work, slot,  9),
      wrk       = get_work_slot(work, slot, 10),
      xfull     = get_work_slot(work, slot, 11);
      const auto
      dl = get_ls_slot(ls, slot, 0),
      d  = get_ls_slot(ls, slot, 1),
      du = get_ls_slot(ls, slot, 2),
      scratch = get_ls_slot(ls, slot, 3);
      LinearSystemSlot x = subview(xfull, Kokkos::pair<int,int>(0,nlev), a);
      LinearSystemSlot z = subview(wrk,   Kokkos::pair<int,int>(0,nlev), a);

      // Load a Real state into the Fad scratch, seeding the derivs if it is the state we differentiate
      // w.r.t. (the np1 state in buffer 0; the n0 and nm1 states, one at a time, in buffer 1).
      const auto state_at = [&] (const int nt) {
        const StateSnapshot& s = nt == np1_tl ? pre : (nt == n0_tl ? n0s : nm1s);
        const int buf = nt == np1_tl ? 0 : 1;
        const bool seed = nt == seed_tl;
        adj_copy_seed(kv, subview(s.v,ie,0,a,a,a), subview(adj_v,slot,buf,0,a,a,a), nlev, seed, win_v1);
        adj_copy_seed(kv, subview(s.v,ie,1,a,a,a), subview(adj_v,slot,buf,1,a,a,a), nlev, seed, win_v2);
        adj_copy_seed(kv, subview(s.vtheta_dp,ie,a,a,a), subview(adj_vth,slot,buf,a,a,a), nlev,   seed, win_vth);
        adj_copy_seed(kv, subview(s.dp3d,ie,a,a,a),      subview(adj_dp, slot,buf,a,a,a), nlev,   seed, win_dp);
        adj_copy_seed(kv, subview(s.w_i,ie,a,a,a),       subview(adj_w,  slot,buf,a,a,a), nlev+1, seed, win_w);
        adj_copy_seed(kv, subview(s.phinh_i,ie,a,a,a),   subview(adj_phi,slot,buf,a,a,a), nlev+1, seed, win_phi);
        kv.team_barrier();
        return make_state_views(subview(adj_w,  slot,buf,a,a,a), subview(adj_phi,slot,buf,a,a,a),
                                subview(adj_vth,slot,buf,a,a,a), subview(adj_dp, slot,buf,a,a,a),
                                subview(adj_v,  slot,buf,a,a,a,a));
      };

      // w0, phi0 (in w_n0 and phi_n0), and the np1 vtheta_dp, dp3d (all with derivs)
      const auto wmax = calc_w0_phi0(kv, hvcoord, eos, dt2, nm1_tl, alphadt_nm1, n0_tl, alphadt_n0, np1_tl,
                                     state_at, subview(gradphis,ie,a,a,a), hybi,
                                     phi_n0, phi_np1, dphi, w_n0, w_np1, dpnh_dp_i, gwh_i,
                                     vtheta_dp, dp3d, pnh, wrk, nerr);
      kv.team_barrier();

      // The converged solution (no derivs), and the dphi it implies
      transpose(kv, nlev+1, subview(post.w_i,ie,a,a,a), w_np1);
      loop_ki(kv, nlev, nvec, [&] (const int k, const int i) { dphi_n0(k,i) = phi_n0(k+1,i) - phi_n0(k,i); });
      kv.team_barrier();
      loop_ki(kv, nlev-1, nvec, [&] (const int k, const int i) {
        dphi(k,i) = dphi_n0(k,i) + dt2*grav*(w_np1(k+1,i) - w_np1(k,i));
      });
      loop_ki(kv, 1, nvec, [&] (const int, const int i) {
        const int k = nlev-1;
        dphi(k,i) = dphi_n0(k,i) - dt2*grav*w_np1(k,i);
      });
      kv.team_barrier();

      // x = -r(w*), and the Jacobian dr/dw
      const bool ok = pnh_and_exner_from_eos(kv, hvcoord, eos, vtheta_dp, dp3d, dphi, pnh, wrk, dpnh_dp_i);
      if ( ! ok) nerr = 1;
      kv.team_barrier();
      calc_neg_residual(kv, nlev, nvec, grav, dt2, w_np1, w_n0, dpnh_dp_i, x);
      calc_jacobian(kv, constants, dt2, dp3d, dphi, pnh, dl, d, du);
      kv.team_barrier();

#ifndef NDEBUG
      // If w* is not the root of r (e.g., the forward Newton solve did not converge), the adjoint is wrong
      ST deltaerr = 0;
      if ( ! exit_on_step(kv, nlev, nvec, wmax, res_tol, x, deltaerr)) nerr = 1;
      kv.team_barrier();
#else
      (void) wmax; (void) res_tol;
#endif

      // z = J^{-T}*omega, with omega = mu_w + dt2*g*mu_phi. mu_phi stays in phi_np1, for later
      transpose(kv, nlev+1, subview(muc.w_i,    ie,a,a,a), wrk);
      transpose(kv, nlev+1, subview(muc.phinh_i,ie,a,a,a), phi_np1);
      kv.team_barrier();
      loop_ki(kv, nlev, nvec, [&] (const int k, const int i) { wrk(k,i) += dt2*grav*phi_np1(k,i); });
      kv.team_barrier();
      solve_transpose(kv, dl, d, du, scratch, z, bfb_solver);
      kv.team_barrier();

      // z is a constant for this purpose (drop the derivs it got from J)
      loop_ki(kv, nlev, nvec, [&] (const int k, const int i) {
        for (int s = 0; s < packn; ++s)
          z(k,i)[s] = ADValue(z(k,i)[s]);
      });
      kv.team_barrier();

      // t_k = z_k*x_k + mu_phi_k*phi0_k, whose derivs are the product we want
      loop_ki(kv, nlev, nvec, [&] (const int k, const int i) {
        x(k,i) = z(k,i)*x(k,i) + phi_np1(k,i)*phi_n0(k,i);
      });
      kv.team_barrier();

      // The np1 output also has the identity blocks (v, vtheta_dp, dp3d, and the surface w).
      // Note: out and mu may alias; each entry is read, if at all, only by the one thread that writes it
      adj_uncolor(kv, x, win_v1, nlev, nlev, [&] (const int gi, const int gj, const int l, const Real val) {
        o_v(ie,0,gi,gj,l) = val + (is_np1 ? m_v(ie,0,gi,gj,l) : 0);
      });
      adj_uncolor(kv, x, win_v2, nlev, nlev, [&] (const int gi, const int gj, const int l, const Real val) {
        o_v(ie,1,gi,gj,l) = val + (is_np1 ? m_v(ie,1,gi,gj,l) : 0);
      });
      adj_uncolor(kv, x, win_vth, nlev, nlev, [&] (const int gi, const int gj, const int l, const Real val) {
        o_vth(ie,gi,gj,l) = val + (is_np1 ? m_vth(ie,gi,gj,l) : 0);
      });
      adj_uncolor(kv, x, win_dp, nlev, nlev, [&] (const int gi, const int gj, const int l, const Real val) {
        o_dp(ie,gi,gj,l) = val + (is_np1 ? m_dp(ie,gi,gj,l) : 0);
      });
      adj_uncolor(kv, x, win_w, nlev+1, nlev, [&] (const int gi, const int gj, const int l, const Real val) {
        o_w(ie,gi,gj,l) = (is_np1 && l == nlev) ? m_w(ie,gi,gj,l) : val;
      });
      adj_uncolor(kv, x, win_phi, nlev+1, nlev, [&] (const int gi, const int gj, const int l, const Real val) {
        o_phi(ie,gi,gj,l) = val;
      });
    };

    int nerr = 0;
    parallel_reduce(policy, toplevel, nerr);
    Kokkos::fence();
    // (The check that the state after the stage is the root of the stage equations is only done in debug builds)
    EKAT_REQUIRE_MSG (nerr == 0,
        "[DirkFunctorImpl::run_adjoint] Error! In some columns, either vtheta_dp<0 or dphi>0, or (debug builds\n"
        "only) the state after the stage is not the root of the stage equations (did the forward Newton solve converge?).\n");
  }
#endif // HOMMEXX_ENABLE_FAD_TYPES

  // Optimal impl of phi_from_eos for the initial guess. See comments for the
  // function phi_from_eos, below, for discussion. This kernel uses standard
  // Hommexx layout and parallelization approaches to compute the scans
  // optimally.
  void run_initial_guess (int np1, const ElementsST<ST>& e, const HybridVCoord& hvcoord) {
    using Kokkos::subview;
    using Kokkos::parallel_for;

    const auto e_phis = e.m_geometry.m_phis;
    const auto e_vtheta_dp = e.m_state.m_vtheta_dp;
    const auto e_dp3d = e.m_state.m_dp3d;
    const auto e_phinh_i = e.m_derived.m_divdp_proj;

    ElementOps elem_ops;
    elem_ops.init(hvcoord);

    const auto work = m_work;
    const auto tu   = m_tu_ig;

    auto eos = m_eos;
    const auto toplevel = KOKKOS_LAMBDA (const MT& team) {
      KernelVariables kv(team, tu);
      const auto ie = kv.ie;

      const ExecViewUnmanaged<PT[NP][NP][NUM_LEV_P]>
        p_i  (reinterpret_cast<PT*>(get_work_slot(work, kv.team_idx, 0).data())),
        phi_i(reinterpret_cast<PT*>(get_work_slot(work, kv.team_idx, 2).data()));
      const ExecViewUnmanaged<PT[NP][NP][NUM_LEV  ]>
        p    (reinterpret_cast<PT*>(get_work_slot(work, kv.team_idx, 1).data()));

      const auto f = [&] (const int idx) {
        const int igp = idx / NP, jgp = idx % NP;

        const auto p_i_c = Homme::subview(p_i,igp,jgp);
        const auto p_c = Homme::subview(p,igp,jgp);
        const auto phi_i_c = Homme::subview(phi_i,igp,jgp);

        elem_ops.compute_hydrostatic_p(kv, Homme::subview(e_dp3d,ie,np1,igp,jgp), p_i_c, p_c);
        eos.compute_phi_i(kv, e_phis(ie,igp,jgp),
                          Homme::subview(e_vtheta_dp,ie,np1,igp,jgp),
                          p_c, phi_i_c);

        // Copy phi_i for use in run_newton. Don't need nlevp, since that's
        // always phis.
        const auto g = [&] (const int k) { e_phinh_i(ie,igp,jgp,k) = phi_i_c(k); };
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV), g);
      };
      parallel_for(Kokkos::TeamThreadRange(kv.team, NP*NP), f);
    };
    Kokkos::parallel_for(m_ig_policy, toplevel);
  }

  void run_newton (int nm1, Real alphadt_nm1, int n0, Real alphadt_n0, int np1, Real dt2,
                   const ElementsST<ST>& e, const HybridVCoord& hvcoord, const bool bfb_solver) {
    using Kokkos::subview;
    using Kokkos::parallel_for;
    const auto a = Kokkos::ALL();

    const auto grav = m_eos.m_constants.g();
    const int nvec = npack;
    const int maxiter = 20;
#ifdef HOMMEXX_BFB_TESTING
    const Real deltatol = 1e-6; // In bfb testing, use coarse tolerance, due to zeroulp calls
#else
    const Real deltatol = 1e-11; // exit if newton increment < deltatol
#endif

    const auto work = m_work;
    const auto ls = m_ls;
    const auto e_w_i = e.m_state.m_w_i;
    const auto e_vtheta_dp = e.m_state.m_vtheta_dp;
    const auto e_phinh_i = e.m_state.m_phinh_i;
    const auto e_dp3d = e.m_state.m_dp3d;
    const auto e_v = e.m_state.m_v;
    const auto e_phis = e.m_geometry.m_phis;
    const auto e_gradphis = e.m_geometry.m_gradphis;
    const auto e_initial_guess = e.m_derived.m_divdp_proj;
    const auto hybi = hvcoord.hybrid_bi;
    const auto tu   = m_tu;
    const auto eos = m_eos;
    const auto constants = eos.m_constants;
    const auto verbose = m_verbose;

    const auto toplevel = KOKKOS_LAMBDA (const MT& team, int& nerr) {
      KernelVariables kv(team, tu);
      const auto ie = kv.ie;
      const int nlev = num_phys_lev;

      const auto
      phi_n0    = get_work_slot(work, kv.team_idx,  0),
      phi_np1   = get_work_slot(work, kv.team_idx,  1),
      dphi      = get_work_slot(work, kv.team_idx,  2),
      w_n0      = get_work_slot(work, kv.team_idx,  3),
      w_np1     = get_work_slot(work, kv.team_idx,  4),
      dpnh_dp_i = get_work_slot(work, kv.team_idx,  5),
      gwh_i     = get_work_slot(work, kv.team_idx,  6),
      dphi_n0   = get_work_slot(work, kv.team_idx,  6), // reuse gwh_i
      vtheta_dp = get_work_slot(work, kv.team_idx,  7),
      dp3d      = get_work_slot(work, kv.team_idx,  8),
      pnh       = get_work_slot(work, kv.team_idx,  9),
      wrk       = get_work_slot(work, kv.team_idx, 10),
      xfull     = get_work_slot(work, kv.team_idx, 11);
      const auto
      dl = get_ls_slot(ls, kv.team_idx, 0),
      d  = get_ls_slot(ls, kv.team_idx, 1),
      du = get_ls_slot(ls, kv.team_idx, 2);

      // View of xfull for use in the solver. We want xfull so that we
      // can use the nlevp-1 entry, which we make sure is 0, when convenient.
      LinearSystemSlot x = subview(xfull, Kokkos::pair<int,int>(0,nlev), a);
      xfull(nlev,0)[0] = 0.0;

      // Compute w_n0, phi_n0 (the 'known' parts of the equations), and the np1 vtheta_dp and dp3d.
      const auto state_at = [&] (const int nt) {
        return make_state_views(subview(e_w_i      ,ie,nt,a,a,a),
                                subview(e_phinh_i  ,ie,nt,a,a,a),
                                subview(e_vtheta_dp,ie,nt,a,a,a),
                                subview(e_dp3d     ,ie,nt,a,a,a),
                                subview(e_v        ,ie,nt,a,a,a,a));
      };
      const auto wmax = calc_w0_phi0(kv, hvcoord, eos, dt2, nm1, alphadt_nm1, n0, alphadt_n0, np1,
                                     state_at, subview(e_gradphis,ie,a,a,a), hybi,
                                     phi_n0, phi_np1, dphi, w_n0, w_np1, dpnh_dp_i, gwh_i,
                                     vtheta_dp, dp3d, pnh, wrk, nerr);

      // Initial guess for phi_np1.
      if (calc_initial_guess_in_newton_kernel) {
        // Use hydrostatic phi.
        phi_from_eos(kv, eos, nlev, nvec, hvcoord, subview(e_phis,ie,a,a), vtheta_dp, dp3d, phi_np1);
      } else {
        // Copy initial guess from where run_initial_guess stashed it.
        transpose(kv, nlev, subview(e_initial_guess,ie,a,a,a), phi_np1);
        loop_ki(kv, 1, nvec, [&] (int, int i) { set_phis(i, subview(e_phis,ie,a,a), phi_np1); });
      }
      kv.team_barrier();
      loop_ki(kv, nlev, nvec, [&] (int k, int i) { dphi(k,i) = phi_np1(k+1,i) - phi_np1(k,i); });
      kv.team_barrier();
      // If any dphi > -g in a column, set it to -g and integrate to get a
      // new initial phi_np1 and w_np1.
      calc_whether_gt_and_set(kv, nlev, nvec, -ADValue(grav), dphi, wrk);
      kv.team_barrier();
      if (wrk(1,0)[0] == 1) {
        scan_dphi(kv, nlev, nvec, wrk, dphi, phi_np1);
        kv.team_barrier();
      }

      // Initial guess for w_np1.
      loop_ki(kv, nlev, nvec, [&] (int k, int i) { w_np1(k,i) = (phi_np1(k,i) - phi_n0(k,i))/(dt2*grav); });

      loop_ki(kv, nlev, nvec, [&] (int k, int i) { dphi_n0(k,i) = phi_n0(k+1,i) - phi_n0(k,i); });

      int it = 0;
      ST deltaerr = 0;
      for (; it < maxiter; ++it) { // Newton iteration
        const bool ok = pnh_and_exner_from_eos(kv, hvcoord, eos, vtheta_dp, dp3d,
                                               dphi, pnh, wrk, dpnh_dp_i);
        if ( ! ok) nerr = 1;
        kv.team_barrier();
        calc_neg_residual(kv, nlev, nvec, grav, dt2, w_np1, w_n0, dpnh_dp_i, x);

        calc_jacobian(kv, constants, dt2, dp3d, dphi, pnh, dl, d, du);
        kv.team_barrier();
        if (bfb_solver) solvebfb(kv, dl, d, du, x); else solve(kv, dl, d, du, x);
        kv.team_barrier();

        loop_ki(kv, 1, nvec, [&] (int k, int i) { wrk(2,i) = 1; });
        kv.team_barrier();
        for (int nsafe = 0; nsafe < 2; ++nsafe) {
          loop_ki(kv, nlev-1, nvec, [&] (int k, int i) {
            dphi(k,i) = dphi_n0(k,i) + dt2*grav*(         (w_np1(k+1,i) - w_np1(k,i)) +
                                                 wrk(2,i)*(    x(k+1,i) -     x(k,i)));
          });
          loop_ki(kv, 1, nvec, [&] (int, int i) {
            const auto k = nlev-1;
            dphi(k,i) = dphi_n0(k,i) - dt2*grav*(w_np1(k,i) + wrk(2,i)*x(k,i));
          });
          kv.team_barrier();
          calc_whether_ge(kv, nlev, nvec, 0, dphi, wrk);
          kv.team_barrier();
          if (wrk(1,0)[0] == 0) break;
          calc_step_size(kv, nlev, nvec, grav, dt2, dphi_n0, w_np1, x, wrk);
          kv.team_barrier();
        }
        kv.team_barrier();

        loop_ki(kv, nlev, nvec, [&] (int k, int i) { w_np1(k,i) += wrk(2,i)*x(k,i); });

        if (exit_on_step(kv, nlev, nvec, wmax, deltatol, x, deltaerr)) break;
      } // Newton iteration
      kv.team_barrier();

      if (it >= maxiter and verbose) {
        Kokkos::printf("[DIRK] WARNING! Newton reached max iteration count,"
                       " with deltaerr = %3.17f\n", ADValue(deltaerr));
        nerr = 1;
      }

      // Update phi_np1.
      loop_ki(kv, nlev, nvec, [&] (int k, int i) { phi_np1(k,i) = phi_n0(k,i) + dt2*grav*w_np1(k,i); });

      kv.team_barrier();
      transpose(kv, nlev+1, phi_np1, subview(e_phinh_i,ie,np1,a,a,a));
      transpose(kv, nlev+1, w_np1,   subview(e_w_i    ,ie,np1,a,a,a));
    };

    int nerr;
    auto policy = m_policy;
    Kokkos::parallel_reduce(policy, toplevel, nerr);
    if (nerr > 0) {
      const int nt[] = {nm1, n0, np1};
      const char* ntname[] = {"nm1", "n0", "np1"};
      for (int i = 0; i < 3; ++i)
        e.m_state.check_print_abort_on_bad_elems(std::string("DIRK Newton loop ") + ntname[i], nt[i]);
    }
  }

  template <typename Fn>
  KOKKOS_INLINE_FUNCTION
  static void loop_ki (const KernelVariables& kv, const int klim,
                       const int ilim, const Fn& g) {
    using Kokkos::parallel_for;
    using Kokkos::TeamThreadRange;
    using Kokkos::ThreadVectorRange;

    if (OnGpu<ExecSpace>::value) {
      const auto tr = TeamThreadRange  (kv.team, klim);
      const auto vr = ThreadVectorRange(kv.team, ilim);
      const auto f = [&] (const int k) {
        const auto h = [&] (const int i) { g(k,i); };
        parallel_for(vr, h);
      };
      parallel_for(tr, f);
    } else if (kv.team.team_size() == 1) {
      for (int k = 0; k < klim; ++k)
        for (int i = 0; i < ilim; ++i)
          g(k,i);
    } else {
      const auto tr = TeamThreadRange  (kv.team, klim);
      const auto f = [&] (const int k) {
        for (int i = 0; i < ilim; ++i)
          g(k,i);
      };
      parallel_for(tr, f);
    }
  }

  // Format of rest of Hxx -> DIRK Newton iteration format.
  template <typename View>
  KOKKOS_INLINE_FUNCTION
  static void transpose (const KernelVariables& kv, const int nlev,
                         const View& src, const WorkSlot& dst,
                         typename std::enable_if<View::rank == 3>::type* = 0) {
    assert(src.extent_int(2)*packn >= nlev);
    assert(src.extent_int(0) == NP && src.extent_int(1) == NP);
    const auto f = [&] (const int k) {
      const auto
      pk = k / packn,
      sk = k % packn;
      const auto g = [&] (const int i) {
        const auto gk0 = packn*i;
        for (int s = 0; s < packn; ++s) {
          const auto
          gk = gk0 + s,
          gi = gk / NP,
          gj = gk % NP;
          if ((int)scaln % (int)packn != 0 && // try to compile out this conditional when possible
              gk >= scaln) break;
          dst(k,i)[s] = src(gi,gj,pk)[sk];
        }
      };
      const int n = npack;
      const auto p = Kokkos::ThreadVectorRange(kv.team, n);
      Kokkos::parallel_for(p, g);
    };    
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, nlev), f);
  }

  // DIRK Newton iteration format -> format of rest of Hxx.
  template <typename View>
  KOKKOS_INLINE_FUNCTION
  static void transpose (const KernelVariables& kv, const int nlev,
                         const WorkSlot& src, const View& dst,
                         typename std::enable_if<View::rank == 3>::type* = 0) {
    assert(dst.extent_int(2)*packn >= nlev);
    assert(dst.extent_int(0) == NP && dst.extent_int(1) == NP);
    const auto f = [&] (const int idx) {
      const auto
      gi = idx / NP,
      gj = idx % NP,
      pi = idx / packn,
      si = idx % packn;
      const auto g = [&] (const int pk) {
        const auto k0 = pk*packn;
        // If there is a remainder at the end, we nonetheless transfer these
        // unused data to avoid a conditional and runtime loop limit.
        for (int sk = 0; sk < packn; ++sk)
          dst(gi,gj,pk)[sk] = src(k0+sk,pi)[si];
      };
      const auto p = Kokkos::ThreadVectorRange(kv.team, dst.extent_int(2));
      Kokkos::parallel_for(p, g);
    };    
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP*NP), f);
  }

  // Views of the state at one time level of one element, in Hxx format, as returned by
  // the state_at callable passed to calc_w0_phi0. The type of each view is independent, so
  // that, e.g., the state may be stored in a different scalar type than the work slots.
  template <typename Vw, typename Vphi, typename Vvth, typename Vdp, typename Vv>
  struct StateViews {
    Vw w_i;         // interfaces
    Vphi phinh_i;   // interfaces
    Vvth vtheta_dp; // midpoints
    Vdp dp3d;       // midpoints
    Vv v;           // horizontal velocity
  };

  template <typename Vw, typename Vphi, typename Vvth, typename Vdp, typename Vv>
  KOKKOS_INLINE_FUNCTION
  static StateViews<Vw,Vphi,Vvth,Vdp,Vv>
  make_state_views (const Vw& w_i, const Vphi& phinh_i, const Vvth& vtheta_dp,
                    const Vdp& dp3d, const Vv& v) {
    return StateViews<Vw,Vphi,Vvth,Vdp,Vv>{w_i, phinh_i, vtheta_dp, dp3d, v};
  }

  // The equations that the Newton iteration solves for w_np1 are, in each column,
  //   r(w) = w - (w0 + g*dt2*(dpnh_dp(vtheta_dp,dp3d,dphi) - 1)) = 0,
  //   dphi = diff(phi0) + dt2*g*diff(w),     phi_np1 = phi0 + dt2*g*w,
  // where w0 and phi0 are the 'known' parts: the np1 state before the stage, plus the terms coming
  // from the states at n0 and nm1 (if alphadt_n0 and alphadt_nm1 are not 0), and, for phi0, the
  // vertical velocity induced by the topography. This computes, in DIRK format and for all the
  // columns of one element, w_n0 (=w0) and phi_n0 (=phi0), and loads the np1 vtheta_dp and dp3d
  // (and w_np1 with the np1 w). The state of each time level comes from state_at(nt), which
  // returns a StateViews (see make_state_views): this way, the state can come from anywhere (not
  // only from the Elements). The work slots are all scratch (they are overwritten), and wrk, pnh,
  // dpnh_dp_i, gwh_i are only used as such. Returns wmax = max(1, max|w(np1)|), computed before
  // adding the n0/nm1 terms. Must be called by the whole team. It does not end with a team barrier
  // (the caller adds one before using the outputs, if needed).
  template <typename StateAt, typename GPhis, typename Hybi>
  KOKKOS_INLINE_FUNCTION
  static ST calc_w0_phi0 (const KernelVariables& kv, const HybridVCoord& hvcoord,
                          const EquationOfState<Constants>& eos, const Real dt2,
                          const int nm1, const Real alphadt_nm1,
                          const int n0, const Real alphadt_n0, const int np1,
                          const StateAt& state_at, const GPhis& gradphis, const Hybi& hybi,
                          const WorkSlot& phi_n0, const WorkSlot& phi_np1, const WorkSlot& dphi,
                          const WorkSlot& w_n0, const WorkSlot& w_np1, const WorkSlot& dpnh_dp_i,
                          const WorkSlot& gwh_i, const WorkSlot& vtheta_dp, const WorkSlot& dp3d,
                          const WorkSlot& pnh, const WorkSlot& wrk, int& nerr) {
    const int nlev = num_phys_lev;
    const int nvec = npack;
    const auto grav = eos.m_constants.g();

    const auto transpose4 = [&] (const auto& s, const bool transpose_phi_np1 = true) {
      transpose(kv, nlev+1, s.w_i      , w_np1    );
      transpose(kv, nlev,   s.vtheta_dp, vtheta_dp);
      transpose(kv, nlev,   s.dp3d     , dp3d     );
      if ( ! transpose_phi_np1) return;
      transpose(kv, nlev+1, s.phinh_i  , phi_np1  );
    };

    // Add to w_n0, phi_n0 the contribution of the state at time level nt (n0 or nm1), weighted by dt3.
    const auto accum_time_level = [&] (const Real dt3, const int nt) {
      // Computing these transposes is inefficient, but doing so lets us use
      // the same pnh_and_exner_from_eos as we use elsewhere. This function is
      // called only in ~1 out of 5 DIRK stage calls, and only outside of the
      // Newton iteration. (Also, tbc, transpose and pnh_and_exner_from_eos
      // are parallel efficient.)
      const auto s = state_at(nt);
      transpose4(s);
      calc_gwphis(kv, s.dp3d, s.v, gradphis, hybi, gwh_i);
      kv.team_barrier();
      loop_ki(kv, nlev, nvec, [&] (int k, int i) {
        dphi(k,i) = phi_np1(k+1,i) - phi_np1(k,i);
      });
      kv.team_barrier();
      const bool ok = pnh_and_exner_from_eos(kv, hvcoord, eos, vtheta_dp, dp3d,
                                             dphi, pnh, wrk, dpnh_dp_i);
      if ( ! ok) nerr = 1;
      kv.team_barrier();
      loop_ki(kv, nlev, nvec, [&] (const int k, const int i) {
        w_n0(k,i) += dt3*grav*(dpnh_dp_i(k,i) - 1);
      });
      loop_ki(kv, nlev, nvec, [&] (int k, int i) {
        phi_n0(k,i) = phi_n0(k,i) + dt3*grav*w_np1(k,i) - dt3*gwh_i(k,i);
      });
    };

    // Compute w_n0, phi_n0.
    const auto s_np1 = state_at(np1);
    transpose(kv, nlev+1, s_np1.phinh_i, phi_n0);
    transpose(kv, nlev+1, s_np1.w_i    , w_n0  );
    kv.team_barrier();
    // wmax is computed before optional updates to w_n0.
    const auto wmax = calc_wmax(kv, nlev+1, nvec, w_n0);
    // Computed only in some cases.
    if (alphadt_n0 != 0) {
      accum_time_level(alphadt_n0, n0);
      kv.team_barrier();
    }
    if (alphadt_nm1 != 0) {
      assert(nm1 >= 0);
      accum_time_level(alphadt_nm1, nm1);
      kv.team_barrier();
    }
    // Always computed.
    transpose4(s_np1, false);
    calc_gwphis(kv, s_np1.dp3d, s_np1.v, gradphis, hybi, gwh_i);
    kv.team_barrier();
    loop_ki(kv, nlev, nvec, [&] (int k, int i) { phi_n0(k,i) -= dt2*gwh_i(k,i); });
    return wmax;
  }

  // x = -r(w), with r the residual of the equations solved by the Newton iteration (see
  // calc_w0_phi0), given dpnh_dp_i (see pnh_and_exner_from_eos) evaluated at w.
  template <typename GravT>
  KOKKOS_INLINE_FUNCTION
  static void calc_neg_residual (const KernelVariables& kv, const int nlev, const int nvec,
                                 const GravT& grav, const Real dt2,
                                 const WorkSlot& w_np1, const WorkSlot& w_n0,
                                 const WorkSlot& dpnh_dp_i, const LinearSystemSlot& x) {
    loop_ki(kv, nlev, nvec, [&] (const int k, const int i) {
      x(k,i) = -(w_np1(k,i) - (w_n0(k,i) + grav*dt2*(dpnh_dp_i(k,i) - 1))); // -residual
    });
  }

  // Compute a vertical velocity induced by surface topography:
  //     wh_i = ubar grad phis
  template <typename R, typename Rv, typename Rgphis, typename Rhybi, typename W>
  KOKKOS_INLINE_FUNCTION
  static void calc_gwphis (
    const KernelVariables& kv,
    // All in arrays are in Hxx format.
    const R& dp3d, const Rv& v, const Rgphis& gradphis, const Rhybi& hybi,
    // Out array is in DIRK format. gwh_i(nlevp,:) is not written since it is
    // not used.
    const W& gwh_i,
    const int nlev = NUM_PHYSICAL_LEV)
  {
    using Kokkos::parallel_for;
    using Kokkos::TeamThreadRange;
    using Kokkos::ThreadVectorRange;

    const int n = npack;
    const auto pv = ThreadVectorRange(kv.team, n);

    const auto f1 = [&] (const int) {
      const auto k0 = [&] (const int i) { gwh_i(0,i) = 0; };
      parallel_for(pv, k0);
    };
    parallel_for(TeamThreadRange(kv.team, 1), f1);

    const auto f2 = [&] (const int km1) {
      const auto
      k = km1 + 1,
      pk = k / packn,
      sk = k % packn,
      pkm1 = (k-1) / packn,
      skm1 = (k-1) % packn;
      const auto g = [&] (const int i) {
        PT dp3dk, dp3dkm1, v1k, v2k, v1km1, v2km1, gphis1, gphis2;
        for (int s = 0; s < packn; ++s) {
          const auto
          idx = packn*i + s,
          gi = idx / NP,
          gj = idx % NP;
          if ((int)scaln % (int)packn != 0 && idx >= scaln) break;
          dp3dkm1[s] = dp3d(gi,gj,pkm1)[skm1];
          dp3dk  [s] = dp3d(gi,gj,pk  )[sk];
          v1km1  [s] = v (0,gi,gj,pkm1)[skm1];
          v2km1  [s] = v (1,gi,gj,pkm1)[skm1];
          v1k    [s] = v (0,gi,gj,pk  )[sk];
          v2k    [s] = v (1,gi,gj,pk  )[sk];
          gphis1 [s] = gradphis(0,gi,gj);
          gphis2 [s] = gradphis(1,gi,gj);
        }
        const auto den = dp3dkm1 + dp3dk;
        const auto v1_i = (dp3dk*v1k + dp3dkm1*v1km1) / den;
        const auto v2_i = (dp3dk*v2k + dp3dkm1*v2km1) / den;
        gwh_i(k,i) = (v1_i*gphis1 + v2_i*gphis2)*hybi(k);
      };
      parallel_for(pv, g);
    };
    parallel_for(TeamThreadRange(kv.team, nlev-1), f2);
  }

  template <typename R, typename W, typename Wi>
  KOKKOS_INLINE_FUNCTION static
  bool pnh_and_exner_from_eos (
    const KernelVariables& kv, const HybridVCoord& hvcoord,
    const EquationOfState<Constants>& eos,
    // All arrays are in DIRK format.
    const R& vtheta_dp, const R& dp3d, const R& dphi,
    // exner is workspace. dpnh_dp_i(nlevp,:) is not computed.
    const W& pnh, const W& exner, const Wi& dpnh_dp_i,
    const int nlev = NUM_PHYSICAL_LEV)
  {
    using Kokkos::parallel_for;

    const int n = npack, ns = packn;
    const auto pv = Kokkos::ThreadVectorRange(kv.team, n);
    bool ok = true;

    // Compute pnh(1:nlev,:). pnh(nlevp,:) is not needed.
    const auto f1 = [&] (const int k) {
      const auto g = [&] (const int i) {
        for (int s = 0; s < ns; ++s) {
          if (vtheta_dp(k,i)[s] < 0 || dphi(k,i)[s] > 0)
            ok = false;
        }

        eos.compute_pnh_and_exner(vtheta_dp(k,i), dphi(k,i), pnh(k,i), exner(k,i));
      };
      parallel_for(pv, g);
    };
    parallel_for(Kokkos::TeamThreadRange(kv.team, nlev), f1);

    // Compute dpnh_dp_i(1:nlev,:), d(pnh)/d(pi) at interfaces. Use one-sided
    // differences at boundaries. Do not compute dpnh_dp_i(nlevp,:).
    kv.team_barrier(); // wait for pnh
    const auto f2 = [&] (const int) {
      const auto k0 = [&] (const int i) {
        const auto pnh_i_0 = hvcoord.hybrid_ai0*hvcoord.ps0; // hydrostatic ptop
        dpnh_dp_i(0,i) = 2*(pnh(0,i) - pnh_i_0)/dp3d(0,i);
      };
      parallel_for(pv, k0);
    };
    parallel_for(Kokkos::TeamThreadRange(kv.team, 1), f2);
    const auto f3 = [&] (const int km1) {
      // The following is morally a const var, but there are issues with
      // gnu and std=c++14. The macro ConstExceptGnu is defined in share/cxx/Config.hpp.
      ConstExceptGnu auto k = km1 + 1;
      const auto kr = [&] (const int i) {
        dpnh_dp_i(k,i) = ((pnh(k,i) - pnh(k-1,i))/
                          ((dp3d(k-1,i) + dp3d(k,i))/2));
      };
      parallel_for(pv, kr);
    };
    parallel_for(Kokkos::TeamThreadRange(kv.team, nlev-1), f3);

    return ok;
  }

  // Suboptimal impl of the initial guess that uses the policy native to the
  // DIRK Newton iteration. It is suboptimal because there are two scans, and
  // both the memory layout and the parallelization are suboptimal for these.
  // run_initial_guess, above, runs a separate kernel with optimal layout and
  // parallelization.
  template <typename Rphis, typename R, typename W>
  KOKKOS_INLINE_FUNCTION static void
  phi_from_eos (const KernelVariables& kv, const EquationOfState<Constants>& eos,
                const int nlev, const int nvec,
                const HybridVCoord& hvcoord, const Rphis& phis, const R& vtheta_dp, const R& dp,
                // phi_i on output
                const W& wrk)
  {
    // Scan to compute pressure.
    loop_ki(kv, 1, nvec, [&] (int, int i) {
      wrk(0,i) = hvcoord.hybrid_ai0*hvcoord.ps0;
      for (int k = 0; k < nlev; ++k) {
        wrk(k+1,i) = wrk(k,i) + dp(k,i);      // p_i at interface k+1
        wrk(k,i) = (wrk(k+1,i) + wrk(k,i))/2; // p_m at midpoint  k
      }
    });
    kv.team_barrier();
    // Do most of the flops.
    loop_ki(kv, nlev, nvec, [&] (int k, int i) {
      wrk(k,i) = eos.compute_dphi(vtheta_dp(k,i), wrk(k,i)); // dphi
    });
    kv.team_barrier();
    // Scan to compute phi_i.
    loop_ki(kv, 1, nvec, [&] (int, int i) {
      set_phis(i, phis, wrk);
      for (int k = nlev-1; k >= 0; --k)
        wrk(k,i) = wrk(k+1,i) + wrk(k,i); // phi_i below + dphi
    });
  }

  template <typename Rphis, typename W>
  KOKKOS_INLINE_FUNCTION static void
  set_phis (const int i, const Rphis& phis, const W& phi_i) {
    for (int s = 0; s < packn; ++s) {
      const int idx = i*packn + s, gi = idx / NP, gj = idx % NP;
      if ((int)scaln % (int)packn != 0 && idx >= scaln) break;
      phi_i(num_phys_lev,i)[s] = phis(gi,gj);
    }    
  }

  KOKKOS_INLINE_FUNCTION
  static ST calc_wmax (const KernelVariables& kv, const int nlev, const int nvec,
                       const WorkSlot& w) {
    using Kokkos::parallel_reduce;
    using Kokkos::TeamThreadRange;
    using Kokkos::ThreadVectorRange;

    const auto f = [&] (int k, ST& maxval) {
      const auto g = [&] (int i, ST& lmaxval) {
        const auto v = w(k,i);
        for (int s = 0; s < packn; ++s) {
          if ((int)scaln % (int)packn != 0 && i*packn + s >= scaln) break;
          lmaxval = max(lmaxval, std::abs(v[s]));
        }
      };
      ST lmaxval = 0;
      const auto vr = ThreadVectorRange(kv.team, nvec);
      parallel_reduce(vr, g, Kokkos::Max<ST>(lmaxval));
      maxval = max(maxval, lmaxval); // benign write race
    };
    ST wmax = 0;
    const auto tr = TeamThreadRange(kv.team, nlev);
    parallel_reduce(tr, f, Kokkos::Max<ST>(wmax));
    return max(ST(1), wmax);
  }

  KOKKOS_INLINE_FUNCTION
  static bool exit_on_step (const KernelVariables& kv, const int nlev, const int nvec,
                            const ST& wmax, const Real& deltatol,
                            const LinearSystemSlot& x, ST& deltaerr) {
    using Kokkos::parallel_reduce;
    using Kokkos::TeamThreadRange;
    using Kokkos::ThreadVectorRange;

    // deltaerr = maxval(abs(x) / wmax
    const auto f = [&] (int k, ST& maxval) {
      const auto g = [&] (int i, ST& lmaxval) {
        const auto v = x(k,i);
        for (int s = 0; s < packn; ++s) {
          if ((int)scaln % (int)packn != 0 && i*packn + s >= scaln) break;
          lmaxval = max(lmaxval, std::abs(v[s]));
        }
      };
      ST lmaxval;
      const auto vr = ThreadVectorRange(kv.team, nvec);
      parallel_reduce(vr, g, Kokkos::Max<ST>(lmaxval));
      maxval = max(maxval, lmaxval); // benign write race
    };
    const auto tr = TeamThreadRange(kv.team, nlev);
    parallel_reduce(tr, f, Kokkos::Max<ST>(deltaerr));
    return deltaerr/wmax < deltatol;
  }

  /* Compute Jacobian of F(phi) = sum(dphi) + const + (dt*g)^2 *(1-dp/dpi)
     column wise with respect to phi. Form the tridiagonal analytical Jacobian J
     to solve J * x = -f.

     This code will need to change when the equation of state is changed.
  */
  template <typename R, typename W>
  KOKKOS_INLINE_FUNCTION
  static void calc_jacobian (const KernelVariables& kv, const Constants& constants, const Real& dt2,
                             // All arrays are in DIRK format.
                             const R& dp3d, const R& dphi, const R& pnh,
                             const W& dl, const W& d, const W& du,
                             const int nlev = NUM_PHYSICAL_LEV) {
    using Kokkos::parallel_for;

    const int n = npack;
    const auto pv = Kokkos::ThreadVectorRange(kv.team, n);
    const auto pt1 = Kokkos::TeamThreadRange(kv.team, 1);

    const auto g = constants.g();
    const auto kappa = constants.kappa();
    // If the input is a Sacado expression, eval will evaluate and return the resulting Fad.
    const auto a = eval(square(dt2*g) / (1-kappa));
    // std::cout << "typeid(a)=" << typeid(a).name() << ", a=" << a << "\n";

    const auto f1 = [&] (const int) {
      const auto ks = [&] (const int i) { // first Jacobian row
        const int k = 0;
        const auto b = a/dp3d(k,i);
        du(k,i) = 2*b*(pnh(k,i)/dphi(k,i));
        d (k,i) = 1 - du(k,i);
      };
      parallel_for(pv, ks);
    };
    parallel_for(pt1, f1);
    const auto f2 = [&] (const int km1) {
      // The following is morally a const var, but there are issues with
      // gnu and std=c++14. The macro ConstExceptGnu is defined in share/cxx/Config.hpp.
      ConstExceptGnu  auto k = km1 + 1;
      const auto kmid = [&] (const int i) { // middle Jacobian rows
        const auto b = 2*a/(dp3d(k-1,i) + dp3d(k,i));
        dl(k,i) = b*(pnh(k-1,i)/dphi(k-1,i));
        du(k,i) = b*(pnh(k  ,i)/dphi(k  ,i));
        // In all rows k,
        //     dl <= 0, du <= 0,
        // and thus
        //     d = 1 + |dl| + |du| > |dl| + |du|,
        // making this Jacobian matrix strictly diagonally dominant. Thus, we
        // need not pivot when factorizing the matrix.
        d (k,i) = 1 - dl(k,i) - du(k,i);
      };
      parallel_for(pv, kmid);
    };
    parallel_for(Kokkos::TeamThreadRange(kv.team, nlev-2), f2);
    const auto f3 = [&] (const int) {
      const auto ke = [&] (const int i) { // last Jacobian row
        const int k = nlev-1;
        const auto b = 2*a/(dp3d(k-1,i) + dp3d(k,i));
        dl(k,i) = b*(pnh(k-1,i)/dphi(k-1,i));
        d (k,i) = 1 - dl(k,i) - b*(pnh(k,i)/dphi(k,i));        
      };
      parallel_for(pv, ke);
    };
    parallel_for(pt1, f3);
  }

  template <typename W>
  KOKKOS_INLINE_FUNCTION
  static void solve (const KernelVariables& kv,
                     const W& dl, const W& d, const W& du, const W& x) {
    assert(d.extent_int(0) == num_phys_lev);
    if (OnGpu<ExecSpace>::value)
      scream::tridiag::cr(kv.team, dl, d, du, x);
    else {
      const auto f = [&] () { scream::tridiag::thomas(dl, d, du, x); };
      Kokkos::single(Kokkos::PerTeam(kv.team), f);
    }
  }

  // Solve J^T x = b, where J is the tridiagonal matrix described by (dl, d, du) (as computed by
  // calc_jacobian, see above) and x holds b on input. J^T has sub-diagonal du(k-1) (row k) and
  // super-diagonal dl(k+1) (row k), so we build the (dl, du) of J^T (in dl and du) and solve as usual.
  // Like solve, this overwrites dl, d, du. scratch is a linear-system slot (as big as dl). Must be called by the whole
  // team, and ends with the same synchronization as solve (the caller adds the barrier before using x).
  template <typename W>
  KOKKOS_INLINE_FUNCTION
  static void solve_transpose (const KernelVariables& kv,
                               const W& dl, const W& d, const W& du, const W& scratch, const W& x,
                               const bool bfb_solver = default_bfb_solver) {
    const int nlev = num_phys_lev, nvec = npack;
    loop_ki(kv, nlev, nvec, [&] (const int k, const int i) {
      if (k > 0) scratch(k,i) = du(k-1,i); else scratch(k,i) = 0;
    });
    kv.team_barrier();
    loop_ki(kv, nlev, nvec, [&] (const int k, const int i) {
      if (k < nlev-1) du(k,i) = dl(k+1,i); else du(k,i) = 0;
    });
    kv.team_barrier();
    loop_ki(kv, nlev, nvec, [&] (const int k, const int i) { dl(k,i) = scratch(k,i); });
    kv.team_barrier();
    if (bfb_solver) solvebfb(kv, dl, d, du, x); else solve(kv, dl, d, du, x);
  }

  template <typename W>
  KOKKOS_INLINE_FUNCTION
  static void solvebfb (const KernelVariables& kv,
                        const W& dl, const W& d, const W& du, const W& x) {
    assert(d.extent_int(0) == num_phys_lev);
    scream::tridiag::bfb(kv.team, dl, d, du, x);
  }

  // Determine a step length 0 < alpha <= 1.
  template<typename GravST>
  KOKKOS_INLINE_FUNCTION static void
  calc_step_size (const KernelVariables& kv, const int nlev, const int nvec,
                  const GravST& grav, const Real& dt2,
                  const WorkSlot& dphi_n0, const WorkSlot& w_np1, const LinearSystemSlot& x,
                  // On input, wrk(0,i)[s] is 1 if the step length should be
                  // smaller. On output, alpha is in wrk(2,:).
                  const WorkSlot& wrk) {
    using Kokkos::parallel_reduce;
    using Kokkos::parallel_for;
    using Kokkos::TeamThreadRange;
    using Kokkos::ThreadVectorRange;
    // Set all alpha_k to 1 except row 0. Include row nlev for use as a flag.
    loop_ki(kv, nlev, nvec, [&] (int k, int i) { wrk(k+1,i) = 1; });
    kv.team_barrier();
    loop_ki(kv, nlev, nvec, [&] (int k, int i) {
      for (int s = 0; s < packn; ++s) {
        if ((int)scaln % (int)packn != 0 && i*packn + s >= scaln) break;
        if (wrk(0,i)[s] == 0) {
          // Indicate this column is already good.
          wrk(nlev,i)[s] = 0;
          continue;
        }
        ST dx, dw;
        if (k < nlev-1) {
          dx =     x(k+1,i)[s] -     x(k,i)[s];
          dw = w_np1(k+1,i)[s] - w_np1(k,i)[s];
        } else {
          dx = -    x(k,i)[s];
          dw = -w_np1(k,i)[s];          
        }
        if (dx != 0) {
          // Step length at which dphi(k,i)[s] would = 0.
          const ST alpha = -(dphi_n0(k,i)[s] + dt2*grav*dw)/(dt2*grav*dx);
          // A negative step is irrelevant.
          if (alpha >= 0) wrk(k,i)[s] = alpha;
        }
      }
    });
    kv.team_barrier();
    // Find minimum alpha in each column. This calculation is performed rarely,
    // so we can use the following suboptimal reduction arising from the
    // Newton-loop team policy without essentially any performance loss. The
    // code itself is fine, but the current policy has too many threads and too
    // few vector lanes on GPU compared with the optimal policy.
    const auto f = [&] (int idx) {
      const int i = idx / packn, s = idx % packn;
      if (wrk(nlev,i)[s] == 0) return;
      const auto g = [&] (int k, ST& lalpha) { lalpha = min(lalpha, wrk(k,i)[s]); };
      ST alpha = 0;
      const auto vr = ThreadVectorRange(kv.team, nlev);
      parallel_reduce(vr, g, Kokkos::Min<ST>(alpha));
      // Step halfway to the distance at which at least one dphi is 0.
      wrk(2,i)[s] = min(ST(1), alpha)/2;
    };
    const auto tr = TeamThreadRange(kv.team, static_cast<int>(scaln));
    parallel_for(tr, f);
  }

  // Determine whether any dphi > threshold. Set the entry to threshold.
  KOKKOS_INLINE_FUNCTION static void
  calc_whether_gt_and_set (const KernelVariables& kv, const int nlev, const int nvec,
                           const Real threshold,
                           // dphi > threshold?
                           const WorkSlot& dphi,
                           // On output, wrk(0,:) contains 0 for no, 1 for yes;
                           // if any is yes, then wrk(1,0) is 1, else 0.
                           const WorkSlot& wrk) {
    loop_ki(kv, 2, nvec, [&] (int k, int i) { wrk(k,i) = 0; });
    kv.team_barrier();
    loop_ki(kv, nlev, nvec, [&] (int k, int i) {
      for (int s = 0; s < packn; ++s) {
        if ((int)scaln % (int)packn != 0 && i*packn + s >= scaln) break;
        if (dphi(k,i)[s] > threshold) {
          dphi(k,i)[s] = threshold;
          wrk(0,i)[s] = 1; // benign write race
          wrk(1,0)[0] = 1; // benign write race
        }
      }
    });
  }

  // Determine whether any dphi >= threshold.
  KOKKOS_INLINE_FUNCTION static void
  calc_whether_ge (const KernelVariables& kv, const int nlev, const int nvec,
                   const Real threshold,
                   // dphi >= threshold?
                   const WorkSlot& dphi,
                   // On output, wrk(0,:) contains 0 for no, 1 for yes;
                   // if any is yes, then wrk(1,0) is 1, else 0.
                   const WorkSlot& wrk) {
    loop_ki(kv, 2, nvec, [&] (int k, int i) { wrk(k,i) = 0; });
    kv.team_barrier();
    loop_ki(kv, nlev, nvec, [&] (int k, int i) {
      for (int s = 0; s < packn; ++s) {
        if ((int)scaln % (int)packn != 0 && i*packn + s >= scaln) break;
        if (dphi(k,i)[s] >= threshold) {
          wrk(0,i)[s] = 1; // benign write race
          wrk(1,0)[0] = 1; // benign write race
        }
      }
    });
  }

  KOKKOS_INLINE_FUNCTION static bool
  scan_dphi (const KernelVariables& kv, const int nlev, const int nvec,
             // On input, wrk(0,:) contains 0 for no, 1 for yes
             const WorkSlot& wrk, const WorkSlot& dphi, const WorkSlot& phi_i) {
    // imex_mod scans all cols if even one is bad. So do that here. The scan is
    // suboptimal, but it's much better to do it like this than have separate
    // kernels just b/c of a safety scan. All scans except the initial guess,
    // which we do indeed handle in a separate kernel, are triggered only very
    // occasionally.
    loop_ki(kv, 1, nvec, [&] (int, int i) {
      for (int k = nlev-1; k >= 0; --k)
        phi_i(k,i) = phi_i(k+1,i) - dphi(k,i);
    });
    return true;
  }

  // Debug routine, not for GPU.
  template <typename R>
  static bool good (const R& a) {
    bool good = true;
    for (int k = 0; k < num_phys_lev; ++k)
      for (int i = 0; i < a.extent_int(1); ++i)
        for (int s = 0; s < packn; ++s) {
          if (i*packn + s >= scaln) continue;
          const auto v = a(k,i)[s];
          if (std::isnan(v) || std::isinf(v)) good = false;
        }
    return good;
  }
};

using DirkFunctorImpl = DirkFunctorImplST<ScalarValue>;

} // namespace Homme

#endif // HOMMEXX_DIRK_FUNCTOR_IMPL_HPP
