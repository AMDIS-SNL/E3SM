/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_REMAP_FUNCTOR_HPP
#define HOMMEXX_REMAP_FUNCTOR_HPP

#include <memory>
#include <type_traits>

#include "ErrorDefs.hpp"

#include "Elements.hpp"
#include "Tracers.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "ColumnOps.hpp"
#include "Types.hpp"
#include "utilities/LoopsUtils.hpp"
#include "utilities/MathUtils.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "RemapStateProvider.hpp"
#include "../../theta-l_kokkos/cxx/StateSnapshot.hpp"

#include "profiling.hpp"

namespace Homme {
namespace Remap {

// The RemapStateAndThicknessProvider provides states and src/tgt thicknesses
// to the RemapFunctor. Its implementation differ depending on whether rsplit
// is zero or not. In case it is not zero, it must provide states to remap.
// For this task, it relies on the RemapStateProvider class, which is implemented
// inside each model (preqx and theta), since the state structure is different
// across models.
template <bool nonzero_rsplit, typename ST>
struct RemapStateAndThicknessProvider;

template <typename ST>
struct RemapStateAndThicknessProvider<false, ST> {
  using PT = PackType<ST>;

  ExecViewManaged<PT * [NP][NP][NUM_LEV]> m_src_layer_thickness;
  ExecViewManaged<PT * [NP][NP][NUM_LEV]> m_tgt_layer_thickness;

  ExecViewUnmanaged<const PT * [NP][NP][NUM_LEV_P]> m_eta_dot_dpdn;

  explicit
  RemapStateAndThicknessProvider (const ElementsST<ST>& elements)
    : m_src_layer_thickness("Source layer thickness", elements.num_elems())
    , m_tgt_layer_thickness("Target layer thickness", elements.num_elems())
    , m_eta_dot_dpdn(elements.m_derived.m_eta_dot_dpdn)
  {}

  KOKKOS_INLINE_FUNCTION
  int num_states_remap () const { return 0; }

  int requested_buffer_size () const {
    return 0;
  }

  void init_buffers(const FunctorsBuffersManager& /* fbm */) {
    // Do nothing
  }

  KOKKOS_INLINE_FUNCTION
  bool is_intrinsic_state (const int /* istate */) const {
    // We should never reach this part
    assert(false);
    return false;
  }

  void preprocess_states (const int /* np1 */) const {
    // Do nothing
  }

  void postprocess_states (const int /* np1 */) const {
    // Do nothing
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<PT[NP][NP][NUM_LEV]>
  get_state(const KernelVariables &/* kv */, int /*np1*/, int /*var*/) const {
    Kokkos::abort("Error! Asked for state in an rsplit=0 remap.\n");

    // Dummy, to silence compiler warnings
    return ExecViewUnmanaged<PT[NP][NP][NUM_LEV]>();
  } 

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<PT[NP][NP][NUM_LEV]>
  get_source_thickness(const int ie, const int /* np1 */) const {
    return Homme::subview(m_src_layer_thickness, ie);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const PT[NP][NP][NUM_LEV]>
  compute_source_thickness(
      KernelVariables &kv, const int &np1, const Real &dt,
      ExecViewUnmanaged<const PT[NP][NP][NUM_LEV]> tgt_layer_thickness)
      const {

    auto src_layer_thickness = get_source_thickness(kv.ie, np1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;

      auto eta_dot_dpdn     = Homme::subview(m_eta_dot_dpdn,kv.ie,igp,jgp);
      auto src_thickness_ij = Homme::subview(src_layer_thickness,igp,jgp);

      ColumnOps::compute_midpoint_delta<CombineMode::Scale>(kv,eta_dot_dpdn,src_thickness_ij,dt);

      auto tgt_thickness_ij = Homme::subview(tgt_layer_thickness,igp,jgp);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int ilev) {
        src_thickness_ij(ilev) += tgt_thickness_ij(ilev);
      });
    });
    kv.team_barrier();
    return src_layer_thickness;
  }
};

template <typename ST>
struct RemapStateAndThicknessProvider<true, ST> {
  using PT = PackType<ST>;

  ExecViewUnmanaged<PT *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]> m_dp3d;
  ExecViewManaged<PT * [NP][NP][NUM_LEV]> m_tgt_layer_thickness;
  RemapStateProvider<ST>          m_state_provider;
  int m_np1;

  struct TagPreProcess  {};
  struct TagPostProcess {};
  Kokkos::TeamPolicy<ExecSpace,TagPreProcess>  m_policy_pre;
  Kokkos::TeamPolicy<ExecSpace,TagPostProcess> m_policy_post;

  TeamUtils<ExecSpace> m_tu;

  explicit
  RemapStateAndThicknessProvider (const ElementsST<ST>& elements)
    : m_dp3d (elements.m_state.m_dp3d)
    , m_tgt_layer_thickness("Target layer thickness", elements.num_elems())
    , m_state_provider (elements)
    , m_policy_pre(Homme::get_default_team_policy<ExecSpace,TagPreProcess>(1))
    , m_policy_post(Homme::get_default_team_policy<ExecSpace,TagPostProcess>(1))
    , m_tu(m_policy_pre)
  {
    // We do not init them in the initialization list, since they require info
    // from the state provider. Although, as they are declared in the class,
    // the state provider is inited before the policies, it is safer to query
    // the state provider inside the ctor body (in case someone rearranges the
    // members of the class.

    const int iters_pre = m_state_provider.num_states_preprocess();
    const int iters_post = m_state_provider.num_states_postprocess();
    const int num_elems = elements.m_state.num_elems();
    m_policy_pre = Homme::get_default_team_policy<ExecSpace,TagPreProcess>(iters_pre*num_elems);
    m_policy_post = Homme::get_default_team_policy<ExecSpace,TagPostProcess>(iters_post*num_elems);

    if (iters_pre*num_elems > 0) {
      m_tu = TeamUtils<ExecSpace>(m_policy_pre);
    }
  }

  KOKKOS_INLINE_FUNCTION
  int num_states_remap () const { return m_state_provider.num_states_remap(); }

  int requested_buffer_size () const {
    return m_state_provider.requested_buffer_size(m_tu.get_num_ws_slots());
  }

  void init_buffers(const FunctorsBuffersManager& fbm) {
    m_state_provider.init_buffers(fbm, m_tu.get_num_ws_slots());
  }

  KOKKOS_INLINE_FUNCTION
  bool is_intrinsic_state (const int istate) const {
    return m_state_provider.is_intrinsic_state(istate);
  }

  void preprocess_states (const int np1) {
    if (m_state_provider.num_states_preprocess()>0) {
      m_np1 = np1;
      Kokkos::parallel_for("Pre-process states",m_policy_pre,*this);
      Kokkos::fence();
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagPreProcess&,
                   const TeamMember& team) const {
    KernelVariables kv(team, m_tu);
    const int istate = kv.ie % m_state_provider.num_states_preprocess();
    kv.ie /= m_state_provider.num_states_preprocess();
    auto dp = get_source_thickness(kv.ie,m_np1);
    m_state_provider.preprocess_state(kv,istate,m_np1,dp);
  }

  void postprocess_states (const int np1) {
    if (m_state_provider.num_states_postprocess()>0) {
      m_np1 = np1;
      Kokkos::parallel_for("Post-process states",m_policy_post,*this);
      Kokkos::fence();
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagPostProcess&,
                   const TeamMember& team) const {
    KernelVariables kv(team, m_tu);
    const int istate = kv.ie % m_state_provider.num_states_preprocess();
    kv.ie /= m_state_provider.num_states_preprocess();
    auto dp = Homme::subview(m_tgt_layer_thickness,kv.ie);
    m_state_provider.postprocess_state(kv,istate,m_np1,dp);
    team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<PT[NP][NP][NUM_LEV]>
  get_state(const KernelVariables &kv, int np1, int var) const {
    return m_state_provider.get_state(kv,np1,var);
  } 

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<PT[NP][NP][NUM_LEV]>
  get_source_thickness (const int ie, const int np1) const {
    return Homme::subview(m_dp3d, ie, np1);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const PT[NP][NP][NUM_LEV]> compute_source_thickness(
      KernelVariables &kv, const int np1, const Real /* dt */,
      ExecViewUnmanaged<const PT[NP][NP][NUM_LEV]> /* tgt_layer_thickness */)
      const {
    return get_source_thickness(kv.ie, np1);
  }
};

// This base class serves the purpose of allowing pimp implementation in VerticalRemapManager
template <typename ST>
struct RemapperST {
  using PT = PackType<ST>;
  virtual ~RemapperST() {}
  virtual void run_remap(int np1, int np1_qdp, double dt) = 0;
  virtual int requested_buffer_size () const = 0;
  virtual void init_buffers(const FunctorsBuffersManager& fbm) = 0;

  // Interface equivalent to Homme's remap1.
  virtual void remap1(
    ExecViewUnmanaged<const PT*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp_src, const int np1,
    ExecViewUnmanaged<const PT*[NP][NP][NUM_LEV]> dp_tgt,
    // remap v(:,1:num_to_remap,:,:,:)
    ExecViewUnmanaged<PT**[NP][NP][NUM_LEV]> v, const int num_to_remap) = 0;
  virtual void remap1(
    ExecViewUnmanaged<const PT*[NP][NP][NUM_LEV]> dp_src,
    ExecViewUnmanaged<const PT*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp_tgt, const int np1,
    // remap v(:,n_v,1:num_to_remap,:,:,:,:)
    ExecViewUnmanaged<PT***[NP][NP][NUM_LEV]> v, const int n_v,
    const int num_to_remap) = 0;
};

using Remapper = RemapperST<ScalarValue>;

// The Remap functor
template <bool nonzero_rsplit,
          typename RemapType,
          typename ST = ScalarValue>
struct RemapFunctor : public RemapperST<ST> {
  using PT = PackType<ST>;

  static_assert(std::is_base_of<VertRemapAlg, RemapType>::value,
                "RemapFunctor not given a remap algorithm to use");

  struct RemapData {
    RemapData(const int qsize_in, const int capacity_in)
      : qsize(qsize_in), capacity(capacity_in)
    {}
    const int qsize, capacity;
    int np1;
    int np1_qdp;
    Real dt;
  };

  RemapStateAndThicknessProvider<nonzero_rsplit,ST> m_fields_provider;
  RemapData m_data;

  const ElementsStateST<ST> m_state;
  const HybridVCoord m_hvcoord;
  ExecViewManaged<PT*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]> m_qdp;

  ExecViewManaged<bool *> valid_layer_thickness;
  typename decltype(valid_layer_thickness)::HostMirror host_valid_input;

  RemapType m_remap;

  TeamUtils<ExecSpace> m_tu_ne, m_tu_ne_nsr, m_tu_ne_ntr;

  explicit
  RemapFunctor (const int qsize,
                const ElementsST<ST>& elements,
                const TracersST<ST>& tracers,
                const HybridVCoord &hvcoord,
                // We don't always want to remap everything at once. State the
                // maximum capacity needed if it differs from
                //    num_states_remap + qsize.
                // If capacity < num_states_remap, num_states_remap is used.
                const int capacity=-1)
   : m_fields_provider(elements)
   , m_data(qsize, std::max(capacity, m_fields_provider.num_states_remap() + qsize))
   , m_state(elements.m_state)
   , m_hvcoord(hvcoord)
   , m_qdp(tracers.qdp)
   , m_remap(elements.num_elems(), m_data.capacity)
   // Functor tags are irrelevant below
   , m_tu_ne(remap_team_policy<ComputeThicknessTag>(m_state.num_elems()))
   , m_tu_ne_nsr(remap_team_policy<ComputeThicknessTag>(m_state.num_elems() * m_fields_provider.num_states_remap()))
   , m_tu_ne_ntr(remap_team_policy<ComputeThicknessTag>(m_state.num_elems() * num_to_remap()))
  {
    // Members used for sanity checks
    valid_layer_thickness = decltype(valid_layer_thickness)("Check for whether the surface thicknesses are positive",elements.num_elems());
    host_valid_input = Kokkos::create_mirror_view(valid_layer_thickness);
  }

  void input_valid_assert() {
    Kokkos::deep_copy(host_valid_input, valid_layer_thickness);
    bool ok = true;
    for (int ie = 0; ie < m_state.num_elems(); ++ie) {
      if ( ! host_valid_input(ie)) {
        ok = false;
        break;
      }
    }
    if ( ! ok) {
      m_state.check_print_abort_on_bad_elems(
        "Vertical remap: Negative (or nan) layer thickness detected, aborting!",
        m_data.np1);
      // If check_print_abort_on_bad_elems can't see the issue -- e.g., if the
      // source grid is bad but the state itself is fine, or the routine has no
      // implementation (as is true for preqx) -- it won't call
      // EKAT_ERROR_MSG. Thus, for safety, call EKAT_ERROR_MSG here if we're still
      // running.
      EKAT_ERROR_MSG(
        "Negative (or nan) layer thickness detected, aborting!");
    }
  }

  KOKKOS_INLINE_FUNCTION
  int num_to_remap() const { return m_fields_provider.num_states_remap() + m_data.qsize; }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<PT[NP][NP][NUM_LEV]>
  get_remap_val(const KernelVariables &kv, int var) const {
    if (!nonzero_rsplit || var >= m_fields_provider.num_states_remap()) {
      if (var >= m_fields_provider.num_states_remap())
        var -= m_fields_provider.num_states_remap();
      return Homme::subview(m_qdp, kv.ie, m_data.np1_qdp, var);
    } else {
      return m_fields_provider.get_state(kv, m_data.np1, var);
    }
  }

  struct ComputeThicknessTag {};
  struct ComputeGridsTag {};
  struct ComputeRemapTag {};
  // Computes the extrinsic values of the states in the initial map
  // i.e. velocity -> momentum
  struct ComputeExtrinsicsTag {};
  // Computes the intrinsic values of the states in the final map
  // i.e. momentum -> velocity
  struct ComputeIntrinsicsTag {};
  // Sets dp to the target dp in the state
  struct UpdateThicknessTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeThicknessTag, const TeamMember &team) const {
    KernelVariables kv(team, m_tu_ne);
    m_hvcoord.compute_ps_ref_from_dp(kv, Homme::subview(m_state.m_dp3d, kv.ie, m_data.np1),
                                         Homme::subview(m_state.m_ps_v, kv.ie, m_data.np1));

    auto tgt_layer_thickness = compute_target_thickness(kv);
    auto src_layer_thickness = m_fields_provider.compute_source_thickness(kv, m_data.np1, m_data.dt, tgt_layer_thickness);
    check_source_thickness(kv, src_layer_thickness);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeExtrinsicsTag, const TeamMember &team) const {
    KernelVariables kv(team, m_tu_ne_nsr);

    assert(m_fields_provider.num_states_remap() > 0);
    const int den = (m_fields_provider.num_states_remap() > 0) ? m_fields_provider.num_states_remap() : 1;
    const int var = kv.ie % den;
    kv.ie /= den;
    assert(kv.ie < m_state.num_elems());

    if (m_fields_provider.is_intrinsic_state(var)) {
      auto src_layer_thickness = m_fields_provider.get_source_thickness(kv.ie, m_data.np1);
      compute_extrinsic_state(
          kv, src_layer_thickness,
          m_fields_provider.get_state(kv, m_data.np1, var));
    }
  }

  // This functor is the only one guaranteed to be run
  // (in rare cases, no tracers and rsplit==0),
  // so it needs to be separated from the others to reduce latency on the GPU
  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeGridsTag, const TeamMember &team) const {
    KernelVariables kv(team, m_tu_ne);
    m_remap.compute_grids_phase(
        kv, m_fields_provider.get_source_thickness(kv.ie, m_data.np1),
        Homme::subview(m_fields_provider.m_tgt_layer_thickness, kv.ie));
  }

  // This asserts if num_to_remap() == 0
  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeRemapTag, const TeamMember &team) const {
    KernelVariables kv(team, m_tu_ne_ntr);
    assert(num_to_remap() != 0);
    const int var = kv.ie % num_to_remap();
    kv.ie /= num_to_remap();
    assert(kv.ie < m_state.num_elems());

    this->m_remap.compute_remap_phase(kv, get_remap_val(kv, var));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(ComputeIntrinsicsTag, const TeamMember &team) const {
    KernelVariables kv(team, m_tu_ne_nsr);

    assert(m_fields_provider.num_states_remap() != 0);
    const int den = (m_fields_provider.num_states_remap() > 0) ? m_fields_provider.num_states_remap() : 1;
    const int var = kv.ie % den;
    kv.ie /= den;
    assert(kv.ie < m_state.num_elems());

    if (m_fields_provider.is_intrinsic_state(var)) {
      auto tgt_layer_thickness = Homme::subview(m_fields_provider.m_tgt_layer_thickness, kv.ie);
      compute_intrinsic_state(kv, tgt_layer_thickness,
                              m_fields_provider.get_state(kv, m_data.np1, var));
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(UpdateThicknessTag, const int idx) const {
    const int ie   =  idx / (NP*NP*NUM_LEV);
    const int igp  = (idx / (NP*NUM_LEV)) % NP;
    const int jgp  = (idx / NUM_LEV) % NP;
    const int ilev =  idx % NUM_LEV;

    m_state.m_dp3d(ie,m_data.np1,igp,jgp,ilev) = m_fields_provider.m_tgt_layer_thickness(ie,igp,jgp,ilev);
  }

  void run_remap(int np1, int np1_qdp, double dt) override {
    m_data.np1 = np1;
    m_data.np1_qdp = np1_qdp;
    m_data.dt = dt;
    run_remap();
  }

  void run_remap() {
    // This runs the remap algorithm after determining it needs to
    // It also verifies the state of the simulation is valid
    // If there's nothing to remap, it will only perform the verification
    run_functor<ComputeThicknessTag>("Remap Thickness Functor",
                                     this->m_state.num_elems());
    this->input_valid_assert();
    if (num_to_remap() > 0) {
      // We don't want the latency of launching an empty kernel
      if (nonzero_rsplit) {
        // Pre-process the states if necessary
        m_fields_provider.preprocess_states(m_data.np1);

        run_functor<ComputeExtrinsicsTag>("Remap Scale States Functor",
                                          m_state.num_elems() * m_fields_provider.num_states_remap());
      }
      run_functor<ComputeGridsTag>("Remap Compute Grids Functor",
                                   m_state.num_elems());
      run_functor<ComputeRemapTag>("Remap Compute Remap Functor",
                                   m_state.num_elems() * num_to_remap());
      if (nonzero_rsplit) {
        run_functor<ComputeIntrinsicsTag>("Remap Rescale States Functor",
                                          m_state.num_elems() * m_fields_provider.num_states_remap());
        m_fields_provider.postprocess_states(m_data.np1);
      }
    }

    auto update_dp_policy = Kokkos::RangePolicy<ExecSpace,UpdateThicknessTag>(0,m_state.num_elems()*NP*NP*NUM_LEV);
    Kokkos::parallel_for(update_dp_policy, *this);
  }

  void remap1 (
    const ExecViewUnmanaged<const PT*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp_src, const int np1,
    const ExecViewUnmanaged<const PT*[NP][NP][NUM_LEV]> dp_tgt,
    const ExecViewUnmanaged<PT**[NP][NP][NUM_LEV]> v,
    const int num_remap) override
  {
    using Kokkos::ALL;
    const int ne = dp_src.extent_int(0), nv = num_remap;
    assert(nv <= m_data.capacity);
    const auto remap = m_remap;
    const auto tu_ne = m_tu_ne;
    const auto g = KOKKOS_LAMBDA (const TeamMember& team) {
      KernelVariables kv(team, tu_ne);
      remap.compute_grids_phase(kv, Homme::subview(dp_src, kv.ie, np1),
                                Homme::subview(dp_tgt, kv.ie));
    };
    Kokkos::parallel_for(get_default_team_policy<ExecSpace>(ne), g);
    const auto tu_ne_ntr = m_tu_ne_ntr;
    const auto r = KOKKOS_LAMBDA (const TeamMember& team) {
      KernelVariables kv(team, nv, tu_ne_ntr);
      remap.compute_remap_phase(kv, Kokkos::subview(v, kv.ie, kv.iq, ALL(), ALL(), ALL()));
    };
    Kokkos::fence();
    Kokkos::parallel_for(get_default_team_policy<ExecSpace>(ne*nv), r);
  }

  void remap1 (
    const ExecViewUnmanaged<const PT*[NP][NP][NUM_LEV]> dp_src,
    const ExecViewUnmanaged<const PT*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]> dp_tgt, const int np1,
    const ExecViewUnmanaged<PT***[NP][NP][NUM_LEV]> v, const int n_v,
    const int num_remap) override
  {
    using Kokkos::ALL;
    const int ne = dp_src.extent_int(0), nv = num_remap;
    assert(nv <= m_data.capacity);
    const auto remap = m_remap;
    const auto tu_ne = m_tu_ne;
    const auto g = KOKKOS_LAMBDA (const TeamMember& team) {
      KernelVariables kv(team, tu_ne);
      remap.compute_grids_phase(kv, Homme::subview(dp_src, kv.ie),
                                Homme::subview(dp_tgt, kv.ie, np1));
    };
    Kokkos::parallel_for(get_default_team_policy<ExecSpace>(ne), g);
    const auto tu_ne_ntr = m_tu_ne_ntr;
    const auto r = KOKKOS_LAMBDA (const TeamMember& team) {
      KernelVariables kv(team, nv, tu_ne_ntr);
      remap.compute_remap_phase(kv, Kokkos::subview(v, kv.ie, n_v, kv.iq, ALL(), ALL(), ALL()));
    };
    Kokkos::fence();
    Kokkos::parallel_for(get_default_team_policy<ExecSpace>(ne*nv), r);
  }

  int requested_buffer_size () const override {
    return m_fields_provider.requested_buffer_size();
  }

  void init_buffers(const FunctorsBuffersManager& fbm) override {
    m_fields_provider.init_buffers(fbm);
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
  std::enable_if_t<std::is_same_v<MyST, DxFadTypeRemap>>
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
  }

  // Compute the product J*V where J = d(state(np1)) / d(state(itl)) (where itl
  // was set in init_J).
  // Preconditions: init_J(itl,e) and run(...) must have been called first.
  // The arg state_dx should contain dU(np1)/dU(itl), where itl is the slice
  // we specified during init_J
  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, DxFadTypeRemap>>
  run_JV (const int /*np1*/, const ElementsStateST<ST>& /*state_dx*/,
          const StateSnapshot& /* x */, StateSnapshot& /* y */) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, DxFadTypeRemap>>
  run_JV (const int np1, const ElementsStateST<ST>& state_dx,
          const StateSnapshot& x, StateSnapshot& y)
  {
    // Extract the Jacobian-vector product using the product rule.
    // dw_v(ie,np1,ip,jp,k).dx(j) = d(w_np1(ie,ip,jp,k)) / d(input_j)
    // dphi_v(ie,np1,ip,jp,k).dx(j) = d(phi_np1(ie,ip,jp,k)) / d(input_j)
    const int nelem = state_dx.num_elems();

    auto dV_v   = ekat::scalarize(state_dx.m_v);
    auto dvth_v = ekat::scalarize(state_dx.m_vtheta_dp);
    auto ddp_v  = ekat::scalarize(state_dx.m_dp3d);
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

    // Compute J*V for u, v, vtheta_dp, or dp3d at np1
    auto prod_rule_mid = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      const auto& Ju   = dV_v  (ie, np1, 0, ip, jp, k).dx();
      const auto& Jv   = dV_v  (ie, np1, 1, ip, jp, k).dx();
      const auto& Jvth = dvth_v(ie, np1,    ip, jp, k).dx();
      const auto& Jdp  = ddp_v (ie, np1,    ip, jp, k).dx();

      Real y_u_new   = 0;
      Real y_v_new   = 0;
      Real y_vth_new = 0;
      Real y_dp_new  = 0;

      // Contributions from midpoint variables (u, v, vtheta_dp, dp3d)
      for (int k2 = 0; k2 < NUM_PHYSICAL_LEV; ++k2) {
        y_u_new   += Ju  [offset_u   + k2] * x_V  (ie, 0, ip, jp, k2)
                   + Ju  [offset_v   + k2] * x_V  (ie, 1, ip, jp, k2)
                   + Ju  [offset_vth + k2] * x_vth(ie,    ip, jp, k2)
                   + Ju  [offset_dp  + k2] * x_dp (ie,    ip, jp, k2);
        y_v_new   += Jv  [offset_u   + k2] * x_V  (ie, 0, ip, jp, k2)
                   + Jv  [offset_v   + k2] * x_V  (ie, 1, ip, jp, k2)
                   + Jv  [offset_vth + k2] * x_vth(ie,    ip, jp, k2)
                   + Jv  [offset_dp  + k2] * x_dp (ie,    ip, jp, k2);
        y_vth_new += Jvth[offset_u   + k2] * x_V  (ie, 0, ip, jp, k2)
                   + Jvth[offset_v   + k2] * x_V  (ie, 1, ip, jp, k2)
                   + Jvth[offset_vth + k2] * x_vth(ie,    ip, jp, k2)
                   + Jvth[offset_dp  + k2] * x_dp (ie,    ip, jp, k2);
        y_dp_new  += Jdp [offset_u   + k2] * x_V  (ie, 0, ip, jp, k2)
                   + Jdp [offset_v   + k2] * x_V  (ie, 1, ip, jp, k2)
                   + Jdp [offset_vth + k2] * x_vth(ie,    ip, jp, k2)
                   + Jdp [offset_dp  + k2] * x_dp (ie,    ip, jp, k2);
      }
      // Contributions from interface variables (w_i, phinh_i)
      for (int k2 = 0; k2 < NUM_INTERFACE_LEV; ++k2) {
        y_u_new   += Ju  [offset_w   + k2] * x_w  (ie, ip, jp, k2)
                   + Ju  [offset_phi + k2] * x_phi(ie, ip, jp, k2);
        y_v_new   += Jv  [offset_w   + k2] * x_w  (ie, ip, jp, k2)
                   + Jv  [offset_phi + k2] * x_phi(ie, ip, jp, k2);
        y_vth_new += Jvth[offset_w   + k2] * x_w  (ie, ip, jp, k2)
                   + Jvth[offset_phi + k2] * x_phi(ie, ip, jp, k2);
        y_dp_new  += Jdp [offset_w   + k2] * x_w  (ie, ip, jp, k2)
                   + Jdp [offset_phi + k2] * x_phi(ie, ip, jp, k2);
      }

      y_V  (ie, 0, ip, jp, k) = y_u_new;
      y_V  (ie, 1, ip, jp, k) = y_v_new;
      y_vth(ie,    ip, jp, k) = y_vth_new;
      y_dp (ie,    ip, jp, k) = y_dp_new;
    };

    // Compute J*V for w_i and phinh_i at np1
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

    Kokkos::parallel_for(p4_mid, prod_rule_mid);
    Kokkos::parallel_for(p4_int, prod_rule_int);
  }

  // Compute the product J^T*V where J = d(state(np1)) / d(state(itl)) (where itl
  // was set in init_J).
  // Preconditions: init_J(itl,e) and run(...) must have been called first.
  // The arg state_dx should contain dU(np1)/dU(itl), where itl is the slice
  // we specified during init_J
  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, DxFadTypeRemap>>
  run_JtV (const int /*np1*/, const ElementsStateST<ST>& /*state_dx*/,
           const StateSnapshot& /*x*/, StateSnapshot& /*y*/) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, DxFadTypeRemap>>
  run_JtV (const int np1, const ElementsStateST<ST>& state_dx,
           const StateSnapshot& x, StateSnapshot& y)
  {
    // General transpose product: y = J^T * x
    const int nelem = state_dx.num_elems();

    // Extract derivative-containing views produced by init_J + run
    auto dV_v   = ekat::scalarize(state_dx.m_v);
    auto dvth_v = ekat::scalarize(state_dx.m_vtheta_dp);
    auto ddp_v  = ekat::scalarize(state_dx.m_dp3d);
    auto dw_v   = ekat::scalarize(state_dx.m_w_i);
    auto dphi_v = ekat::scalarize(state_dx.m_phinh_i);

    // Input (adjoint at np1)
    auto x_V   = ekat::scalarize(x.v);
    auto x_vth = ekat::scalarize(x.vtheta_dp);
    auto x_dp  = ekat::scalarize(x.dp3d);
    auto x_w   = ekat::scalarize(x.w_i);
    auto x_phi = ekat::scalarize(x.phinh_i);

    // Output (adjoint at itl)
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

    // Compute (J^T * x)_w(k2) and (J^T * x)_phi(k2) for interface inputs k2.
    // Sum contributions from all midpoint outputs and interface outputs.
    auto jtv_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k2) {
      Real y_w_new   = 0;
      Real y_phi_new = 0;

      // Contributions from midpoint outputs
      for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
        const auto& Ju   = dV_v  (ie, np1, 0, ip, jp, k).dx();
        const auto& Jv   = dV_v  (ie, np1, 1, ip, jp, k).dx();
        const auto& Jvth = dvth_v(ie, np1,    ip, jp, k).dx();
        const auto& Jdp  = ddp_v (ie, np1,    ip, jp, k).dx();

        const Real x_V_u = x_V  (ie, 0, ip, jp, k);
        const Real x_V_v = x_V  (ie, 1, ip, jp, k);
        const Real x_vth_k = x_vth(ie,    ip, jp, k);
        const Real x_dp_k  = x_dp (ie,    ip, jp, k);

        y_w_new   += Ju  [offset_w   + k2] * x_V_u
                   + Jv  [offset_w   + k2] * x_V_v
                   + Jvth[offset_w   + k2] * x_vth_k
                   + Jdp [offset_w   + k2] * x_dp_k;

        y_phi_new += Ju  [offset_phi + k2] * x_V_u
                   + Jv  [offset_phi + k2] * x_V_v
                   + Jvth[offset_phi + k2] * x_vth_k
                   + Jdp [offset_phi + k2] * x_dp_k;
      }

      // Contributions from interface outputs
      for (int k = 0; k < NUM_INTERFACE_LEV; ++k) {
        const auto& Jw   = dw_v  (ie, np1, ip, jp, k).dx();
        const auto& Jphi = dphi_v(ie, np1, ip, jp, k).dx();

        const Real x_w_k   = x_w  (ie, ip, jp, k);
        const Real x_phi_k = x_phi(ie, ip, jp, k);

        y_w_new   += Jw  [offset_w   + k2] * x_w_k
                   + Jphi[offset_w   + k2] * x_phi_k;
        y_phi_new += Jw  [offset_phi + k2] * x_w_k
                   + Jphi[offset_phi + k2] * x_phi_k;
      }

      y_w  (ie, ip, jp, k2) = y_w_new;
      y_phi(ie, ip, jp, k2) = y_phi_new;
    };

    // Compute (J^T * x)_u(k2), _v(k2), _vth(k2), _dp(k2) for midpoint inputs k2.
    // Sum contributions from midpoint and interface outputs.
    auto jtv_mid = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k2) {
      Real y_u_new   = 0;
      Real y_v_new   = 0;
      Real y_vth_new = 0;
      Real y_dp_new  = 0;

      // Contributions from midpoint outputs
      for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
        const auto& Ju   = dV_v  (ie, np1, 0, ip, jp, k).dx();
        const auto& Jv   = dV_v  (ie, np1, 1, ip, jp, k).dx();
        const auto& Jvth = dvth_v(ie, np1,    ip, jp, k).dx();
        const auto& Jdp  = ddp_v (ie, np1,    ip, jp, k).dx();

        const Real x_V_u = x_V  (ie, 0, ip, jp, k);
        const Real x_V_v = x_V  (ie, 1, ip, jp, k);
        const Real x_vth_k = x_vth(ie,    ip, jp, k);
        const Real x_dp_k  = x_dp (ie,    ip, jp, k);

        y_u_new   += Ju  [offset_u   + k2] * x_V_u
                   + Jv  [offset_u   + k2] * x_V_v
                   + Jvth[offset_u   + k2] * x_vth_k
                   + Jdp [offset_u   + k2] * x_dp_k;

        y_v_new   += Ju  [offset_v   + k2] * x_V_u
                   + Jv  [offset_v   + k2] * x_V_v
                   + Jvth[offset_v   + k2] * x_vth_k
                   + Jdp [offset_v   + k2] * x_dp_k;

        y_vth_new += Ju  [offset_vth + k2] * x_V_u
                   + Jv  [offset_vth + k2] * x_V_v
                   + Jvth[offset_vth + k2] * x_vth_k
                   + Jdp [offset_vth + k2] * x_dp_k;

        y_dp_new  += Ju  [offset_dp  + k2] * x_V_u
                   + Jv  [offset_dp  + k2] * x_V_v
                   + Jvth[offset_dp  + k2] * x_vth_k
                   + Jdp [offset_dp  + k2] * x_dp_k;
      }

      // Contributions from interface outputs
      for (int k = 0; k < NUM_INTERFACE_LEV; ++k) {
        const auto& Jw   = dw_v  (ie, np1, ip, jp, k).dx();
        const auto& Jphi = dphi_v(ie, np1, ip, jp, k).dx();

        const Real x_w_k   = x_w  (ie, ip, jp, k);
        const Real x_phi_k = x_phi(ie, ip, jp, k);

        y_u_new   += Jw  [offset_u   + k2] * x_w_k
                   + Jphi[offset_u   + k2] * x_phi_k;
        y_v_new   += Jw  [offset_v   + k2] * x_w_k
                   + Jphi[offset_v   + k2] * x_phi_k;
        y_vth_new += Jw  [offset_vth + k2] * x_w_k
                   + Jphi[offset_vth + k2] * x_phi_k;
        y_dp_new  += Jw  [offset_dp  + k2] * x_w_k
                   + Jphi[offset_dp  + k2] * x_phi_k;
      }

      y_V  (ie, 0, ip, jp, k2) = y_u_new;
      y_V  (ie, 1, ip, jp, k2) = y_v_new;
      y_vth(ie,    ip, jp, k2) = y_vth_new;
      y_dp (ie,    ip, jp, k2) = y_dp_new;
    };

    Kokkos::parallel_for(p4_int, jtv_int);
    Kokkos::parallel_for(p4_mid, jtv_mid);
  }
#endif // HOMMEXX_ENABLE_FAD_TYPES

private:
  template <typename FunctorTag>
  typename std::enable_if<OnGpu<ExecSpace>::value == false,
                          Kokkos::TeamPolicy<ExecSpace, FunctorTag> >::type
  remap_team_policy(int num_exec) {
    return Homme::get_default_team_policy<ExecSpace, FunctorTag>(num_exec);
  }

  template <typename FunctorTag>
  typename std::enable_if<OnGpu<ExecSpace>::value == true,
                          Kokkos::TeamPolicy<ExecSpace, FunctorTag> >::type
  remap_team_policy(int num_exec) {
    ThreadPreferences tp;
    tp.max_threads_usable = 16;
    tp.max_vectors_usable = 32;
    tp.prefer_larger_team = true;
    return Homme::get_default_team_policy<ExecSpace, FunctorTag>(num_exec, tp);
  }

  template <typename FunctorTag>
  void run_functor(const std::string functor_name, int num_exec) {
    const auto policy = remap_team_policy<FunctorTag>(num_exec);
    // Timers don't work on CUDA, so place them here
    GPTLstart(functor_name.c_str());
    profiling_resume();
    Kokkos::parallel_for("vertical remap", policy, *this);
    Kokkos::fence();
    profiling_pause();
    GPTLstop(functor_name.c_str());
  }

  KOKKOS_INLINE_FUNCTION
  void compute_extrinsic_state(
      KernelVariables &kv,
      ExecViewUnmanaged<const PT[NP][NP][NUM_LEV]> src_layer_thickness,
      ExecViewUnmanaged<PT[NP][NP][NUM_LEV]> state_remap) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &ilev) {
        state_remap(igp, jgp, ilev) *= src_layer_thickness(igp, jgp, ilev);

      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_intrinsic_state(
      KernelVariables &kv,
      ExecViewUnmanaged<const PT[NP][NP][NUM_LEV]> tgt_layer_thickness,
      ExecViewUnmanaged<PT[NP][NP][NUM_LEV]> state_remap) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &ilev) {
        state_remap(igp, jgp, ilev) /= tgt_layer_thickness(igp, jgp, ilev);
      });
    });
  }

  KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<const PT[NP][NP][NUM_LEV]>
  compute_target_thickness(KernelVariables &kv) const {
    auto tgt_layer_thickness = Homme::subview(m_fields_provider.m_tgt_layer_thickness, kv.ie);

    m_hvcoord.compute_dp_ref(kv,Homme::subview(m_state.m_ps_v,kv.ie,m_data.np1),tgt_layer_thickness);

    return tgt_layer_thickness;
  }

  KOKKOS_INLINE_FUNCTION void check_source_thickness(
      KernelVariables &kv,
      ExecViewUnmanaged<const PT[NP][NP][NUM_LEV]> src_layer_thickness)
      const {
    // Kokkos parallel reduce doesn't support bool as a reduction type, so use
    // int instead
    // Reduce starts with false (0), making that the default state
    // If there is an error, this becomes true
    int invalid = false;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(kv.team, NP * NP * NUM_PHYSICAL_LEV),
        [&](const int &loop_idx, int &is_invalid) {
          const int igp = (loop_idx / NUM_PHYSICAL_LEV) / NP;
          const int jgp = (loop_idx / NUM_PHYSICAL_LEV) % NP;
          const int level = loop_idx % NUM_PHYSICAL_LEV;
          const int ilev = level / VECTOR_SIZE;
          const int vlev = level % VECTOR_SIZE;
          is_invalid |= Homme::isnan(ADValue(src_layer_thickness(igp, jgp, ilev)[vlev]));
          is_invalid |= (src_layer_thickness(igp, jgp, ilev)[vlev] < 0.0);
        },
        invalid);
    valid_layer_thickness(kv.ie) = !invalid;
  }
};

} // namespace Remap
} // namespace Homme

#endif // HOMMEXX_REMAP_FUNCTOR_HPP
