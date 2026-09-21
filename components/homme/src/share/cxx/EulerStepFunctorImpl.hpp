/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_EULER_STEP_FUNCTOR_IMPL_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_IMPL_HPP

#include "Context.hpp"
#include "ReferenceElement.hpp"
#include "ElementsGeometry.hpp"
#include "ElementsDerivedState.hpp"
#include "EulerStepAdjointTape.hpp"
#include "FunctorsBuffersManager.hpp"
#include "ErrorDefs.hpp"
#include "EulerStepFunctor.hpp"
#include "HommexxEnums.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "Tracers.hpp"
#include "profiling.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/MpiBuffersManager.hpp"
#include "mpi/Connectivity.hpp"
#include "utilities/SubviewUtils.hpp"
#include "vector/vector_pragmas.hpp"

namespace Homme {

// On older machines, low memory b/w is the most important performance-influence
// characteristic. On these machines, a tracer will be processed in serial. Take
// advantage of this to optimize memory access.
template <typename ExecSpace>
struct SerialLimiter {
  template <int limiter_option, typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl,
            typename Array2GllLvl>
  KOKKOS_INLINE_FUNCTION static void
  run(const ArrayGll& sphweights, const ArrayGllLvl& idpmass,
      const Array2Lvl& iqlim, const ArrayGllLvl& iptens,
      const Array2GllLvl& irwrk);
};
// GPU doesn't have a serial impl.
#ifdef HOMMEXX_ENABLE_GPU
template <>
struct SerialLimiter<HommexxGPU> {
  template <int limiter_option, typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl,
            typename Array2GllLvl>
  KOKKOS_INLINE_FUNCTION static void
  run (const ArrayGll& sphweights, const ArrayGllLvl& idpmass,
       const Array2Lvl& iqlim, const ArrayGllLvl& iptens,
       const Array2GllLvl& irwrk) {
    Kokkos::abort("SerialLimiter::run: Should not be called on GPU.");
  }
};
#endif


template<typename ST>
class EulerStepFunctorImplST {
  using PT = PackType<ST>;

  struct EulerStepData {
    EulerStepData ()
      : qsize(-1), limiter_option(0), nu_p(0), nu_q(0), consthv(1)
    {}

    int   qsize;
    int   limiter_option;
    Real  rhs_viss;
    Real  rhs_multiplier;

    Real  nu_p;
    Real  nu_q;

    Real  dt;
    int   np1_qdp;
    int   n0_qdp;

    DSSOption   DSSopt;

    bool consthv;
  };

  struct Buffers {
    static constexpr int num_3d_scalar_mid_buf = 2;
    static constexpr int num_3d_vector_mid_buf = 1;

    ExecViewUnmanaged<PT*   [NP][NP][NUM_LEV]> dp;
    ExecViewUnmanaged<PT*   [NP][NP][NUM_LEV]> dpdissk;
    ExecViewUnmanaged<PT*[2][NP][NP][NUM_LEV]> vstar;
  };

  using deriv_type = ReferenceElement::deriv_type;

  ElementsGeometry            m_geometry;
  const int                   m_num_elems;
  Buffers                     m_buffers;
  ElementsDerivedStateST<ST>  m_derived_state;
  TracersST<ST>               m_tracers;
  deriv_type                  m_deriv;
  HybridVCoord                m_hvcoord;
  EulerStepData               m_data;
  SphereOperatorsST<ST>       m_sphere_ops;

  Kokkos::TeamPolicy<ExecSpace> m_tv_policy;
  TeamUtils<ExecSpace> m_tu_ne, m_tu_ne_qsize;

  int m_prev_num_elems, m_prev_qsize;

  bool                m_kernel_will_run_limiters;

  ThreadPreferences m_tpref;

  std::shared_ptr<BoundaryExchangeST<ST>> m_mm_be, m_mmqb_be;
  Kokkos::Array<std::shared_ptr<BoundaryExchangeST<ST>>, 3*Q_NUM_TIME_LEVELS> m_bes;

  enum { m_mem_per_team = 2 * NP * NP * sizeof(ST) };

  // ============================================================
  // Adjoint of euler_step (limiter_option==9 only). See euler_step_adj
  // for the full design writeup. Real-only (mirrors how HV's run_JtV and
  // ForcingFunctor::states_forcing_adj are Real-only). The bulk of the
  // adjoint's scratch/tape storage (9 Views, plus the lazily-built
  // boundary-exchange objects) used to live here as plain data members,
  // which meant every instantiation of this class template -- including
  // ST==DpFadType, ST==DxFadTypeCaar, etc., none of which ever touch the
  // adjoint -- carried it (harmlessly, but wastefully) unused. It now lives
  // once, lazily, in Context::singleton().any_map() as an
  // EulerStepAdjointTape (see EulerStepAdjointTape.hpp for the struct
  // itself and its fields' meanings), fetched via get_adjoint_tape()/
  // alloc_adjoint_tape() below, mirroring how Tape<StateSnapshot> is
  // stashed there under "imex_tape" for the CAAR/DIRK adjoint.
  // ============================================================
  using RPT = PackType<Real>;

  bool m_tape_for_adjoint = false;

  // Real-only, device-reachable cache of two of the Context-held adjoint
  // tape's Views (EulerStepAdjointTape::qtens_prelimiter and ::qlim_final).
  // Unlike every other adjoint-tape access in this file (which fetch
  // straight from Context in a host-only function and capture local Views
  // into a KOKKOS_LAMBDA, costing nothing extra), run_tracer_phase is a
  // KOKKOS_INLINE_FUNCTION invoked via the `*this`-dispatched
  // Kokkos::parallel_for in advect_and_limit() (see
  // operator()(const AALTracerPhase&, ...)), so its body can execute on a
  // GPU device. Context::singleton().any_map() (std::map/std::any) cannot
  // be touched from device code, so these two Views must already be part
  // of `*this` -- a plain Kokkos::View is a cheap, device-copyable handle,
  // unlike the map/any lookup needed to reach it inside Context -- by the
  // time that parallel_for launches. advect_and_limit() refreshes these
  // (a shallow View-handle copy, no allocation) from the Context-owned
  // EulerStepAdjointTape right before dispatch. For any ST other than
  // Real, these stay default-constructed (unallocated, unused), same as
  // every other adjoint-only member of this class.
  ExecViewManaged<RPT**[NP][NP][NUM_LEV]>    m_adj_qtens_prelimiter_dev;
  ExecViewManaged<RPT*[QSIZE_D][2][NUM_LEV]> m_adj_qlim_final_dev;

public:

  EulerStepFunctorImplST ()
   : m_geometry      (Context::singleton().get<ElementsGeometry>())
   , m_num_elems     (m_geometry.num_elems())
   , m_derived_state (Context::singleton().get<ElementsDerivedStateST<ST>>())
   , m_tracers       (Context::singleton().get<TracersST<ST>>())
   , m_deriv         (Context::singleton().get<ReferenceElement>().get_deriv())
   , m_hvcoord       (Context::singleton().get<HybridVCoord>())
   , m_sphere_ops    (Context::singleton().get<SphereOperatorsST<ST>>())
   , m_tv_policy     (Homme::get_default_team_policy<ExecSpace>(1))
   , m_tu_ne         (Homme::get_default_team_policy<ExecSpace>(1))
   , m_tu_ne_qsize   (Homme::get_default_team_policy<ExecSpace>(1))
   , m_prev_num_elems(0)
   , m_prev_qsize    (0)
  {
    m_kernel_will_run_limiters = false;
    m_tpref.prefer_larger_team = true;
  }

  EulerStepFunctorImplST (const int num_elems)
    : m_num_elems     (num_elems)
    , m_tv_policy     (Homme::get_default_team_policy<ExecSpace>(1))
    , m_tu_ne         (Homme::get_default_team_policy<ExecSpace>(1))
    , m_tu_ne_qsize   (Homme::get_default_team_policy<ExecSpace>(1))
    , m_prev_num_elems(0)
    , m_prev_qsize    (0)
  {}

  void setup ()
  {
    m_geometry      = Context::singleton().get<ElementsGeometry>();
    assert(m_num_elems == m_geometry.num_elems()); // Sanity check
    m_derived_state = Context::singleton().get<ElementsDerivedStateST<ST>>();
    m_tracers       = Context::singleton().get<TracersST<ST>>();
    m_deriv         = Context::singleton().get<ReferenceElement>().get_deriv();
    m_hvcoord       = Context::singleton().get<HybridVCoord>();
    m_sphere_ops    = Context::singleton().get<SphereOperatorsST<ST>>();

    m_kernel_will_run_limiters = false;
    m_tpref.prefer_larger_team = true;
  }

  void reset (const SimulationParams& params) {
    m_data.rhs_viss = 0.0;
    m_data.qsize = params.qsize;
    m_data.limiter_option = params.limiter_option;
    m_data.nu_p = params.nu_p;
    m_data.nu_q = params.nu_q;
    m_data.consthv = (params.hypervis_scaling == 0);

    if (m_data.limiter_option == 4) {
      std::string msg = "[EulerStepFunctorImplST::reset]:";
      msg += "limiter_option=4 is not yet supported in C++. ";
      msg += "The program should have errored out earlier though. Please, investigate.";
      EKAT_ERROR_MSG(msg);
    }

    // Make sure sphere ops have buffers large enough to accommodate this functor's needs
    if (m_geometry.num_elems() != m_prev_num_elems || m_data.qsize != m_prev_qsize) {
      m_prev_num_elems = m_geometry.num_elems();
      m_prev_qsize     = m_data.qsize;

      const auto num_parallel_iterations = m_geometry.num_elems() * m_data.qsize;

      auto tp_ne       = Homme::get_default_team_policy<ExecSpace>(m_geometry.num_elems());
      auto tp_ne_qsize = Homme::get_default_team_policy<ExecSpace>(num_parallel_iterations, m_tpref);

      ThreadPreferences tp;
      tp.max_threads_usable = NUM_LEV;
      tp.max_vectors_usable = 1;
      const auto tv =
        DefaultThreadsDistribution<ExecSpace>::team_num_threads_vectors(
          num_parallel_iterations, tp);
      m_tv_policy = decltype(m_tv_policy)(num_parallel_iterations, tv.first, tv.second);

      m_tu_ne       = TeamUtils<ExecSpace>(tp_ne);
      m_tu_ne_qsize = TeamUtils<ExecSpace>(tp_ne_qsize);

      m_sphere_ops.allocate_buffers(m_tu_ne_qsize);
    }
  }

  int requested_buffer_size () const {
    constexpr int size_scalar =   NP*NP*NUM_LEV*VECTOR_SIZE;
    constexpr int size_vector = 2*NP*NP*NUM_LEV*VECTOR_SIZE;
    constexpr int num_scalars = Buffers::num_3d_scalar_mid_buf;
    constexpr int num_vectors = Buffers::num_3d_vector_mid_buf;

    int scl_sz = sizeof(ST) / sizeof(Real);
    return m_geometry.num_elems() * (num_scalars*size_scalar + num_vectors*size_vector) * scl_sz;
  }

  void init_buffers (const FunctorsBuffersManager& fbm) {
    EKAT_REQUIRE_MSG(fbm.allocated_size()>=requested_buffer_size(), "Error! Buffers size not sufficient.\n");

    constexpr int size_scalar =   NP*NP*NUM_LEV;
    const int ne = m_geometry.num_elems();

    PT* mem = reinterpret_cast<PT*>(fbm.get_memory());

    m_buffers.dp      = decltype(m_buffers.dp)(mem,ne);
    mem += size_scalar*ne;

    m_buffers.dpdissk = decltype(m_buffers.dpdissk)(mem,ne);
    mem += size_scalar*ne;

    m_buffers.vstar   = decltype(m_buffers.vstar)(mem,ne);
  }

  void init_boundary_exchanges () {
    assert(m_data.qsize >= 0); // after reset() called

    auto bm_exchange = Context::singleton().get<MpiBuffersManagerMap>()[MPI_EXCHANGE];
    DSSOption dss_vars[3] = {DSSOption::ETA, DSSOption::OMEGA, DSSOption::DIV_VDP_AVE};
    for (int np1_qdp = 0, k = 0; np1_qdp < Q_NUM_TIME_LEVELS; ++np1_qdp) {
      for (auto dssi : dss_vars) {
        m_bes[k] = std::make_shared<BoundaryExchangeST<ST>>();
        BoundaryExchangeST<ST>& be = *m_bes[k];
        be.set_buffers_manager(bm_exchange);
        int num_mid = dssi==DSSOption::ETA ? 0 : 1;
        int num_int = 1 - num_mid;
        be.set_num_fields(0, 0, m_data.qsize+num_mid,num_int);
        be.register_field(m_tracers.qdp, np1_qdp, m_data.qsize, 0);
        switch(dssi) {
          case DSSOption::ETA:
            be.register_field(m_derived_state.m_eta_dot_dpdn);
            break;
          case DSSOption::OMEGA:
            be.register_field(m_derived_state.m_omega_p);
            break;
          case DSSOption::DIV_VDP_AVE:
            be.register_field(m_derived_state.m_divdp_proj);
            break;
        }
        be.registration_completed();
        ++k;
      }
    }

    {
      m_mmqb_be = std::make_shared<BoundaryExchangeST<ST>>();
      m_mmqb_be->set_buffers_manager(bm_exchange);
      m_mmqb_be->set_num_fields(0, 0, m_data.qsize);
      m_mmqb_be->register_field(m_tracers.qtens_biharmonic, m_data.qsize, 0);
      m_mmqb_be->registration_completed();
    }

    {
      auto bm_exchange_minmax = Context::singleton().get<MpiBuffersManagerMap>()[MPI_EXCHANGE_MIN_MAX];
      m_mm_be = std::make_shared<BoundaryExchangeST<ST>>();
      BoundaryExchangeST<ST>& be = *m_mm_be;
      be.set_buffers_manager(bm_exchange_minmax);
      be.set_num_fields(m_data.qsize, 0, 0);
      be.register_min_max_fields(m_tracers.qlim, m_data.qsize, 0);
      be.registration_completed();
    }
  }

  static size_t limiter_team_shmem_size (const int team_size) {
    return Memory<ExecSpace>::on_gpu ?
      (team_size * m_mem_per_team) :
      0;
  }

  size_t team_shmem_size (const int team_size) const {
    return m_kernel_will_run_limiters ? limiter_team_shmem_size(team_size) : 0;
  }

  struct BIHPreNup {};
  struct BIHPreNoNup {};
  struct BIHPostConstHV {};
  struct BIHPostTensorHV {};

  /*
    ! get new min/max values, and also compute biharmonic mixing term

    ! two scalings depending on nu_p:
    ! nu_p=0:    qtens_biharmonic *= dp0                   (apply viscosity only to q)
    ! nu_p>0):   qtens_biharmonc *= elem()%psdiss_ave      (for consistency, if nu_p=nu_q)
   */
  void compute_biharmonic_pre() {
    profiling_resume();
    assert(m_data.rhs_multiplier == 2.0);
    m_data.rhs_viss = 3.0;

    if(m_data.nu_p > 0){
    Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, BIHPreNup>(
                           m_geometry.num_elems() * m_data.qsize, m_tpref),
                         *this);
    }else{
    Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, BIHPreNoNup>(
                           m_geometry.num_elems() * m_data.qsize, m_tpref),
                         *this);

    }

    Kokkos::fence();
    profiling_pause();
  }

  void compute_biharmonic_post() {
    profiling_resume();
    assert(m_data.rhs_multiplier == 2.0);

    if(m_data.consthv){
    Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, BIHPostConstHV>(
                           m_geometry.num_elems() * m_data.qsize, m_tpref),
                         *this);
    }else{
    Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace, BIHPostTensorHV>(
                           m_geometry.num_elems() * m_data.qsize, m_tpref),
                         *this);
    }
    Kokkos::fence();
    profiling_pause();
  }

//case when nu_p > 0
  KOKKOS_INLINE_FUNCTION
  void operator() (const BIHPreNup&, const TeamMember& team) const {
    KernelVariables kv(team, m_data.qsize, m_tu_ne_qsize);
    const auto qtens_biharmonic = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
    dpdiss_adjustment(kv, team);
    m_sphere_ops.laplace_simple(kv, qtens_biharmonic, qtens_biharmonic);
  }

//case when nu_p == 0
  KOKKOS_INLINE_FUNCTION
  void operator() (const BIHPreNoNup&, const TeamMember& team) const {
    KernelVariables kv(team, m_data.qsize, m_tu_ne_qsize);
    const auto qtens_biharmonic = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
    m_sphere_ops.laplace_simple(kv, qtens_biharmonic, qtens_biharmonic);
  }

  KOKKOS_INLINE_FUNCTION
  void dpdiss_adjustment (KernelVariables & kv, const TeamMember& team) const {

    const auto qtens_biharmonic = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
      const auto dpdiss_ave = Homme::subview(m_derived_state.m_dpdiss_ave, kv.ie);
      Kokkos::parallel_for (
        Kokkos::TeamThreadRange(team, NP*NP),
        [&] (const int loop_idx) {
          const int i = loop_idx / NP;
          const int j = loop_idx % NP;
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team, NUM_LEV),
            [&] (const int& k) {
              qtens_biharmonic(i,j,k) = qtens_biharmonic(i,j,k) * dpdiss_ave(i,j,k) / m_hvcoord.dp0(k);
            });
        });
      team.team_barrier();
  }



  KOKKOS_INLINE_FUNCTION
  void operator() (const BIHPostConstHV&, const TeamMember& team) const {
    KernelVariables kv(team, m_data.qsize, m_tu_ne_qsize);
    const auto qtens_biharmonic = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
    team.team_barrier();
    m_sphere_ops.laplace_simple(kv, qtens_biharmonic, qtens_biharmonic);
    // laplace_simple provides the barrier.
    rhsviss_adjustment(kv, team);
  }//end of BIHPostConstHV ()

  KOKKOS_INLINE_FUNCTION
  void operator() (const BIHPostTensorHV&, const TeamMember& team) const {
    KernelVariables kv(team,m_data.qsize, m_tu_ne_qsize);
    const auto qtens_biharmonic = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
    const auto tensor = Homme::subview(m_geometry.m_tensorvisc, kv.ie);
    team.team_barrier();
    m_sphere_ops.laplace_tensor(kv, tensor, qtens_biharmonic, qtens_biharmonic);
    // divergence_sphere_wk provides the barrier.
    rhsviss_adjustment(kv, team);
    }//end of BIHPostTensorHV ()

  KOKKOS_INLINE_FUNCTION
  void rhsviss_adjustment (KernelVariables & kv, const TeamMember & team) const {
    const auto qtens_biharmonic = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
    const auto f = -m_data.rhs_viss * m_data.dt * m_data.nu_q;
    const auto spheremp = Homme::subview(m_geometry.m_spheremp, kv.ie);
      Kokkos::parallel_for (
        Kokkos::TeamThreadRange(team, NP*NP),
        [&] (const int loop_idx) {
          const int i = loop_idx / NP;
          const int j = loop_idx % NP;
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team, NUM_LEV),
            [&] (const int& k) {
              qtens_biharmonic(i,j,k) = (f * m_hvcoord.dp0(k) * qtens_biharmonic(i,j,k) /
                                         spheremp(i,j));
            });
        });
    }



  struct AALSetupPhase {};
  struct AALTracerPhase {};

  void advect_and_limit() {
    profiling_resume();
    // Refresh the device-reachable adjoint-tape View cache (see
    // m_adj_qtens_prelimiter_dev/m_adj_qlim_final_dev's declaration) from
    // the Context-held EulerStepAdjointTape right before the `*this`-
    // dispatched kernels below launch; run_tracer_phase (invoked from the
    // AALTracerPhase kernel) needs these two Views but, unlike every other
    // adjoint-tape use in this file, cannot fetch them from Context itself
    // since it may execute on a GPU device.
    if constexpr (std::is_same_v<ST,Real>) {
      if (m_tape_for_adjoint) {
        auto& adj_tape = get_adjoint_tape();
        m_adj_qtens_prelimiter_dev = adj_tape.qtens_prelimiter;
        m_adj_qlim_final_dev       = adj_tape.qlim_final;
      }
    }
    Kokkos::parallel_for(
      Homme::get_default_team_policy<ExecSpace, AALSetupPhase>(
        m_geometry.num_elems(), m_tpref),
      *this);
    Kokkos::fence();
    m_kernel_will_run_limiters = true;
    Kokkos::parallel_for(
      //to play with launch bounds
      //Homme::get_default_team_policy<ExecSpace, AALTracerPhase, Kokkos::LaunchBounds<128,1> >(
      Homme::get_default_team_policy<ExecSpace, AALTracerPhase >(
        m_geometry.num_elems() * m_data.qsize, m_tpref),
      *this);
    Kokkos::fence();
    m_kernel_will_run_limiters = false;
    profiling_pause();
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const AALSetupPhase&, const TeamMember& team) const {
    KernelVariables kv(team, m_tu_ne);
    run_setup_phase(kv);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const AALTracerPhase&, const TeamMember& team) const {
    KernelVariables kv(team, m_data.qsize, m_tu_ne_qsize);
    run_tracer_phase(kv);
  }

  struct PrecomputeDivDp {};

  void precompute_divdp() {
    assert(m_data.qsize >= 0); // reset() already called
    profiling_resume();

    Kokkos::parallel_for(
        Homme::get_default_team_policy<ExecSpace, PrecomputeDivDp>(
            m_geometry.num_elems(), m_tpref),
        *this);

    Kokkos::fence();
    profiling_pause();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const PrecomputeDivDp &, const TeamMember &team) const {
    KernelVariables kv(team, m_tu_ne);
    m_sphere_ops.divergence_sphere(kv,
                      Homme::subview(m_derived_state.m_vn0, kv.ie),
                      Homme::subview(m_derived_state.m_divdp, kv.ie));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV),
                           [&](const int ilev) {
        m_derived_state.m_divdp_proj(kv.ie, igp, jgp, ilev) =
            m_derived_state.m_divdp(kv.ie, igp, jgp, ilev);
      });
    });
  }

  void qdp_time_avg (const int n0_qdp, const int np1_qdp) {
    const int qsize = m_data.qsize;
    const auto qdp = m_tracers.qdp;
    const Real rkstage = 3.0;
    Kokkos::parallel_for(
      Homme::get_default_team_policy<ExecSpace>(m_geometry.num_elems()*m_data.qsize,
                                                m_tpref),
      KOKKOS_LAMBDA(const TeamMember& team) {
        KernelVariables kv(team, qsize); // no team-idx used, so no need for TU
        const auto qdp_n0 = Homme::subview(qdp, kv.ie, n0_qdp, kv.iq);
        const auto qdp_np1 = Homme::subview(qdp, kv.ie, np1_qdp, kv.iq);
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(kv.team, NP*NP),
          [&] (const int& idx) {
            const int i = idx / NP;
            const int j = idx % NP;
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
              [&] (const int& ilev) {
               qdp_np1(i,j,ilev) =
                 (qdp_n0(i,j,ilev) + (rkstage-1)*qdp_np1(i,j,ilev)) /
                 rkstage;
            });
          });
      });
  }

  // TODO make GPUable.
  void compute_dp () {
    const auto& c = m_data;
    const auto dp = m_derived_state.m_dp;
    const auto divdp_proj = m_derived_state.m_divdp_proj;
    const auto rhsmdt = c.rhs_multiplier * c.dt;
    const auto buf = m_buffers.dp;
    Kokkos::parallel_for(
      Homme::get_default_team_policy<ExecSpace>(m_geometry.num_elems(), m_tpref),
      KOKKOS_LAMBDA (const TeamMember& team) {
        KernelVariables kv(team); // no team-idx used, so no need for TU
        Kokkos::parallel_for (
          Kokkos::TeamThreadRange(kv.team, NP*NP),
          [&] (const int loop_idx) {
            const int i = loop_idx / NP;
            const int j = loop_idx % NP;
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
              [&] (const int& k) {
                //! derived variable divdp_proj() (DSS'd version of divdp) will only
                //! be correct on 2nd and 3rd stage but that's ok because
                //! rhs_multiplier=0 on the first stage:
                // Store this in unused buffer.
                buf(kv.ie,i,j,k) =
                  dp(kv.ie,i,j,k) - rhsmdt * divdp_proj(kv.ie,i,j,k);
              });
          });
      });
  }

  void compute_qmin_qmax() {
    // Temporaries, due to issues capturing *this on device
    const int qsize = m_data.qsize;
    const Real rhs_multiplier = m_data.rhs_multiplier;
    const int n0_qdp = m_data.n0_qdp;
    const auto qdp = m_tracers.qdp;
    const auto dp = m_buffers.dp;
    const auto qtens_biharmonic = m_tracers.qtens_biharmonic;
    const auto qlim = m_tracers.qlim;
    // Adjoint taping (limiter_option==9 only; see euler_step_adj): record
    // qlim as it stood on entry to this call (qlim_pre_local -- only
    // meaningful when rhs_multiplier==1, where the running min/max is
    // seeded from the *previous* call's qlim rather than reset here), and
    // as it stands right after the local (per-element) reduction below, but
    // before any neighbor/MPI min-max reduction (qlim_pre_exchange). Both
    // are cheap [ne][qsize][2][NUM_LEV] copies, harmless when taping is off
    // (the branch is skipped entirely for other ST, and no Context lookup
    // happens at all) and required because qlim is mutated in place by
    // this function and by the subsequent neighbor exchange. This function
    // is host-only (no KOKKOS_INLINE_FUNCTION), so it is safe to fetch the
    // Context-held adjoint tape here directly and capture its (plain,
    // device-copyable) Views into the KOKKOS_LAMBDA below.
    bool tape = false;
    ExecViewManaged<RPT*[QSIZE_D][2][NUM_LEV]> adj_qlim_pre_local, adj_qlim_pre_exchange;
    if constexpr (std::is_same_v<ST,Real>) {
      tape = m_tape_for_adjoint;
      if (tape) {
        auto& adj_tape = get_adjoint_tape();
        adj_qlim_pre_local    = adj_tape.qlim_pre_local;
        adj_qlim_pre_exchange = adj_tape.qlim_pre_exchange;
      }
    }
    Kokkos::parallel_for(
      m_tv_policy,
      KOKKOS_LAMBDA (const TeamMember& team) {
        KernelVariables kv(team, qsize); // no team-idx used, so no need for TU
        const auto dp_t = Homme::subview(dp, kv.ie);
        const auto qdp_t = Homme::subview(qdp, kv.ie, n0_qdp, kv.iq);
        const auto qtens_biharmonic_t = Homme::subview(qtens_biharmonic, kv.ie, kv.iq);
        const auto qlim_t = Homme::subview(qlim, kv.ie, kv.iq);
        if constexpr (std::is_same_v<ST,Real>) {
        if (tape) {
          const auto pre_local_t = Homme::subview(adj_qlim_pre_local, kv.ie, kv.iq);
          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(kv.team, NUM_LEV),
            [&] (const int& k) {
              pre_local_t(0,k) = qlim_t(0,k);
              pre_local_t(1,k) = qlim_t(1,k);
            });
        }
        }
        if (rhs_multiplier != 1.0) {
          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(kv.team, NUM_LEV),
            [&] (const int& k) {
              const auto v = qdp_t(0,0,k) / dp_t(0,0,k);
              qtens_biharmonic_t(0,0,k) = v;
              qlim_t(0,k) = v;
              qlim_t(1,k) = v;
            });
        }
        for (int i = 0; i < NP; ++i)
          for (int j = 0; j < NP; ++j) {
            Kokkos::parallel_for(
              Kokkos::TeamThreadRange(kv.team, NUM_LEV),
              [&] (const int& k) {
                const auto v = qdp_t(i,j,k) / dp_t(i,j,k);
                qtens_biharmonic_t(i,j,k) = v;
                qlim_t(0,k) = min(qlim_t(0,k), v);
                qlim_t(1,k) = max(qlim_t(1,k), v);
              });
          }
        if constexpr (std::is_same_v<ST,Real>) {
        if (tape) {
          const auto pre_exch_t = Homme::subview(adj_qlim_pre_exchange, kv.ie, kv.iq);
          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(kv.team, NUM_LEV),
            [&] (const int& k) {
              pre_exch_t(0,k) = qlim_t(0,k);
              pre_exch_t(1,k) = qlim_t(1,k);
            });
        }
        }
      });
    Kokkos::fence();
  }

  void neighbor_minmax_start() {
    assert(m_mm_be->is_registration_completed());
    m_mm_be->pack_and_send_min_max();
  }

  void neighbor_minmax_finish() {
    m_mm_be->recv_and_unpack_min_max();
  }

  void minmax_and_biharmonic() {
    neighbor_minmax_start();
    compute_biharmonic_pre();
    m_mmqb_be->exchange(m_geometry.m_rspheremp);
    compute_biharmonic_post();
    neighbor_minmax_finish();
  }

  void neighbor_minmax() {
    assert(m_mm_be->is_registration_completed());
    m_mm_be->exchange_min_max();
  }

  void exchange_qdp_dss_var () {
    GPTLstart("eus_bexch");
    const int idx = 3*m_data.np1_qdp + static_cast<int>(m_data.DSSopt);
    m_bes[idx]->exchange(m_geometry.m_rspheremp);
    GPTLstop("eus_bexch");
  }

  void euler_step(const int np1_qdp, const int n0_qdp, const Real dt,
                  const Real rhs_multiplier, const DSSOption DSSopt) {

    m_data.n0_qdp         = n0_qdp;
    m_data.np1_qdp        = np1_qdp;
    m_data.dt             = dt;
    m_data.rhs_multiplier = rhs_multiplier;
    m_data.DSSopt         = DSSopt;

    if (EulerStepFunctor::is_quasi_monotone(m_data.limiter_option)) {
      // when running lim8, we also need to limit the biharmonic, so that term
      // needs to be included in each euler step.  three possible algorithms
      // here:
      // most expensive:
      //   compute biharmonic (which also computes qmin/qmax) during all 3
      //   stages be sure to set rhs_viss=1 cost:  3 biharmonic steps with 3 DSS

      // cheapest:
      //   compute biharmonic (which also computes qmin/qmax) only on first
      //   stage be sure to set rhs_viss=3 reuse qmin/qmax for all following
      //   stages (but update based on local qmin/qmax) cost:  1 biharmonic
      //   steps with 1 DSS main concern: viscosity

      // compromise:
      //   compute biharmonic (which also computes qmin/qmax) only on last stage
      //   be sure to set rhs_viss=3
      //   compute qmin/qmax directly on first stage
      //   reuse qmin/qmax for 2nd stage stage (but update based on local
      //   qmin/qmax) cost:  1 biharmonic steps, 2 DSS

      //  NOTE  when nu_p=0 (no dissipation applied in dynamics to dp equation),
      //        we should apply dissipation to Q (not Qdp) to preserve Q=1
      //        i.e.  laplace(Qdp) ~  dp0 laplace(Q)
      //        for nu_p=nu_q>0, we need to apply dissipation to Q *diffusion_dp

      // initialize dp, and compute Q from Qdp(and store Q in Qtens_biharmonic)

      compute_dp();
      compute_qmin_qmax();
      if (m_data.rhs_multiplier == 0.0) {
        neighbor_minmax();
      } else if (m_data.rhs_multiplier == 2.0) {
        minmax_and_biharmonic();
      }
    }

    GPTLstart("tl-at adv-n-limit");
    advect_and_limit();
    GPTLstop("tl-at adv-n-limit");
    exchange_qdp_dss_var();
  }

private:

  KOKKOS_INLINE_FUNCTION
  void run_setup_phase (const KernelVariables& kv) const {
    compute_2d_advection_step(kv);
  }

  KOKKOS_INLINE_FUNCTION
  void run_tracer_phase (const KernelVariables& kv) const {
    compute_qtens(kv);
    kv.team_barrier();

    if (m_data.limiter_option == 8) {
      limiter_optim_iter_full(kv);
      kv.team_barrier();
    } else if (m_data.limiter_option == 9) {
      // Adjoint taping (limiter_option==9, Real only; see euler_step_adj):
      // record compute_qtens's raw output (m_adj_qtens_prelimiter_dev) and
      // the qlim entering the limiter (m_adj_qlim_final_dev -- "final" in
      // the sense that it is qlim as fully reduced by compute_qmin_qmax/the
      // neighbor exchange, and it is also the value the limiter's own
      // 0-floor and mass-relaxation logic will further, locally, adjust),
      // both BEFORE limiter_clip_and_sum mutates qtens_biharmonic/qlim in
      // place. m_adj_qtens_prelimiter_dev/m_adj_qlim_final_dev are a
      // device-reachable cache of two Views owned by the Context-held
      // EulerStepAdjointTape, refreshed by advect_and_limit() right before
      // this (`*this`-dispatched) kernel launches -- see their declaration
      // above for why this function can't fetch them from Context itself.
      if constexpr (std::is_same_v<ST,Real>) {
      if (m_tape_for_adjoint) {
        const auto ptens = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
        const auto tape_ptens = Homme::subview(m_adj_qtens_prelimiter_dev, kv.ie, kv.iq);
        const auto qlim = Homme::subview(m_tracers.qlim, kv.ie, kv.iq);
        const auto tape_qlim = Homme::subview(m_adj_qlim_final_dev, kv.ie, kv.iq);
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(kv.team, NP*NP),
          [&] (const int loop_idx) {
            const int i = loop_idx / NP;
            const int j = loop_idx % NP;
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
              [&] (const int& k) {
                tape_ptens(i,j,k) = ptens(i,j,k);
              });
          });
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(kv.team, NUM_LEV),
          [&] (const int& k) {
            tape_qlim(0,k) = qlim(0,k);
            tape_qlim(1,k) = qlim(1,k);
          });
        kv.team_barrier();
      }
      }
      limiter_clip_and_sum(kv);
      kv.team_barrier();
    }

    apply_spheremp(kv);
  }

  KOKKOS_INLINE_FUNCTION
  void compute_2d_advection_step (const KernelVariables& kv) const {
    const auto& c = m_data;
    const bool lim_quasi_monotone
      = EulerStepFunctor::is_quasi_monotone(c.limiter_option);
    const bool add_ps_diss = c.nu_p > 0 && c.rhs_viss != 0.0;
    const Real diss_fac = add_ps_diss ? -c.rhs_viss * c.dt * c.nu_q : 0;

    const auto& eta   = m_derived_state.m_eta_dot_dpdn;
    const auto& omega = m_derived_state.m_omega_p;
    const auto& divdp = m_derived_state.m_divdp_proj;
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP*NP),
      [&] (const int loop_idx) {
        const int i = loop_idx / NP;
        const int j = loop_idx % NP;
        auto f_dss_ptr = c.DSSopt==DSSOption::ETA
                           ? &eta.impl_map().reference(kv.ie, i,j, 0)
                           : (c.DSSopt==DSSOption::OMEGA
                                 ? &omega.impl_map().reference(kv.ie,i,j,0)
                                 : &divdp.impl_map().reference(kv.ie,i,j,0));
        ExecViewUnmanaged<PT[NUM_LEV]> f_dss (f_dss_ptr);
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
          [&] (const int& k) {
            const auto dp = m_buffers.dp(kv.ie,i,j,k);
            m_buffers.vstar(kv.ie,0,i,j,k) = m_derived_state.m_vn0(kv.ie,0,i,j,k) / dp;
            m_buffers.vstar(kv.ie,1,i,j,k) = m_derived_state.m_vn0(kv.ie,1,i,j,k) / dp;
            if (lim_quasi_monotone) {
              //! Note that the term dpdissk is independent of Q
              //! UN-DSS'ed dp at timelevel n0+1:
              m_buffers.dpdissk(kv.ie,i,j,k) = dp - c.dt * m_derived_state.m_divdp(kv.ie,i,j,k);
              if (add_ps_diss) {
                //! add contribution from UN-DSS'ed PS dissipation
                //!          dpdiss(:,:) = ( hvcoord%hybi(k+1) - hvcoord%hybi(k) ) *
                //!          elem(ie)%derived%psdiss_biharmonic(:,:)
                m_buffers.dpdissk(kv.ie,i,j,k) += diss_fac *
                  m_derived_state.m_dpdiss_biharmonic(kv.ie,i,j,k) / m_geometry.m_spheremp(kv.ie,i,j);
              }
            }
            //! also DSS extra field
            //! note: eta_dot_dpdn is actually dimension nlev+1, but nlev+1 data is
            //! all zero so we only have to DSS 1:nlev
            f_dss(k) *= m_geometry.m_spheremp(kv.ie,i,j);
          });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_qtens (const KernelVariables& kv) const {
    m_sphere_ops.divergence_sphere_update(
      kv, -m_data.dt, m_data.rhs_viss != 0.0,
      Homme::subview(m_buffers.vstar, kv.ie),
      Homme::subview(m_tracers.qdp, kv.ie, m_data.n0_qdp, kv.iq),
      // On input, qtens_biharmonic if add_hyperviscosity, undefined
      // if not; on output, qtens.
      Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq));
  }

  KOKKOS_INLINE_FUNCTION
  void limiter_optim_iter_full (const KernelVariables& kv) const {
    const auto sphweights = Homme::subview(m_geometry.m_spheremp, kv.ie);
    const auto dpmass = Homme::subview(m_buffers.dpdissk, kv.ie);
    const auto ptens = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
    const auto qlim = Homme::subview(m_tracers.qlim, kv.ie, kv.iq);
    if ( ! OnGpu<ExecSpace>::value && kv.team.team_size() == 1)
      SerialLimiter<ExecSpace>::run<8>(
        sphweights, dpmass, qlim, ptens,
        Homme::subview(m_sphere_ops.vector_buf_ml, kv.team_idx, 0));
    else
      limiter_optim_iter_full(kv.team, sphweights, dpmass, qlim, ptens);
  }

  KOKKOS_INLINE_FUNCTION
  void limiter_clip_and_sum (const KernelVariables& kv) const {
    const auto sphweights = Homme::subview(m_geometry.m_spheremp, kv.ie);
    const auto dpmass = Homme::subview(m_buffers.dpdissk, kv.ie);
    const auto ptens = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
    const auto qlim = Homme::subview(m_tracers.qlim, kv.ie, kv.iq);

    if ( ! OnGpu<ExecSpace>::value && kv.team.team_size() == 1)
      SerialLimiter<ExecSpace>::run<9>(
        sphweights, dpmass, qlim, ptens,
        Homme::subview(m_sphere_ops.vector_buf_ml, kv.team_idx, 0));
    else
      limiter_clip_and_sum(kv.team, sphweights, dpmass, qlim, ptens);
  }

  //! apply mass matrix, overwrite np1 with solution:
  //! dont do this earlier, since we allow np1_qdp == n0_qdp
  //! and we dont want to overwrite n0_qdp until we are done using it
  KOKKOS_INLINE_FUNCTION
  void apply_spheremp (const KernelVariables& kv) const {
    const auto qdp = Homme::subview(m_tracers.qdp, kv.ie, m_data.np1_qdp, kv.iq);
    const auto qtens = Homme::subview(m_tracers.qtens_biharmonic, kv.ie, kv.iq);
    const auto spheremp = Homme::subview(m_geometry.m_spheremp, kv.ie);
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP * NP),
      [&] (const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
          [&] (const int& ilev) {
            qdp(igp, jgp, ilev) = spheremp(igp, jgp) * qtens(igp, jgp, ilev);
          });
      });
  }

  // Do all the setup and teardown associated with a limiter. Call a limiter
  // functor to do the actual math given the problem data (mass, minp, maxp, c,
  // x), where the limiter possibly alters x to place it in the constraint set
  //    {x: (i) minp <= x_k <= maxp and (ii) c'x = mass }.
  template <typename Limit, typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl>
  KOKKOS_INLINE_FUNCTION static void
  with_limiter_shell (const TeamMember& team, const Limit& limit,
                      const ArrayGll& sphweights, const ArrayGllLvl& dpmass,
                      const Array2Lvl& qlim, const ArrayGllLvl& ptens) {
    const int NP2 = NP * NP;

    // Size doesn't matter; just need to get a pointer to the start of the
    // shared memory.
    ST* const team_data = Memory<ExecSpace>::get_shmem<ST>(team);

    const auto f = [&] (const int ilev) {
      const int vpi = ilev / VECTOR_SIZE, vsi = ilev % VECTOR_SIZE;

      ST* const data = team_data ?
      team_data + 2 * NP2 * team.team_rank() :
      nullptr;
      Memory<ExecSpace>::AutoArray<ST, NP2> x(data), c(data + NP2);

      Dispatch<>::parallel_for_NP2(team, [&] (const int& k) {
          const int i = k / NP, j = k % NP;
          const auto& dpm = dpmass(i,j,vpi)[vsi];
          c[k] = sphweights(i,j)*dpm;
          x[k] = ptens(i,j,vpi)[vsi]/dpm;
        });

      TwoT<ST> sums;
      Dispatch<>::parallel_reduce_NP2(team, [&] (const int& k, TwoT<ST>& sums) {
          sums.v[0] += x[k]*c[k];
          sums.v[1] += c[k];
        }, sums);
      if (sums.v[1] <= 0) return; //! this should never happen, but if it does, dont limit

      ST minp = qlim(0,vpi)[vsi], maxp = qlim(1,vpi)[vsi];
      // This is a slightly different spot than where this comment came from,
      // but it's logically equivalent to do it here.
      //! IMPOSE ZERO THRESHOLD.  do this here so it can be turned off for
      //! testing
      if (minp < 0)
        minp = qlim(0,vpi)[vsi] = 0;

      //! relax constraints to ensure limiter has a solution:
      //! This is only needed if running with the SSP CFL>1 or
      //! due to roundoff errors
      // This is technically a write race condition, but the same value is
      // being written, so it doesn't matter.
      if (sums.v[0] < minp*sums.v[1])
        minp = qlim(0,vpi)[vsi] = sums.v[0]/sums.v[1];
      if (sums.v[0] > maxp*sums.v[1])
        maxp = qlim(1,vpi)[vsi] = sums.v[0]/sums.v[1];

      const bool modified =
      limit(team, sums.v[0], minp, maxp, x.data(), c.data());

      if (modified)
        Dispatch<>::parallel_for_NP2(team, [&] (const int& k) {
            const int i = k / NP, j = k % NP;
            ptens(i,j,vpi)[vsi] = x[k]*dpmass(i,j,vpi)[vsi];
          });
    };

    if (OnGpu<ExecSpace>::value || team.team_size() > 1) {
      Kokkos::parallel_for (
        Kokkos::TeamThreadRange(team, NUM_PHYSICAL_LEV),
        f);
    } else {
VECTOR_SIMD_LOOP
      for (int ilev = 0; ilev < NUM_PHYSICAL_LEV; ++ilev)
        f(ilev);
    }
  }

public: // Expose for unit testing.

  // limiter_option = 8.
  template <typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl>
  KOKKOS_INLINE_FUNCTION static void
  limiter_optim_iter_full (const TeamMember& team,
                           const ArrayGll& sphweights, const ArrayGllLvl& dpmass,
                           const Array2Lvl& qlim, const ArrayGllLvl& ptens) {
    struct Limit {
      KOKKOS_INLINE_FUNCTION bool
      operator() (const TeamMember& team, const ST& mass,
                  const ST& minp, const ST& maxp,
                  ST* KOKKOS_RESTRICT const x,
                  ST const* KOKKOS_RESTRICT const c) const {
        const int maxiter = NP*NP - 1;
        const Real tol_limiter = 5e-14;

        for (int iter = 0; iter < maxiter; ++iter) {
         ST addmass = 0;
          Dispatch<>::parallel_reduce_NP2(team, [&] (const int& k, ST& addmass) {
              ST delta = 0;
              if (x[k] > maxp) {
                delta = x[k] - maxp;
                x[k] = maxp;
              } else if (x[k] < minp) {
                delta = x[k] - minp;
                x[k] = minp;
              }
              addmass += delta*c[k];
            }, addmass);

          if (std::abs(addmass) <= tol_limiter*std::abs(mass))
            break;

          if (addmass > 0) {
            ST weightssum = 0;
            Dispatch<>::parallel_reduce_NP2(team, [&] (const int& k, ST& weightssum) {
                if (x[k] < maxp)
                  weightssum += c[k];
              }, weightssum);
            const auto adw = addmass/weightssum;
            Dispatch<>::parallel_for_NP2(team, [&] (const int& k) {
                x[k] += (x[k] < maxp) ? adw : ST(0);
              });
          } else {
            ST weightssum = 0;
            Dispatch<>::parallel_reduce_NP2(team, [&] (const int& k, ST& weightssum) {
                if (x[k] > minp)
                  weightssum += c[k];
              }, weightssum);
            const auto adw = addmass/weightssum;
            Dispatch<>::parallel_for_NP2(team, [&] (const int& k) {
                x[k] += (x[k] > minp) ? adw : ST(0);
              });
          }
        }
        return true;
      }
    };

    with_limiter_shell(team, Limit(), sphweights, dpmass, qlim, ptens);
  }

  // This is limiter_option = 9 in ACME master.
  template <typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl>
  KOKKOS_INLINE_FUNCTION static void
  limiter_clip_and_sum (const TeamMember& team,
                        const ArrayGll& sphweights, const ArrayGllLvl& dpmass,
                        const Array2Lvl& qlim, const ArrayGllLvl& ptens) {
    struct Limit {
      KOKKOS_INLINE_FUNCTION bool
      operator() (const TeamMember& team, const ST& /* mass */,
                  const ST& minp, const ST& maxp,
                  ST* KOKKOS_RESTRICT const x,
                  ST const* KOKKOS_RESTRICT const c) const {
        // Clip.
        TwoT<ST> reds;
        Dispatch<>::parallel_reduce_NP2(team, [&] (const int& k, TwoT<ST>& reds) {
            ST delta = 0;
            if (x[k] > maxp) {
              delta = x[k] - maxp;
              x[k] = maxp;
              reds.v[1] += 1;
            } else if (x[k] < minp) {
              delta = x[k] - minp;
              x[k] = minp;
              reds.v[1] += 1;
            }
            reds.v[0] += delta*c[k];
          }, reds);
        if (reds.v[0] == 0) return false;
        const ST addmass = reds.v[0];

        if (addmass > 0) {
          ST fac = 0;
          // Get sum of weights. Don't store them; we don't want another array.
          Dispatch<>::parallel_reduce_NP2(team, [&] (const int& k, ST& fac) {
              fac += c[k]*(maxp - x[k]);
            }, fac);
          if (fac > 0) {
            // Update.
            fac = addmass/fac;
            Dispatch<>::parallel_for_NP2(team, [&] (const int& k) {
                x[k] += fac*(maxp - x[k]);
              });
          }
        } else {
          ST fac = 0;
          Dispatch<>::parallel_reduce_NP2(team, [&] (const int& k, ST& fac) {
              fac += c[k]*(x[k] - minp);
            }, fac);
          if (fac > 0) {
            fac = addmass/fac;
            Dispatch<>::parallel_for_NP2(team, [&] (const int& k) {
                x[k] += fac*(x[k] - minp);
              });
          }
        }
        return true;
      }
    };

    with_limiter_shell(team, Limit(), sphweights, dpmass, qlim, ptens);
  }

public:
  // ============================================================
  // Adjoint of euler_step (battleplan Step 4). Real-only (mirrors
  // HyperviscosityFunctorImplST::run_JtV and
  // ForcingFunctor::states_forcing_adj); a pure state-adjoint (lambda-in,
  // lambda-out) propagator with no Theta/dJ-dF-shaped output (Theta -- the
  // NN weights -- never appears in this functor).
  //
  // Preconditions (documented, not all enforced):
  //  - limiter_option must be 9 (limiter_clip_and_sum); anything else
  //    (including the legacy-preqx-only limiter_option==8) errors out.
  //  - If rhs_multiplier==2 (the biharmonic-mixing stage), nu_p must be 0
  //    and consthv (hypervis_scaling==0) must be true; the dpdiss_ave
  //    scaling (bilinear in state) and tensor-viscosity laplace_tensor
  //    (self-adjointness unverified -- see HyperviscosityFunctorImpl's own
  //    caveat) are out of scope and error out rather than silently
  //    producing a wrong or approximate gradient.
  //  - Must be called with the SAME (np1_qdp,n0_qdp,dt,rhs_multiplier,
  //    DSSopt) as the euler_step() call being differentiated, which must
  //    have been run with set_tape_for_adjoint(true) beforehand, and
  //    nothing since that call may have run another euler_step (or
  //    otherwise touched m_buffers.{dp,dpdissk,vstar}, m_tracers.qlim, or
  //    the adjoint tape views) -- euler_step_adj reads those buffers
  //    directly as its "tape", rather than re-snapshotting them, exactly
  //    as prim_advance_adj's ttype10_imex_adjoint relies on the
  //    just-completed forward trajectory's own checkpoints.
  //  - adj_tracers/adj_derived must be the SAME objects across repeated
  //    calls (the lazily-built adjoint boundary-exchange objects bind to
  //    their specific field Views on the first call).
  //
  // Adjoint state convention (mirrors states_forcing_adj: the caller reads
  // seeds from, and results into, the SAME slots the forward state used):
  //  - adj_tracers.qdp(np1_qdp,...): IN on entry (seed = dJ/d state after
  //    euler_step), consumed/zeroed here (apply_spheremp overwrites, not
  //    accumulates, so nothing should remain to double-count if the same
  //    slot is reused by a later call).
  //  - adj_tracers.qdp(n0_qdp,...): OUT, ACCUMULATED (+=) into (not
  //    overwritten), so repeated/chained calls compose correctly, exactly
  //    like the forward code's own "we allow np1_qdp==n0_qdp" aliasing.
  //  - adj_tracers.qlim(...): IN/OUT. On entry, any downstream dependency
  //    on qlim as it stood *before* this euler_step call (relevant only
  //    when rhs_multiplier==1, which reuses/refines a previous call's
  //    qlim rather than resetting it -- see compute_qmin_qmax). On exit,
  //    ACCUMULATED with this call's own contribution to that same
  //    quantity. The caller (a future multi-stage orchestrator, out of
  //    scope for this step) is responsible for chaining this across the
  //    (rhs_multiplier==0,1,2) call sequence, exactly as it must already
  //    do for qdp via the n0_qdp/np1_qdp slot indices.
  //  - adj_tracers.qtens_biharmonic: used purely as internal scratch
  //    (reused across the several stages below); callers should not rely
  //    on its value before or after the call.
  //  - adj_derived.{m_dp,m_divdp,m_vn0,m_dpdiss_biharmonic}: OUT,
  //    accumulated (+=).
  //  - adj_derived.{m_eta_dot_dpdn,m_omega_p,m_divdp_proj}: IN/OUT for
  //    whichever one DSSopt selects (the other two untouched); IN = seed
  //    on the post-DSS value, OUT = seed transformed back through the
  //    DSS-prep+exchange, further accumulated with compute_dp's own
  //    contribution to m_divdp_proj if that's the one DSSopt selects.
  // ============================================================
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  euler_step_adj (const int np1_qdp, const int n0_qdp, const Real dt,
                   const Real rhs_multiplier, const DSSOption DSSopt,
                   TracersST<Real>& adj_tracers,
                   ElementsDerivedStateST<Real>& adj_derived)
  {
    EKAT_REQUIRE_MSG(EulerStepFunctorST<ST>::is_quasi_monotone(m_data.limiter_option),
      "[euler_step_adj] Error! Adjoint is only implemented when euler_step "
      "used a quasi-monotone limiter (limiter_option 8 or 9).\n");
    EKAT_REQUIRE_MSG(m_data.limiter_option == 9,
      std::string("[euler_step_adj] Error! Adjoint is only implemented for ") +
      "limiter_option==9 (limiter_clip_and_sum); limiter_option==8 " +
      "(limiter_optim_iter_full) is legacy-preqx-only and out of scope. Got " +
      "limiter_option=" + std::to_string(m_data.limiter_option) + ".\n");
    if (rhs_multiplier == 2.0) {
      EKAT_REQUIRE_MSG(m_data.nu_p == 0,
        "[euler_step_adj] Error! Adjoint of the biharmonic-mixing branch "
        "(rhs_multiplier==2) is only implemented for nu_p==0: the "
        "dpdiss_adjustment scaling by dpdiss_ave is bilinear in state, "
        "and its adjoint (dJ/d(dpdiss_ave)) is not implemented.\n");
      EKAT_REQUIRE_MSG(m_data.consthv,
        "[euler_step_adj] Error! Adjoint of the biharmonic-mixing branch "
        "(rhs_multiplier==2) is only implemented for the const-viscosity "
        "path (hypervis_scaling==0); tensor viscosity (laplace_tensor) "
        "self-adjointness is unverified (see HyperviscosityFunctorImpl's "
        "own caveat about it) and out of scope here.\n");
    }

    init_adjoint_boundary_exchanges(adj_tracers, adj_derived);

    const int ne     = m_geometry.num_elems();
    const int qsize   = m_data.qsize;
    const auto spheremp  = m_geometry.m_spheremp;
    const auto rspheremp = m_geometry.m_rspheremp;
    const auto dinv       = m_geometry.m_dinv;
    const auto metdet     = m_geometry.m_metdet;
    const auto dvv         = m_deriv;
    const Real scale_factor_inv = m_sphere_ops.m_scale_factor_inv;
    const auto hvcoord = m_hvcoord;

    // Real-only Context-held adjoint scratch/tape (see
    // EulerStepAdjointTape.hpp). Fetched once here, then copied into local
    // named references below, exactly mirroring the existing pattern of
    // this function copying adj_tracers/adj_derived Views into locals
    // before every Kokkos::parallel_for -- that local-copy step already
    // has to happen regardless, so fetching from Context instead of
    // `this` costs nothing extra.
    auto& adj_tape = get_adjoint_tape();
    const auto adj_dp                     = adj_tape.dp;
    const auto adj_dpdissk                = adj_tape.dpdissk;
    const auto adj_vstar                  = adj_tape.vstar;
    const auto adj_qlim_final_grad        = adj_tape.qlim_final_grad;
    const auto adj_qlim_final_grad_summed = adj_tape.qlim_final_grad_summed;
    const auto adj_qtens_prelimiter       = adj_tape.qtens_prelimiter;
    const auto adj_qlim_final             = adj_tape.qlim_final;
    const auto adj_qlim_pre_exchange      = adj_tape.qlim_pre_exchange;
    const auto adj_qlim_pre_local         = adj_tape.qlim_pre_local;
    const auto& adj_bes                   = adj_tape.bes;
    const auto& adj_mmqb_be               = adj_tape.mmqb_be;

    // ---- 0. Zero the per-call adjoint scratch accumulators. ----
    Kokkos::deep_copy(adj_dp, Real(0));
    Kokkos::deep_copy(adj_dpdissk, Real(0));
    Kokkos::deep_copy(adj_vstar, Real(0));
    Kokkos::deep_copy(adj_qlim_final_grad, Real(0));

    // ---- 5. Reverse of exchange_qdp_dss_var(): self-adjoint DSS exchange,
    // applied directly to adj_tracers.qdp(np1_qdp,...) and whichever of
    // adj_derived's {eta_dot_dpdn,omega_p,divdp_proj} DSSopt selects (both
    // bundled in the same BoundaryExchangeST object, exactly mirroring how
    // m_bes[idx] bundles them forward). ----
    {
      const int idx = 3*np1_qdp + static_cast<int>(DSSopt);
      adj_bes[idx]->exchange(rspheremp);
    }
    Kokkos::fence();

    // ---- 4b-iii. Reverse of apply_spheremp: qdp(np1_qdp) = spheremp*qtens
    // (overwrite, not accumulate: np1_qdp's adjoint slot is fully consumed
    // -- zeroed -- here; qtens's adjoint (bufA, reusing
    // adj_tracers.qtens_biharmonic as scratch) starts fresh). ----
    {
      auto adj_qdp   = adj_tracers.qdp;
      auto adj_qtens = adj_tracers.qtens_biharmonic;
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>> pol({0,0,0,0,0},{ne,qsize,NP,NP,NUM_PHYSICAL_LEV});
      Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie,const int iq,const int i,const int j,const int lev) {
        const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
        const Real g = adj_qdp(ie,np1_qdp,iq,i,j,vpi)[vsi];
        adj_qdp(ie,np1_qdp,iq,i,j,vpi)[vsi] = 0;
        adj_qtens(ie,iq,i,j,vpi)[vsi] = spheremp(ie,i,j)*g;
      });
    }
    Kokkos::fence();

    // ---- 4b-ii. Reverse of limiter_clip_and_sum. Per (elem,tracer,level)
    // group of NP2=16 GLL points: replay the forward clip+redistribute
    // logic bit-exactly from taped/still-live data (adj_qtens_prelimiter,
    // adj_qlim_final, and the still-valid m_buffers.dpdissk/
    // m_geometry.m_spheremp -- no separate flag tape is needed since every
    // branch condition is exactly recomputable), then differentiate that
    // short, purely local sequence in reverse. See the design writeup
    // above euler_step_adj for the derivation. ----
    {
      auto adj_qtens    = adj_tracers.qtens_biharmonic; // bufA in, bufB out
      auto tape_ptens    = adj_qtens_prelimiter;          // ptens_in (pre-limiter)
      auto tape_qlimF    = adj_qlim_final;                // minp0/maxp0 (pre 0-floor/relax)
      auto dpdissk       = m_buffers.dpdissk;             // dpmass
      // adj_dpdissk itself is already a local (fetched once at the top of
      // this function from the Context-held adjoint tape); no re-copy
      // needed here, unlike tape_ptens/tape_qlimF/adj_qlimFgrad above/below
      // which give it a block-local name distinct from the outer one.
      auto adj_qlimFgrad = adj_qlim_final_grad;
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>> pol({0,0,0},{ne,qsize,NUM_PHYSICAL_LEV});
      Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie, const int iq, const int lev) {
        const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;

        // ---- Replay, bit-exact, forward's with_limiter_shell + Limit(). ----
        Real c[NP][NP], xpre[NP][NP], gout[NP][NP];
        Real sum0 = 0, sum1 = 0;
        for (int i=0;i<NP;++i) for (int j=0;j<NP;++j) {
          const Real dpm = dpdissk(ie,i,j,vpi)[vsi];
          c[i][j]    = spheremp(ie,i,j)*dpm;
          xpre[i][j] = tape_ptens(ie,iq,i,j,vpi)[vsi]/dpm;
          sum0 += xpre[i][j]*c[i][j];
          sum1 += c[i][j];
          gout[i][j] = adj_qtens(ie,iq,i,j,vpi)[vsi]; // read bufA now, before any writes
        }
        if (sum1 <= 0) return; // fwd: "this should never happen"; identity map (bufB==bufA already)

        const Real minp0 = tape_qlimF(ie,iq,0,vpi)[vsi];
        const Real maxp0 = tape_qlimF(ie,iq,1,vpi)[vsi];
        const bool clamped0 = minp0 < 0;
        Real minp = clamped0 ? Real(0) : minp0;
        Real maxp = maxp0;
        const bool relaxA = sum0 < minp*sum1;
        if (relaxA) minp = sum0/sum1;
        const bool relaxB = sum0 > maxp*sum1;
        if (relaxB) maxp = sum0/sum1;

        int  dir[NP][NP];
        Real xpost[NP][NP];
        Real addmass = 0;
        for (int i=0;i<NP;++i) for (int j=0;j<NP;++j) {
          Real d = 0;
          if (xpre[i][j] > maxp)      { d = xpre[i][j]-maxp; xpost[i][j]=maxp; dir[i][j]=+1; }
          else if (xpre[i][j] < minp) { d = xpre[i][j]-minp; xpost[i][j]=minp; dir[i][j]=-1; }
          else                        { xpost[i][j]=xpre[i][j];               dir[i][j]=0;  }
          addmass += d*c[i][j];
        }
        if (addmass == 0) return; // fwd: modified==false; identity map (bufB==bufA already)

        const bool pos = addmass > 0;
        Real fac_sum = 0;
        for (int i=0;i<NP;++i) for (int j=0;j<NP;++j)
          fac_sum += pos ? c[i][j]*(maxp-xpost[i][j]) : c[i][j]*(xpost[i][j]-minp);
        const bool redistrib = fac_sum > 0;
        const Real fac = redistrib ? addmass/fac_sum : Real(0);

        // ---- Reverse: steps 1-2 (ptens_out=x_out*dpmass; x_out from
        // clip+redistribute), producing gx[] (dJ/d post-clip x_k) and the
        // direct dpmass contribution. ----
        Real gx[NP][NP] = {{0}};
        Real g_fac = 0, g_minp = 0, g_maxp = 0;
        for (int i=0;i<NP;++i) for (int j=0;j<NP;++j) {
          const Real dpm = dpdissk(ie,i,j,vpi)[vsi];
          const Real xoutk = redistrib
            ? (pos ? (xpost[i][j] + fac*(maxp-xpost[i][j]))
                   : (xpost[i][j] + fac*(xpost[i][j]-minp)))
            : xpost[i][j];
          const Real g_xout = gout[i][j]*dpm; // dJ/d(x_out_k)
          Kokkos::atomic_add(&adj_dpdissk(ie,i,j,vpi)[vsi], gout[i][j]*xoutk);
          if (redistrib) {
            if (pos) {
              gx[i][j] += (1-fac)*g_xout;
              g_fac    += g_xout*(maxp - xpost[i][j]);
              g_maxp   += fac*g_xout;
            } else {
              gx[i][j] += (1+fac)*g_xout;
              g_fac    += g_xout*(xpost[i][j] - minp);
              g_minp   += -fac*g_xout;
            }
          } else {
            gx[i][j] += g_xout;
          }
        }

        // ---- Reverse: steps 3-4 (fac=addmass/fac_sum; fac_sum=sum c_k*(.)). ----
        Real g_addmass = 0;
        Real gc[NP][NP] = {{0}};
        if (redistrib) {
          g_addmass = g_fac/fac_sum;
          const Real g_fac_sum = -g_fac*addmass/(fac_sum*fac_sum);
          for (int i=0;i<NP;++i) for (int j=0;j<NP;++j) {
            if (pos) {
              gc[i][j] += g_fac_sum*(maxp - xpost[i][j]);
              g_maxp   += g_fac_sum*c[i][j];
              gx[i][j] += -g_fac_sum*c[i][j];
            } else {
              gc[i][j] += g_fac_sum*(xpost[i][j] - minp);
              g_minp   += -g_fac_sum*c[i][j];
              gx[i][j] += g_fac_sum*c[i][j];
            }
          }
        }

        // ---- Reverse: the clip itself (routes gx[] by branch; adds the
        // addmass/delta_k path, which only touches clipped points). ----
        Real gxp[NP][NP] = {{0}};
        for (int i=0;i<NP;++i) for (int j=0;j<NP;++j) {
          if (dir[i][j]==0) {
            gxp[i][j] += gx[i][j];
          } else if (dir[i][j]==+1) {
            g_maxp += gx[i][j];
          } else {
            g_minp += gx[i][j];
          }
          if (dir[i][j] != 0 && g_addmass != 0) {
            const Real bound   = (dir[i][j]==+1) ? maxp : minp;
            const Real delta_k = xpre[i][j] - bound;
            gxp[i][j] += g_addmass*c[i][j];
            gc[i][j]  += g_addmass*delta_k;
            if (dir[i][j]==+1) g_maxp += -g_addmass*c[i][j];
            else                g_minp += -g_addmass*c[i][j];
          }
        }

        // ---- Write bufB (dJ/d ptens_in, in mass units) and dpmass's
        // remaining (c_k- and x_preclip-routed) contributions. ----
        for (int i=0;i<NP;++i) for (int j=0;j<NP;++j) {
          const Real dpm = dpdissk(ie,i,j,vpi)[vsi];
          const Real gdpm = gc[i][j]*spheremp(ie,i,j) - gxp[i][j]*xpre[i][j]/dpm;
          Kokkos::atomic_add(&adj_dpdissk(ie,i,j,vpi)[vsi], gdpm);
          adj_qtens(ie,iq,i,j,vpi)[vsi] = gxp[i][j]/dpm;
        }

        // ---- Route dJ/d(minp)/dJ/d(maxp): per the battleplan's decision
        // #2, the 0-floor/relaxation branches are a documented, frozen
        // approximation -- if any fired, drop the corresponding gradient
        // rather than route it (not to qlim, not to ptens/dpmass via any
        // other path); if none fired, route it through cleanly. ----
        if (!(clamped0 || relaxA)) adj_qlimFgrad(ie,iq,0,vpi)[vsi] += g_minp;
        if (!relaxB)                adj_qlimFgrad(ie,iq,1,vpi)[vsi] += g_maxp;
      });
    }
    Kokkos::fence();

    // ---- 4b-i. Reverse of compute_qtens (divergence_sphere_update):
    // bilinear in (vstar,qdp), so both get product-rule treatment (see
    // prim_advance_adj's handling of CAAR's own nonlinear/bilinear terms
    // for the precedent this follows). Pure consumer of bufB (left in
    // adj_tracers.qtens_biharmonic by the limiter reverse above); if
    // add_hyperviscosity, that same bufB value (unmodified) is exactly the
    // seed the biharmonic-chain reverse (stage 3, below) needs, so nothing
    // further is written back here. ----
    {
      auto adj_qdp   = adj_tracers.qdp;
      auto adj_qtens = adj_tracers.qtens_biharmonic;
      // adj_vstar is already a local (fetched once at the top of this
      // function from the Context-held adjoint tape); no re-copy needed.
      auto vstar_fwd = m_buffers.vstar;
      auto qdp_fwd   = m_tracers.qdp;
      const Real alpha = -dt;
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>> pol({0,0,0},{ne,qsize,NUM_PHYSICAL_LEV});
      Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie, const int iq, const int lev) {
        const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
        Real g[NP][NP], h[NP][NP];
        for (int i=0;i<NP;++i) for (int j=0;j<NP;++j) {
          g[i][j] = adj_qtens(ie,iq,i,j,vpi)[vsi];
          adj_qdp(ie,n0_qdp,iq,i,j,vpi)[vsi] += g[i][j]; // direct "+qdp" term
          h[i][j] = g[i][j]*alpha*scale_factor_inv/metdet(ie,i,j);
        }
        Real adjgv0[NP][NP], adjgv1[NP][NP];
        for (int i=0;i<NP;++i) for (int k=0;k<NP;++k) {
          Real s = 0; for (int j=0;j<NP;++j) s += dvv(j,k)*h[i][j];
          adjgv0[i][k] = s;
        }
        for (int k=0;k<NP;++k) for (int j=0;j<NP;++j) {
          Real s = 0; for (int i=0;i<NP;++i) s += dvv(i,k)*h[i][j];
          adjgv1[k][j] = s;
        }
        for (int i=0;i<NP;++i) for (int j=0;j<NP;++j) {
          const Real d00 = dinv(ie,0,0,i,j), d10 = dinv(ie,1,0,i,j);
          const Real d01 = dinv(ie,0,1,i,j), d11 = dinv(ie,1,1,i,j);
          const Real md  = metdet(ie,i,j);
          const Real av0 = (adjgv0[i][j]*d00 + adjgv1[i][j]*d01)*md;
          const Real av1 = (adjgv0[i][j]*d10 + adjgv1[i][j]*d11)*md;
          const Real qdpv = qdp_fwd(ie,n0_qdp,iq,i,j,vpi)[vsi];
          const Real vs0  = vstar_fwd(ie,0,i,j,vpi)[vsi];
          const Real vs1  = vstar_fwd(ie,1,i,j,vpi)[vsi];
          Kokkos::atomic_add(&adj_vstar(ie,0,i,j,vpi)[vsi], av0*qdpv);
          Kokkos::atomic_add(&adj_vstar(ie,1,i,j,vpi)[vsi], av1*qdpv);
          adj_qdp(ie,n0_qdp,iq,i,j,vpi)[vsi] += av0*vs0 + av1*vs1;
        }
      });
    }
    Kokkos::fence();

    // ---- 3. Reverse of minmax_and_biharmonic's biharmonic-mixing half
    // (rhs_multiplier==2 only; nu_p==0/consthv==true enforced above). Each
    // factor (laplace_simple, the m_mmqb_be exchange, the rhs_viss diagonal
    // scale) is self-adjoint (same precedent as
    // HyperviscosityFunctorImplST::run_JtV), so the reverse is the same
    // factors run on the adjoint field, in reverse order. Consumes bufB
    // (left, unmodified, in adj_tracers.qtens_biharmonic by stage 4b-i
    // above) and leaves the result (g_bih) in that same view, consumed
    // directly by stage 2 below. ----
    if (rhs_multiplier == 2.0) {
      auto adj_qtb = adj_tracers.qtens_biharmonic;
      const Real f = -m_data.rhs_viss * dt * m_data.nu_q;
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>> polS({0,0,0,0,0},{ne,qsize,NP,NP,NUM_PHYSICAL_LEV});
      Kokkos::parallel_for(polS, KOKKOS_LAMBDA (const int ie,const int iq,const int i,const int j,const int lev) {
        const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
        adj_qtb(ie,iq,i,j,vpi)[vsi] *= (f*hvcoord.dp0(vpi)[vsi]/spheremp(ie,i,j));
      });
      Kokkos::fence();

      const auto sphere_ops = m_sphere_ops;
      const int qs = qsize;
      auto laplace_pass = [&] () {
        Kokkos::parallel_for(
          Homme::get_default_team_policy<ExecSpace>(ne*qs, m_tpref),
          KOKKOS_LAMBDA (const TeamMember& team) {
            KernelVariables kv(team, qs);
            const auto f = Homme::subview(adj_qtb, kv.ie, kv.iq);
            sphere_ops.laplace_simple(kv, f, f);
          });
        Kokkos::fence();
      };
      laplace_pass();
      adj_mmqb_be->exchange(rspheremp);
      laplace_pass();
      // adj_qtb (== adj_tracers.qtens_biharmonic) now holds g_bih.
    }

    // ---- 2. Reverse of compute_qmin_qmax + whichever neighbor/MPI min-max
    // exchange ran (rhs_multiplier==0 or 2), plus the unconditional,
    // ordinary linear dependency qtens_biharmonic(i,j,k)=v (separate from
    // the min/max routing -- only nonzero when rhs_multiplier==2, via
    // g_bih from stage 3 above). The min/max reduction adjoint needs *no*
    // new communication (min/max never modifies the winning value, only
    // selects it): compare the taped pre-reduction candidate(s) against
    // the post-reduction result, bit-exact, and route full gradient to
    // every matching candidate (documented mild over-counting on exact
    // ties, consistent with the project's kink-convention approach). ----
    const bool has_exchange = (rhs_multiplier == 0.0 || rhs_multiplier == 2.0);

    // ---- 2a. exchange_min_max's forward reduction is NOT a broadcast to
    // a shared group value: it is, for EVERY element ie independently,
    // final(ie) = reduce({ie} union ie's own direct 1-hop neighbors) --
    // each element reduces over its OWN neighbor list, which generally
    // differs from its neighbors' own lists (e.g. an edge-neighbor of ie
    // that is itself a corner-neighbor of a *third* element ie has no
    // connection to). So different elements' final values are, in
    // general, genuinely different -- there is no equivalence-class
    // "group" here to sum a seed across (that was wrong in an earlier
    // version of this fix; a full mesh is one connected component under
    // *transitive* adjacency, but the reduction is only ever 1-hop).
    // The correct adjoint is a SCATTER, not a sum: element ie's own
    // downstream seed (adj_qlim_final_grad(ie), from the limiter's
    // reverse above) must be routed entirely to whichever ONE of
    // {ie itself, ie's direct neighbors} actually produced ie's own
    // final(ie) -- bit-exact comparison against the pre-exchange tape,
    // same "route to every matching candidate on ties" convention used
    // elsewhere. Implemented as a direct host round trip over
    // Connectivity's own per-element connection list (mirrors
    // exchange_min_max's own pack/unpack grouping, but as a scatter into
    // the *winner's* accumulator instead of a local combine) rather than
    // a real BoundaryExchange, since this harness -- like the rest of
    // this adjoint -- only ever runs single-rank (a real multi-rank
    // version would need real MPI communication for the SHARED
    // connections, out of scope for this step). ----
    if (has_exchange) {
      auto& conn = Context::singleton().get<Connectivity>();
      EKAT_REQUIRE_MSG(conn.get_comm().size() == 1,
        "[euler_step_adj] Error! The adjoint of the cross-element "
        "min/max reduction (exchange_min_max) is only implemented here "
        "for a single-rank run (see the comment above); a genuine "
        "multi-rank version needs real MPI communication for the "
        "SHARED connections too, which is out of scope for this step.\n");
      auto grad_h = Kokkos::create_mirror_view(adj_qlim_final_grad);
      Kokkos::deep_copy(grad_h, adj_qlim_final_grad);
      auto preexch_h = Kokkos::create_mirror_view(adj_qlim_pre_exchange);
      Kokkos::deep_copy(preexch_h, adj_qlim_pre_exchange);
      auto final_h = Kokkos::create_mirror_view(adj_qlim_final);
      Kokkos::deep_copy(final_h, adj_qlim_final);
      auto sum_h = Kokkos::create_mirror_view(adj_qlim_final_grad_summed);
      Kokkos::deep_copy(sum_h, Real(0)); // scatter target, starts empty
      const auto h_ucon     = conn.get_h_ucon();
      const auto h_ucon_ptr = conn.get_h_ucon_ptr();
      for (int ie = 0; ie < ne; ++ie) {
        const int cbeg = h_ucon_ptr(ie), cend = h_ucon_ptr(ie+1);
        for (int iq=0; iq<qsize; ++iq)
          for (int vpi=0; vpi<NUM_LEV; ++vpi)
            for (int vsi=0; vsi<VECTOR_SIZE; ++vsi) {
              for (int b=0; b<2; ++b) { // b=0: min, b=1: max
                const Real fin  = final_h(ie,iq,b,vpi)[vsi];
                const Real seed = grad_h(ie,iq,b,vpi)[vsi];
                if (preexch_h(ie,iq,b,vpi)[vsi] == fin)
                  sum_h(ie,iq,b,vpi)[vsi] += seed;
                for (int ic = cbeg; ic < cend; ++ic) {
                  const auto& info = h_ucon(ic);
                  if (info.sharing == etoi(ConnectionSharing::MISSING)) continue;
                  EKAT_REQUIRE_MSG(info.sharing != etoi(ConnectionSharing::SHARED),
                    "[euler_step_adj] Error! Unexpected remote-rank "
                    "connection in a single-rank run.\n");
                  const int je = info.remote.lid;
                  if (preexch_h(je,iq,b,vpi)[vsi] == fin)
                    sum_h(je,iq,b,vpi)[vsi] += seed;
                }
              }
            }
      }
      Kokkos::deep_copy(adj_qlim_final_grad_summed, sum_h);
    }

    {
      auto adj_qdp      = adj_tracers.qdp;
      auto adj_qlim      = adj_tracers.qlim;
      auto g_bih_view     = adj_tracers.qtens_biharmonic;
      auto qlim_grad      = has_exchange ? adj_qlim_final_grad_summed
                                         : adj_qlim_final_grad;
      auto tape_preexch    = adj_qlim_pre_exchange;
      auto tape_prelocal   = adj_qlim_pre_local;
      auto qdp_fwd         = m_tracers.qdp;
      auto dp_fwd          = m_buffers.dp;
      // adj_dp is already a local (fetched once at the top of this
      // function from the Context-held adjoint tape); no re-copy needed.
      const bool has_bih      = (rhs_multiplier == 2.0);
      const bool reuse_prev   = (rhs_multiplier == 1.0);

      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>> pol({0,0,0},{ne,qsize,NUM_PHYSICAL_LEV});
      Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie, const int iq, const int lev) {
        const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;

        // has_exchange: qlim_grad is already the fully-resolved SCATTER
        // result computed in stage 2a above -- element ie's own seed if
        // it won its own 1-hop reduction, PLUS any other element's seed
        // for which ie was *that* element's winning neighbor. No further
        // gating against ie's own premin/finalmin here: that would
        // incorrectly drop the "ie was some other element's winner"
        // contribution whenever ie didn't also win its own reduction.
        // !has_exchange: no exchange ran, so this element's own local
        // seed applies directly (unchanged from before).
        const Real gmin_preexch = qlim_grad(ie,iq,0,vpi)[vsi];
        const Real gmax_preexch = qlim_grad(ie,iq,1,vpi)[vsi];

        const Real premin = tape_preexch(ie,iq,0,vpi)[vsi];
        const Real premax = tape_preexch(ie,iq,1,vpi)[vsi];
        if (reuse_prev) {
          const Real prelocalmin = tape_prelocal(ie,iq,0,vpi)[vsi];
          const Real prelocalmax = tape_prelocal(ie,iq,1,vpi)[vsi];
          if (prelocalmin == premin) adj_qlim(ie,iq,0,vpi)[vsi] += gmin_preexch;
          if (prelocalmax == premax) adj_qlim(ie,iq,1,vpi)[vsi] += gmax_preexch;
        }

        for (int i=0;i<NP;++i) for (int j=0;j<NP;++j) {
          const Real dpv = dp_fwd(ie,i,j,vpi)[vsi];
          const Real v   = qdp_fwd(ie,n0_qdp,iq,i,j,vpi)[vsi]/dpv;
          Real g = 0;
          if (v == premin) g += gmin_preexch;
          if (v == premax) g += gmax_preexch;
          if (has_bih) g += g_bih_view(ie,iq,i,j,vpi)[vsi];
          if (g != 0) {
            adj_qdp(ie,n0_qdp,iq,i,j,vpi)[vsi] += g/dpv;
            Kokkos::atomic_add(&adj_dp(ie,i,j,vpi)[vsi], -g*v/dpv);
          }
        }
      });
    }
    Kokkos::fence();

    // ---- 4a. Reverse of compute_2d_advection_step: the DSS-prep step
    // (f_dss *= spheremp -- self-adjoint diagonal, its exchange half
    // already reversed in stage 5 above), vstar=vn0/dp (elementwise
    // divide), and, since limiter_option==9 is required, the dpdissk
    // (UN-DSS'ed dp) computation. ----
    {
      auto eta    = adj_derived.m_eta_dot_dpdn;
      auto omega  = adj_derived.m_omega_p;
      auto divdpp = adj_derived.m_divdp_proj;
      const DSSOption dss = DSSopt;
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> polF({0,0,0,0},{ne,NP,NP,NUM_PHYSICAL_LEV});
      Kokkos::parallel_for(polF, KOKKOS_LAMBDA (const int ie,const int i,const int j,const int lev) {
        const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
        const Real s = spheremp(ie,i,j);
        if (dss == DSSOption::ETA)        eta(ie,i,j,vpi)[vsi]    *= s;
        else if (dss == DSSOption::OMEGA) omega(ie,i,j,vpi)[vsi]  *= s;
        else                                divdpp(ie,i,j,vpi)[vsi] *= s;
      });
      Kokkos::fence();

      auto adj_vn0        = adj_derived.m_vn0;
      // adj_dp/adj_vstar/adj_dpdissk are already locals (fetched once at
      // the top of this function from the Context-held adjoint tape); no
      // re-copy needed.
      auto adj_divdp       = adj_derived.m_divdp;
      auto adj_dpdiss_bih  = adj_derived.m_dpdiss_biharmonic;
      auto vstar_fwd       = m_buffers.vstar;
      auto dp_fwd           = m_buffers.dp;
      const bool add_ps_diss = (m_data.nu_p > 0 && m_data.rhs_viss != 0.0);
      const Real diss_fac    = add_ps_diss ? -m_data.rhs_viss*dt*m_data.nu_q : Real(0);

      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> pol({0,0,0,0},{ne,NP,NP,NUM_PHYSICAL_LEV});
      Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie,const int i,const int j,const int lev) {
        const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
        const Real dpv = dp_fwd(ie,i,j,vpi)[vsi];

        const Real gv0 = adj_vstar(ie,0,i,j,vpi)[vsi];
        const Real gv1 = adj_vstar(ie,1,i,j,vpi)[vsi];
        adj_vn0(ie,0,i,j,vpi)[vsi] += gv0/dpv;
        adj_vn0(ie,1,i,j,vpi)[vsi] += gv1/dpv;
        const Real vs0 = vstar_fwd(ie,0,i,j,vpi)[vsi];
        const Real vs1 = vstar_fwd(ie,1,i,j,vpi)[vsi];
        Real gdp = -gv0*vs0/dpv - gv1*vs1/dpv;

        const Real gdk = adj_dpdissk(ie,i,j,vpi)[vsi];
        gdp += gdk;
        adj_divdp(ie,i,j,vpi)[vsi] += -dt*gdk;
        if (add_ps_diss) {
          adj_dpdiss_bih(ie,i,j,vpi)[vsi] += gdk*diss_fac/spheremp(ie,i,j);
        }

        adj_dp(ie,i,j,vpi)[vsi] += gdp;
      });
    }
    Kokkos::fence();

    // ---- 1. Reverse of compute_dp: buf = dp - rhs_multiplier*dt*divdp_proj. ----
    {
      // adj_dp is already a local (fetched once at the top of this
      // function from the Context-held adjoint tape); no re-copy needed.
      auto adj_derived_dp   = adj_derived.m_dp;
      auto adj_divdp_proj   = adj_derived.m_divdp_proj;
      const Real rhsmdt = rhs_multiplier*dt;
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> pol({0,0,0,0},{ne,NP,NP,NUM_PHYSICAL_LEV});
      Kokkos::parallel_for(pol, KOKKOS_LAMBDA (const int ie,const int i,const int j,const int lev) {
        const int vpi = lev/VECTOR_SIZE, vsi = lev%VECTOR_SIZE;
        const Real g = adj_dp(ie,i,j,vpi)[vsi];
        adj_derived_dp(ie,i,j,vpi)[vsi]  += g;
        adj_divdp_proj(ie,i,j,vpi)[vsi]  += -rhsmdt*g;
      });
    }
    Kokkos::fence();
  }

  // Enable/disable adjoint taping for the *next* euler_step() call (Real
  // only). Must be set to true before the specific euler_step() call that
  // will later be passed to euler_step_adj; may be reset to false right
  // after (taping costs a handful of extra elementwise copies per call).
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  set_tape_for_adjoint (const bool tape) {
    if (tape) alloc_adjoint_tape();
    m_tape_for_adjoint = tape;
  }

private:
  // Key under which the Real-only Euler-step adjoint scratch/tape is
  // stashed in Context::singleton().any_map(), mirroring how
  // Tape<StateSnapshot> is stashed there under "imex_tape" for the
  // CAAR/DIRK adjoint (see EulerStepAdjointTape.hpp).
  static constexpr const char* s_adjoint_tape_key = "euler_step_adjoint_tape";

  // Fetch the Context-held adjoint tape, assuming it has already been
  // allocated (at the current num_elems/qsize) by alloc_adjoint_tape() --
  // i.e. that set_tape_for_adjoint(true) has already been called. Host-only
  // (this must never be called from a `*this`-dispatched device kernel; see
  // m_adj_qtens_prelimiter_dev/m_adj_qlim_final_dev's declaration above for
  // the one case, run_tracer_phase, where the adjoint tape is needed from
  // code that may run on a GPU device, and how that's handled instead).
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>, EulerStepAdjointTape&>
  get_adjoint_tape () const {
    auto& any_map = Context::singleton().any_map();
    auto it = any_map.find(s_adjoint_tape_key);
    EKAT_REQUIRE_MSG(it != any_map.end(),
      "[EulerStepFunctorImplST] Error! The Euler-step adjoint tape was "
      "requested before set_tape_for_adjoint(true) allocated it.\n");
    return std::any_cast<EulerStepAdjointTape&>(it->second);
  }

  // Ensures the Context-held adjoint tape exists and is sized for the
  // current (num_elems, qsize), (re)constructing it if not. This replaces
  // this class's old per-member allocate-if-wrong-size logic: since a
  // std::any-held struct can't be resized field by field the way individual
  // Views could, a size mismatch instead reconstructs and reassigns the
  // whole EulerStepAdjointTape entry, which reallocates all 9 of its Views
  // together -- equivalent net behavior, just at struct granularity.
  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  alloc_adjoint_tape () {
    const int ne = m_geometry.num_elems();
    const int qs = m_data.qsize;
    assert(qs >= 0); // reset() must have been called already
    auto& any_map = Context::singleton().any_map();
    auto it = any_map.find(s_adjoint_tape_key);
    if (it == any_map.end()) {
      any_map.try_emplace(s_adjoint_tape_key,
                           std::in_place_type<EulerStepAdjointTape>, ne, qs);
      return;
    }
    auto& tape = std::any_cast<EulerStepAdjointTape&>(it->second);
    if (static_cast<int>(tape.qtens_prelimiter.extent(0)) == ne &&
        static_cast<int>(tape.qtens_prelimiter.extent(1)) == qs) {
      return; // already sized correctly
    }
    it->second = EulerStepAdjointTape(ne, qs);
  }

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, Real>>
  init_adjoint_boundary_exchanges (TracersST<Real>& adj_tracers,
                                    ElementsDerivedStateST<Real>& adj_derived) {
    auto& adj_tape = get_adjoint_tape();
    if (adj_tape.bex_ready) return;
    auto bm_exchange = Context::singleton().get<MpiBuffersManagerMap>()[MPI_EXCHANGE];
    DSSOption dss_vars[3] = {DSSOption::ETA, DSSOption::OMEGA, DSSOption::DIV_VDP_AVE};
    for (int np1_qdp = 0, k = 0; np1_qdp < Q_NUM_TIME_LEVELS; ++np1_qdp) {
      for (auto dssi : dss_vars) {
        adj_tape.bes[k] = std::make_shared<BoundaryExchangeST<Real>>();
        BoundaryExchangeST<Real>& be = *adj_tape.bes[k];
        be.set_buffers_manager(bm_exchange);
        int num_mid = dssi==DSSOption::ETA ? 0 : 1;
        int num_int = 1 - num_mid;
        be.set_num_fields(0, 0, m_data.qsize+num_mid, num_int);
        be.register_field(adj_tracers.qdp, np1_qdp, m_data.qsize, 0);
        switch(dssi) {
          case DSSOption::ETA:
            be.register_field(adj_derived.m_eta_dot_dpdn);
            break;
          case DSSOption::OMEGA:
            be.register_field(adj_derived.m_omega_p);
            break;
          case DSSOption::DIV_VDP_AVE:
            be.register_field(adj_derived.m_divdp_proj);
            break;
        }
        be.registration_completed();
        ++k;
      }
    }

    adj_tape.mmqb_be = std::make_shared<BoundaryExchangeST<Real>>();
    adj_tape.mmqb_be->set_buffers_manager(bm_exchange);
    adj_tape.mmqb_be->set_num_fields(0, 0, m_data.qsize);
    adj_tape.mmqb_be->register_field(adj_tracers.qtens_biharmonic, m_data.qsize, 0);
    adj_tape.mmqb_be->registration_completed();

    adj_tape.bex_ready = true;
  }
};

// Code repetition results from needing BFB and slight differences between lim 8
// and 9 Fortran impls.
template <typename ExecSpace>
template <int limiter_option, typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl,
          typename Array2GllLvl>
KOKKOS_INLINE_FUNCTION void SerialLimiter<ExecSpace>
::run (const ArrayGll& sphweights, const ArrayGllLvl& idpmass,
       const Array2Lvl& iqlim, const ArrayGllLvl& iptens,
       const Array2GllLvl& irwrk)
{
  using ST = std::remove_const_t<std::remove_reference_t<decltype(idpmass(0,0,0)[0])>>;

#define forij for (int i = 0; i < NP; ++i) for (int j = 0; j < NP; ++j)
#define forlev for (int lev = 0; lev < NUM_PHYSICAL_LEV; ++lev)

  ViewUnmanaged<const ST[NP][NP][NUM_LEV*VECTOR_SIZE]>
    dpmass(&idpmass(0,0,0)[0]);
  ViewUnmanaged<ST[NP][NP][NUM_LEV*VECTOR_SIZE]>
    c(&irwrk(0,0,0,0)[0]);
  ViewUnmanaged<ST[2][NUM_LEV*VECTOR_SIZE]>
    qlim(&iqlim(0,0)[0]);

  if (limiter_option == 8) {
    ViewUnmanaged<ST[NP][NP][NUM_LEV*VECTOR_SIZE]>
      x(&iptens(0,0,0)[0]);
    ST mass[NUM_PHYSICAL_LEV] = {0}, sumc[NUM_PHYSICAL_LEV] = {0};

    forij {
      const auto& sphij = sphweights(i,j);
      VECTOR_SIMD_LOOP forlev {
        const auto& dpm = dpmass(i,j,lev);
        c(i,j,lev) = sphij*dpm;
        x(i,j,lev) /= dpm;
        mass[lev] += c(i,j,lev)*x(i,j,lev);
        sumc[lev] += c(i,j,lev);
      }
    }

    VECTOR_SIMD_LOOP forlev {
      if (qlim(0,lev) < 0)
        qlim(0,lev) = 0;
      if (mass[lev] < qlim(0,lev)*sumc[lev])
        qlim(0,lev) = mass[lev]/sumc[lev];
      if (mass[lev] > qlim(1,lev)*sumc[lev])
        qlim(1,lev) = mass[lev]/sumc[lev];
    }

    static const int maxiter = NP*NP - 1;
    static const Real tol_limiter = 5e-14;
    int donecnt = 0;
    char done[NUM_PHYSICAL_LEV] = {0};
    for (int iter = 0; iter < maxiter; ++iter) {
      ST addmass[NUM_PHYSICAL_LEV] = {0};

      forij {
        VECTOR_SIMD_LOOP forlev {
          auto& xij = x(i,j,lev);
          ST delta = 0;
          if (xij < qlim(0,lev)) {
            delta = xij - qlim(0,lev);
            xij = qlim(0,lev);
          } else if (xij > qlim(1,lev)) {
            delta = xij - qlim(1,lev);
            xij = qlim(1,lev);
          }
          addmass[lev] += delta*c(i,j,lev);
        }
      }

      forlev {
        if (std::abs(addmass[lev]) <= tol_limiter*std::abs(mass[lev]) &&
            ! done[lev]) {
          done[lev] = 1;
          ++donecnt;
        }
      }
      if (donecnt == NUM_PHYSICAL_LEV) break;

      ST f[NUM_PHYSICAL_LEV] = {0};
      forij {
        VECTOR_SIMD_LOOP forlev {
          if (done[lev]) continue;
          if (addmass[lev] <= 0) {
            if (x(i,j,lev) > qlim(0,lev))
              f[lev] += c(i,j,lev);
          } else {
            if (x(i,j,lev) < qlim(1,lev))
              f[lev] += c(i,j,lev);
          }
        }
      }

      VECTOR_SIMD_LOOP forlev {
        if (f[lev] != 0)
          f[lev] = addmass[lev] / f[lev];
      }

      forij {
        VECTOR_SIMD_LOOP forlev {
          if (done[lev]) continue;
          if (addmass[lev] <= 0) {
            if (x(i,j,lev) > qlim(0,lev))
              x(i,j,lev) += f[lev];
          } else {
            if (x(i,j,lev) < qlim(1,lev))
              x(i,j,lev) += f[lev];
          }
        }
      }
    }

    forij {
      VECTOR_SIMD_LOOP forlev {
        x(i,j,lev) *= dpmass(i,j,lev);
      }
    }
  } else if (limiter_option == 9) {
    ViewUnmanaged<ST[NP][NP][NUM_LEV*VECTOR_SIZE]>
      ptens(&iptens(0,0,0)[0]);
    ViewUnmanaged<ST[NP][NP][NUM_LEV*VECTOR_SIZE]>
      x(&irwrk(1,0,0,0)[0]);
    ST mass[NUM_PHYSICAL_LEV] = {0}, sumc[NUM_PHYSICAL_LEV] = {0};

    forij {
      const auto& sphij = sphweights(i,j);
      VECTOR_SIMD_LOOP forlev {
        const auto& dpm = dpmass(i,j,lev);
        c(i,j,lev) = sphij*dpm;
        x(i,j,lev) = ptens(i,j,lev) / dpm;
        mass[lev] += c(i,j,lev)*x(i,j,lev);
        sumc[lev] += c(i,j,lev);
      }
    }

    VECTOR_SIMD_LOOP forlev {
      if (qlim(0,lev) < 0)
        qlim(0,lev) = 0;
      if (mass[lev] < qlim(0,lev)*sumc[lev])
        qlim(0,lev) = mass[lev]/sumc[lev];
      if (mass[lev] > qlim(1,lev)*sumc[lev])
        qlim(1,lev) = mass[lev]/sumc[lev];
    }

    ST addmass[NUM_PHYSICAL_LEV] = {0};
    // This is here to be BFB with the Fortran impl. When BFB is no longer
    // required, it is almost certainly better to remove this variable and the
    // conditionals on it.
    int modified[NUM_PHYSICAL_LEV] = {0};

    forij {
      VECTOR_SIMD_LOOP forlev {
        auto& xij = x(i,j,lev);
        ST delta = 0;
        if (xij < qlim(0,lev)) {
          delta = xij - qlim(0,lev);
          xij = qlim(0,lev);
          modified[lev] = 1;
        } else if (xij > qlim(1,lev)) {
          delta = xij - qlim(1,lev);
          xij = qlim(1,lev);
          modified[lev] = 1;
        }
        if (modified[lev])
          addmass[lev] += delta*c(i,j,lev);
      }
    }

    ST f[NUM_PHYSICAL_LEV] = {0};
    forij {
      VECTOR_SIMD_LOOP forlev {
        auto& xij = x(i,j,lev);
        if (addmass[lev] <= 0) {
          if (xij > qlim(0,lev))
            f[lev] += c(i,j,lev)*(xij - qlim(0,lev));
        } else {
          if (xij < qlim(1,lev))
            f[lev] += c(i,j,lev)*(qlim(1,lev) - xij);
        }
      }
    }

    VECTOR_SIMD_LOOP forlev {
      if (f[lev] != 0)
        f[lev] = addmass[lev] / f[lev];
    }

    forij {
      VECTOR_SIMD_LOOP forlev {
        auto& xij = x(i,j,lev);
        if (addmass[lev] <= 0) {
          if (xij > qlim(0,lev))
            xij += f[lev]*(xij - qlim(0,lev));
        } else {
          if (xij < qlim(1,lev))
            xij += f[lev]*(qlim(1,lev) - xij);
        }
      }
    }

    forij {
      VECTOR_SIMD_LOOP forlev {
        if (modified[lev])
          ptens(i,j,lev) = x(i,j,lev) * dpmass(i,j,lev);
      }
    }
  } else {
    Kokkos::abort("Only limiter_option 8 and 9 is impl'ed.");
  }

# undef forlev
# undef forij
} // end SerialLimiter

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_IMPL_HPP
