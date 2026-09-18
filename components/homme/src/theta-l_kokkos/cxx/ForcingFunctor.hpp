/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Context.hpp"
#include "CamForcing.hpp"
#include "ColumnOps.hpp"
#include "ElementOps.hpp"
#include "ElementsForcing.hpp"
#include "ElementsGeometry.hpp"
#include "ElementsState.hpp"
#include "EquationOfState.hpp"
#include "FunctorsBuffersManager.hpp"
#include "Tracers.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "KernelVariables.hpp"
#include "profiling.hpp"

#include "StateSnapshot.hpp"

#include "utilities/BfbUtils.hpp"

#include <type_traits>

namespace Homme {

template<typename ST>
class ForcingFunctorST
{
  using PT = PackType<ST>;

public:

  struct TagStates {};
  struct TagStatesAdjoint {};
  struct TagTracersPre {};
  struct TagTracers {};
  struct TagTracersPost {};

  ForcingFunctorST ()
   : m_state(Context::singleton().get<ElementsStateST<ST>>())
   , m_forcing(Context::singleton().get<ElementsForcingST<ST>>())
   , m_geometry(Context::singleton().get<ElementsGeometry>())
   , m_tracers(Context::singleton().get<TracersST<ST>>())
   , m_hvcoord(Context::singleton().get<HybridVCoord>())
   , m_num_state_elems(m_state.num_elems())
   , m_num_tracer_elems(m_tracers.num_elems())
   , m_num_tracers(m_tracers.num_tracers())
   , is_setup(true)
   , m_policy_states(1,1)       // Need to init in constructor. Fix sizes in the body.
   , m_policy_tracers_pre(1,1)
   , m_policy_tracers(1,1)
   , m_policy_tracers_post(1,1)
   , m_tu_states(m_policy_states)
   , m_tu_tracers_pre(m_policy_tracers_pre)
   , m_tu_tracers(m_policy_tracers)
   , m_tu_tracers_post(m_policy_tracers_post)
   , m_lambda(0) // Placeholder; reassigned (shallow) by states_forcing_adj
  {
    // Check everything is init-ed
    assert (m_state.num_elems()>0);
    assert (m_forcing.num_elems()>0);
    assert (m_tracers.num_elems()>0);
    assert (m_geometry.num_elems()>0);
    assert (m_hvcoord.m_inited);

    init_params();

    // Create the team policy
    init_team_policies();
  }

  // This constructor is useful for using buffer functionality without
  // having all other Functor information available.
  // If this constructor is used, the setup() function must be called
  // before using any other ForcingFunctor functions.
  ForcingFunctorST (const int num_state_elems, const int num_tracer_elems, const int num_tracers)
    : m_hvcoord(Context::singleton().get<HybridVCoord>())
    , m_num_state_elems(num_state_elems)
    , m_num_tracer_elems(num_tracer_elems)
    , m_num_tracers(num_tracers)
    , is_setup(false)
    , m_policy_states(1,1)       // Need to init in constructor. Fix sizes in the body.
    , m_policy_tracers_pre(1,1)
    , m_policy_tracers(1,1)
    , m_policy_tracers_post(1,1)
    , m_tu_states(m_policy_states)
    , m_tu_tracers_pre(m_policy_tracers_pre)
    , m_tu_tracers(m_policy_tracers)
    , m_tu_tracers_post(m_policy_tracers_post)
    , m_lambda(0) // Placeholder; reassigned (shallow) by states_forcing_adj
  {
    assert (m_hvcoord.m_inited);

    init_params();

    // Create the team policy
    init_team_policies();
  }

  void init_params ()
  {
    const auto& c = Context::singleton();
    const auto& p = c.get<SimulationParams>();

    m_hydrostatic = p.theta_hydrostatic_mode;
    m_qsize = p.qsize;

    m_adjust_ps = (p.dt_remap_factor == 0);

    m_eos.init(m_hydrostatic,m_hvcoord);
    m_elem_ops.init(m_hvcoord);
  }

  void init_team_policies ()
  {
    m_policy_states = Homme::get_default_team_policy<ExecSpace,TagStates>(m_num_state_elems);
    m_policy_tracers_pre = Homme::get_default_team_policy<ExecSpace,TagTracersPre>(m_num_tracer_elems);
    m_policy_tracers = Homme::get_default_team_policy<ExecSpace,TagTracers>(m_num_tracer_elems*m_num_tracers);
    m_policy_tracers_post = Homme::get_default_team_policy<ExecSpace,TagTracersPost>(m_num_tracer_elems);

    m_tu_states       = TeamUtils<ExecSpace>(m_policy_states);
    m_tu_tracers_pre  = TeamUtils<ExecSpace>(m_policy_tracers_pre);
    m_tu_tracers      = TeamUtils<ExecSpace>(m_policy_tracers);
    m_tu_tracers_post = TeamUtils<ExecSpace>(m_policy_tracers_post);
  }

  bool setup_needed () { return !is_setup; }

  void setup ()
  {
    assert(!is_setup);

    const auto& c = Context::singleton();

    m_state    = c.get<ElementsStateST<ST>>();
    m_forcing  = c.get<ElementsForcingST<ST>>();
    m_geometry = c.get<ElementsGeometry>();
    m_tracers  = c.get<TracersST<ST>>();

    // Check everything is init-ed
    assert (m_state.num_elems()>0);
    assert (m_forcing.num_elems()>0);
    assert (m_tracers.num_elems()>0);
    assert (m_geometry.num_elems()>0);

    // Sanity checks
    assert(m_num_state_elems == m_state.num_elems());
    assert(m_num_tracer_elems == m_tracers.num_elems());
    assert(m_num_tracers == m_tracers.num_tracers());

    is_setup = true;
  }

  int requested_buffer_size () const {
    const int nslots = std::max(m_tu_tracers_pre.get_num_ws_slots(),
                                m_tu_tracers.get_num_ws_slots());
    // const int nslots = m_tu_tracers.get_num_ws_slots();
    const int nelems = m_num_state_elems;
    constexpr int mid_size = NP*NP*NUM_LEV*VECTOR_SIZE;
    constexpr int int_size = NP*NP*NUM_LEV_P*VECTOR_SIZE;

    int scl_sz = sizeof(ST) / sizeof(Real);

    // 3 persistent midlayers, 2 non-persistent midlayer, and 1 non-persistent interface
    return scl_sz*(mid_size*(nelems*4+nslots) + int_size*nslots);
  }

  void init_buffers (const FunctorsBuffersManager& fbm) {
    const int num_slots = std::max(m_tu_tracers_pre.get_num_ws_slots(),
                                   m_tu_tracers.get_num_ws_slots());

    constexpr int mid_size = NP*NP*NUM_LEV;
    constexpr int int_size = NP*NP*NUM_LEV_P;

    PT* mem = reinterpret_cast<PT*>(fbm.get_memory());

    m_tn1 = decltype(m_tn1)(mem,m_num_state_elems);
    mem += mid_size*m_num_state_elems;

    m_dp_adj = decltype(m_dp_adj)(mem,m_num_state_elems);
    mem += mid_size*m_num_state_elems;

    m_pnh = decltype(m_pnh)(mem,m_num_state_elems);
    mem += mid_size*m_num_state_elems;

    m_exner = decltype(m_exner)(mem,m_num_state_elems);
    mem += mid_size*m_num_state_elems;

    m_Rstar = decltype(m_Rstar)(mem,num_slots);
    mem += mid_size*num_slots;

    m_pi_i = decltype(m_pi_i)(mem,num_slots);
    mem += int_size*num_slots;

    int used = reinterpret_cast<Real*>(mem) - fbm.get_memory();
    if (used!=requested_buffer_size()) {
      printf("used: %d\n",used);
      printf("requested: %d\n",requested_buffer_size());
      printf("nelems: %d\n",m_num_state_elems);
      printf("nslots: %d\n",num_slots);
      printf("mid_size: %d\n",mid_size);
      printf("int_size: %d\n",int_size);
      throw std::runtime_error("wrong used size");
    }
  }

  void states_forcing (const Real dt, const int np1) {
    // The Functor needs to be fully setup to use this function
    assert (is_setup);

    m_dt = dt;
    m_np1 = np1;

    Kokkos::parallel_for("compute states forcing",m_policy_states,*this);
    Kokkos::fence();
  }

  // Adjoint of states_forcing. states_forcing is an exactly affine map of
  // (state(np1), forcing) -> state(np1) (a per-level += for v/w_i/vtheta_dp/
  // phinh_i, followed by an overwrite of w_i's surface level using a linear
  // combination of the *just-forced* horizontal velocity), so no state or
  // forcing values are needed here at all: the Jacobian is constant.
  //
  // On entry, lambda holds dJ/d(state(np1) after states_forcing ran); on
  // exit, it holds dJ/d(state(np1) before states_forcing ran). dJ_dforcing
  // accumulates (+=, not overwrites) dJ/d(fm), dJ/d(fvtheta), dJ/d(fphi).
  // fm, fvtheta and fphi are exactly the tendencies states_forcing itself
  // reads, so dJ_dforcing is deliberately just an ElementsForcingST<ST> (the
  // same type/shape as the forcing it differentiates) rather than a new
  // type. The caller owns its lifecycle (zero() it once, then accumulate
  // into it across calls); this function never zeroes it itself.
  void states_forcing_adj (const Real dt, const int np1,
                           StateSnapshot& lambda,
                           ElementsForcingST<ST>& dJ_dforcing) {
    // StateSnapshot's scalar type is fixed to Real; only that specialization
    // of ForcingFunctorST can be paired with it here. Extending this to a
    // Fad-typed ForcingFunctorST (used elsewhere for forward sensitivities)
    // would need a templated adjoint-state container instead of
    // StateSnapshot.
    static_assert(std::is_same<ST,Real>::value,
                  "states_forcing_adj requires ForcingFunctorST<Real>.");

    // The Functor needs to be fully setup to use this function
    assert (is_setup);

    m_dt = dt;
    m_np1 = np1;
    // Shallow copies: m_lambda/m_dJ_dforcing's Views are reseated to alias
    // lambda/dJ_dforcing's underlying (device-accessible) data, exactly
    // like m_state/m_forcing are aliased to the Context's data. A raw
    // pointer to lambda/dJ_dforcing would not be safe here, since *this is
    // copied by value into the parallel_for below (and, on a GPU build, to
    // the device), so anything reachable only via a host pointer would be
    // unusable inside operator().
    m_lambda = lambda;
    m_dJ_dforcing = dJ_dforcing;

    // Same index space as states_forcing (one entry per element), so reuse
    // m_policy_states'/m_tu_states' sizing rather than recomputing it; only
    // the work tag (which selects operator(TagStatesAdjoint,...) below) has
    // to differ from m_policy_states itself.
    Kokkos::TeamPolicy<ExecSpace,TagStatesAdjoint> policy_adj(
        m_policy_states.league_size(), m_policy_states.team_size(), m_policy_states.vector_length());
    policy_adj.set_chunk_size(1);
    Kokkos::parallel_for("adjoint of states forcing",policy_adj,*this);
    Kokkos::fence();
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagStates&, const TeamMember& team) const {
    constexpr int LAST_MID_PACK     = ColInfo<NUM_PHYSICAL_LEV>::LastPack;
    constexpr int LAST_MID_PACK_END = ColInfo<NUM_PHYSICAL_LEV>::LastPackEnd;
    constexpr int LAST_INT_PACK     = ColInfo<NUM_INTERFACE_LEV>::LastPack;
    constexpr int LAST_INT_PACK_END = ColInfo<NUM_INTERFACE_LEV>::LastPackEnd;

    KernelVariables kv(team, m_tu_states);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {

      const int igp  = idx / NP;
      const int jgp  = idx % NP;

      const auto fm_x = Homme::subviewConst(m_forcing.m_fm,kv.ie,0,igp,jgp);
      const auto fm_y = Homme::subviewConst(m_forcing.m_fm,kv.ie,1,igp,jgp);
      const auto fm_z = Homme::subviewConst(m_forcing.m_fm,kv.ie,2,igp,jgp);
      const auto fvtheta = Homme::subviewConst(m_forcing.m_fvtheta,kv.ie,igp,jgp);
      const auto fphi    = Homme::subviewConst(m_forcing.m_fphi,kv.ie,igp,jgp);

      auto u      = Homme::subview(m_state.m_v,kv.ie,m_np1,0,igp,jgp);
      auto v      = Homme::subview(m_state.m_v,kv.ie,m_np1,1,igp,jgp);
      auto w      = Homme::subview(m_state.m_w_i,kv.ie,m_np1,igp,jgp);
      auto vtheta = Homme::subview(m_state.m_vtheta_dp,kv.ie,m_np1,igp,jgp);
      auto phi    = Homme::subview(m_state.m_phinh_i,kv.ie,m_np1,igp,jgp);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {

        vtheta(ilev) += m_dt*fvtheta(ilev);
        phi(ilev) += m_dt*fphi(ilev);

        u(ilev) += m_dt*fm_x(ilev);
        v(ilev) += m_dt*fm_y(ilev);
        w(ilev) += m_dt*fm_z(ilev);
      });

      // Fix w at the surface
      w(LAST_INT_PACK)[LAST_INT_PACK_END] =
        (u(LAST_MID_PACK)[LAST_MID_PACK_END]*m_geometry.m_gradphis(kv.ie,0,igp,jgp) +
         v(LAST_MID_PACK)[LAST_MID_PACK_END]*m_geometry.m_gradphis(kv.ie,1,igp,jgp)) / PhysicalConstants::g;
    });
  }

  // Adjoint of operator()(TagStates,...), run in exact reverse order of the
  // forward stages there: first undo the w_i surface fix (an overwrite, so
  // it both deposits into lambda_u/lambda_v and cuts off lambda_w's incoming
  // path at that one level), then undo the per-level += of the forcing.
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagStatesAdjoint&, const TeamMember& team) const {
    constexpr int LAST_MID_PACK     = ColInfo<NUM_PHYSICAL_LEV>::LastPack;
    constexpr int LAST_MID_PACK_END = ColInfo<NUM_PHYSICAL_LEV>::LastPackEnd;
    constexpr int LAST_INT_PACK     = ColInfo<NUM_INTERFACE_LEV>::LastPack;
    constexpr int LAST_INT_PACK_END = ColInfo<NUM_INTERFACE_LEV>::LastPackEnd;

    KernelVariables kv(team, m_tu_states);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {

      const int igp  = idx / NP;
      const int jgp  = idx % NP;

      auto lambda_u      = Homme::subview(m_lambda.v,kv.ie,0,igp,jgp);
      auto lambda_v      = Homme::subview(m_lambda.v,kv.ie,1,igp,jgp);
      auto lambda_w      = Homme::subview(m_lambda.w_i,kv.ie,igp,jgp);
      auto lambda_vtheta = Homme::subview(m_lambda.vtheta_dp,kv.ie,igp,jgp);
      auto lambda_phi    = Homme::subview(m_lambda.phinh_i,kv.ie,igp,jgp);

      auto dJ_dfm_x      = Homme::subview(m_dJ_dforcing.m_fm,kv.ie,0,igp,jgp);
      auto dJ_dfm_y      = Homme::subview(m_dJ_dforcing.m_fm,kv.ie,1,igp,jgp);
      auto dJ_dfm_z      = Homme::subview(m_dJ_dforcing.m_fm,kv.ie,2,igp,jgp);
      auto dJ_dfvtheta   = Homme::subview(m_dJ_dforcing.m_fvtheta,kv.ie,igp,jgp);
      auto dJ_dfphi      = Homme::subview(m_dJ_dforcing.m_fphi,kv.ie,igp,jgp);

      // Reverse of the w_i surface fix. It *overwrote* (not accumulated)
      // that one level using the forced u,v at the last mid level, so:
      //  - lambda_w at that level deposits into lambda_u/lambda_v there,
      //  - lambda_w at that level is then zeroed (nothing upstream of the
      //    overwrite should see a gradient through it).
      const Real lambda_w_surf = lambda_w(LAST_INT_PACK)[LAST_INT_PACK_END];
      lambda_u(LAST_MID_PACK)[LAST_MID_PACK_END] +=
        lambda_w_surf*m_geometry.m_gradphis(kv.ie,0,igp,jgp) / PhysicalConstants::g;
      lambda_v(LAST_MID_PACK)[LAST_MID_PACK_END] +=
        lambda_w_surf*m_geometry.m_gradphis(kv.ie,1,igp,jgp) / PhysicalConstants::g;
      lambda_w(LAST_INT_PACK)[LAST_INT_PACK_END] = 0.0;

      // Reverse of the per-level accumulation: state += dt*forcing, so
      // dJ/d(forcing) += dt*lambda(state), while lambda(state) itself is
      // unchanged (d(state_out)/d(state_in) = 1 for a plain +=).
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        dJ_dfvtheta(ilev) += m_dt*lambda_vtheta(ilev);
        dJ_dfphi(ilev)    += m_dt*lambda_phi(ilev);
        dJ_dfm_x(ilev)    += m_dt*lambda_u(ilev);
        dJ_dfm_y(ilev)    += m_dt*lambda_v(ilev);
        dJ_dfm_z(ilev)    += m_dt*lambda_w(ilev);
      });
    });
  }

  void tracers_forcing (const Real dt, const int np1, const int np1_qdp, const bool adjustment, const bool use_moisture) {
    // The Functor needs to be fully setup to use this function
    assert (is_setup);

    m_dt = dt;
    m_np1 = np1;
    m_np1_qdp = np1_qdp;
    m_adjustment = adjustment;

    m_moist = use_moisture;

    Kokkos::parallel_for("temperature, NH perturb press, FQps",m_policy_tracers_pre,*this);
    Kokkos::fence();

    if (m_qsize > 0) {
      Kokkos::parallel_for("apply tracers forcing", m_policy_tracers,*this);
      Kokkos::fence();
    }

    Kokkos::parallel_for("update temperature, pressure and phi", m_policy_tracers_post,*this);
    Kokkos::fence();
  }

  KOKKOS_INLINE_FUNCTION
  ScalarValue compute_fqdt (const int& k,
                     const ExecViewUnmanaged<Scalar[NUM_LEV]>& fq,
                     const ExecViewUnmanaged<Scalar[NUM_LEV]>& qdp) const {
    const int ilev = k / VECTOR_SIZE;
    const int ivec = k % VECTOR_SIZE;
    ScalarValue fqdt = m_dt * fq(ilev)[ivec];
    const ScalarValue& qdp_s = qdp(ilev)[ivec];
    if (qdp_s + fqdt < 0.0 && fqdt < 0.0) {
      if (qdp_s < 0.0) {
        fqdt = 0.0;
      } else {
        fqdt = -qdp_s;
      }
    }
    return fqdt;
  }

  KOKKOS_INLINE_FUNCTION
  Scalar compute_fqdt_pack (const int& ilev,
                            const ExecViewUnmanaged<Scalar[NUM_LEV]>& fq,
                            const ExecViewUnmanaged<Scalar[NUM_LEV]>& qdp) const {
    Scalar fqdt = m_dt * fq(ilev);
    const Scalar& qdp_s = qdp(ilev);
    // NOTE: here is where masks for simd operations would be handy
    VECTOR_SIMD_LOOP
    for (int iv=0; iv<VECTOR_SIZE; ++iv) {
      if (qdp_s[iv] + fqdt[iv] < 0.0 && fqdt[iv] < 0.0) {
        if (qdp_s[iv] < 0.0) {
          fqdt[iv] = 0.0;
        } else {
          fqdt[iv] = -qdp_s[iv];
        }
      }
    }
    return fqdt;
  }

  KOKKOS_INLINE_FUNCTION
  void operator () (const TagTracersPre&, const TeamMember& team) const {
    KernelVariables kv(team, m_tu_tracers_pre);
    constexpr Real Rgas = PhysicalConstants::Rgas;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto dp = Homme::subview(m_state.m_dp3d,kv.ie,m_np1,igp,jgp);
      ScalarValue& ps = m_state.m_ps_v(kv.ie,m_np1,igp,jgp);

      // The hydrostatic pressure in compute_pnh_and_exner in EOS is only used
      // for the theta_hydrostatic_mode case. So only compute it then
      auto vtheta = Homme::subview(m_state.m_vtheta_dp,kv.ie,m_np1,igp,jgp);
      auto phinh  = Homme::subview(m_state.m_phinh_i,kv.ie,m_np1,igp,jgp);
      auto pnh    = Homme::subview(m_pnh,kv.ie,igp,jgp);
      auto exner  = Homme::subview(m_exner,kv.ie,igp,jgp);
      if (m_hydrostatic) {
        auto p_i = Homme::subview(m_pi_i,kv.team_idx,igp,jgp);
        m_elem_ops.compute_hydrostatic_p(kv,dp,p_i,pnh);
        m_eos.compute_exner(kv,pnh,exner);
      } else {
        m_eos.compute_pnh_and_exner(kv,vtheta,phinh,pnh,exner);
      }

      if (m_moist) {
        // Compute pprime=pnh-pi, store in pnh (this is 0 for hydrostatic mode)
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                             [&](const int ilev) {
          pnh(ilev) -= m_hvcoord.ps0*m_hvcoord.hybrid_am(ilev) +
                                 ps *m_hvcoord.hybrid_bm(ilev);
        });
      }

      // Compute Rstar
      auto Rstar = Homme::subview(m_Rstar,kv.team_idx,igp,jgp);
      m_elem_ops.get_R_star (kv, m_moist,
                             (m_moist ?
                              Homme::subview(m_tracers.Q,kv.ie,0,igp,jgp) :
                              // If not moist, qsize might be 0, so we can't use
                              // Q. Use Rstar as an unused argument in its
                              // place.
                              Rstar),
                             Rstar);

      // Compute temperature
      auto tn1 = Homme::subview(m_tn1,kv.ie,igp,jgp);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        tn1(ilev) = exner(ilev) * vtheta(ilev)*(Rgas/Rstar(ilev)) / dp(ilev);
      });

      if (m_moist) {
        auto fq = Homme::subview(m_tracers.fq,kv.ie,0,igp,jgp);
        auto q = Homme::subview(m_tracers.Q,kv.ie,0,igp,jgp);
        auto qdp = Homme::subview(m_tracers.qdp,kv.ie,m_np1_qdp,0,igp,jgp);
        auto dp_adj = Homme::subview(m_dp_adj,kv.ie,igp,jgp);
        if (m_adjustment) {
          ScalarValue added_mass = 0;
          Dispatch<ExecSpace>::parallel_reduce(
            kv.team, Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
            [&](const int &k, ScalarValue &accumulator) {
              const int ilev = k / VECTOR_SIZE;
              const int ivec = k % VECTOR_SIZE;
              accumulator += dp(ilev)[ivec]*(fq(ilev)[ivec]-q(ilev)[ivec]);
            },added_mass);
          Kokkos::single(Kokkos::PerThread(kv.team), [&]() {
            ps += added_mass;
          });
          if (!m_adjust_ps) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                                [&](const int ilev) {
              dp_adj(ilev) = dp(ilev) + dp(ilev)*(fq(ilev)-q(ilev));
            });
          }
        } else {
          ScalarValue ps_forcing = 0.0;
          Dispatch<ExecSpace>::parallel_reduce(
            kv.team, Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
            [&](const int &k, ScalarValue &accumulator) {
              accumulator += compute_fqdt(k,fq,qdp)/m_dt;
            },ps_forcing);
          Kokkos::single(Kokkos::PerThread(kv.team), [&]() {
            ps += ps_forcing*m_dt;
          });
          if (!m_adjust_ps) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                                [&](const int& ilev) {
              dp_adj(ilev) = dp(ilev) + compute_fqdt_pack(ilev,fq,qdp);
            });
          }
        }

        if (m_adjust_ps) {
          m_hvcoord.compute_dp_ref(kv,ps,dp_adj);
        }
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagTracers&, const TeamMember& team) const {
    KernelVariables kv(team, m_qsize, m_tu_tracers);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto qdp    = Homme::subview(m_tracers.qdp,kv.ie,m_np1_qdp,kv.iq,igp,jgp);
      auto fq     = Homme::subview(m_tracers.fq, kv.ie,kv.iq,igp,jgp);
      auto Q      = Homme::subview(m_tracers.Q, kv.ie,kv.iq,igp,jgp);
      auto dp     = Homme::subview(m_state.m_dp3d, kv.ie,m_np1,igp,jgp);
      auto dp_adj = Homme::subview(m_dp_adj, kv.ie,igp,jgp);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        Scalar& qdp_ilev = qdp(ilev);
        if (m_adjustment) {
          qdp_ilev = dp(ilev)*fq(ilev);
        } else {
          qdp_ilev += compute_fqdt_pack(ilev,fq,qdp);
        }

        // Update tracers concentration
        if (m_moist) {
          Q(ilev) = qdp_ilev/dp_adj(ilev);
        } else {
          Q(ilev) = qdp_ilev/dp(ilev);
        }
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagTracersPost&, const TeamMember& team) const {
    constexpr int LAST_INT_PACK     = ColInfo<NUM_INTERFACE_LEV>::LastPack;
    constexpr int LAST_INT_PACK_END = ColInfo<NUM_INTERFACE_LEV>::LastPackEnd;

    KernelVariables kv(team, m_tu_tracers_post);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // Compute Rstar
      auto Rstar = Homme::subview(m_Rstar,kv.team_idx,igp,jgp);
      m_elem_ops.get_R_star (kv, m_moist,
                             (m_moist ?
                              Homme::subview(m_tracers.Q,kv.ie,0,igp,jgp) :
                              Rstar),
                             Rstar);

      auto tn1    = Homme::subview(m_tn1,kv.ie,igp,jgp);
      auto pnh    = Homme::subview(m_pnh,kv.ie,igp,jgp);
      auto exner  = Homme::subview(m_exner,kv.ie,igp,jgp);
      auto dp     = Homme::subview(m_state.m_dp3d,kv.ie,m_np1,igp,jgp);
      auto dp_adj = Homme::subview(m_dp_adj,kv.ie,igp,jgp);
      auto ft     = Homme::subview(m_forcing.m_ft,kv.ie,igp,jgp);
      auto fphi   = Homme::subview(m_forcing.m_fphi,kv.ie,igp,jgp);
      const ScalarValue ps = m_state.m_ps_v(kv.ie,m_np1,igp,jgp);

      if (m_moist) {
        // Need to update dp, pnh and exner. Currently, pnh is storing pnh-pi
        if (m_adjust_ps) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                               [&](const int ilev) {
            dp(ilev) = dp_adj(ilev);

            pnh(ilev) += m_hvcoord.ps0*m_hvcoord.hybrid_am(ilev) +
                                   ps *m_hvcoord.hybrid_bm(ilev);
          });
        } else {
          // Compute hydrostatic p from dp. Store in exner, then add to pnh
          auto p_i = Homme::subview(m_pi_i,kv.team_idx,igp,jgp);
          m_elem_ops.compute_hydrostatic_p(kv,dp_adj,p_i,exner);
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                              [&](const int ilev) {
            pnh(ilev) += exner(ilev);
            dp(ilev) = dp_adj(ilev);
          });
        }

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                             [&](const int ilev) {
          using namespace PhysicalConstants;
#ifdef HOMMEXX_BFB_TESTING
          exner(ilev) = bfb_pow(pnh(ilev)/p0,kappa);
#else
          exner(ilev) = pow(pnh(ilev)/p0,kappa);
#endif
        });
      }

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {

        // Update temperature
        tn1(ilev) += m_dt*ft(ilev);

        // Compute theta, store in tn1.
        tn1(ilev) = (Rstar(ilev)/PhysicalConstants::Rgas)*tn1(ilev)*dp(ilev)/exner(ilev);
      });

      // Compute phi as fcn of new theta, store in fphi
      auto phi_i  = Homme::subview(m_state.m_phinh_i,kv.ie,m_np1,igp,jgp);
      fphi(LAST_INT_PACK)[LAST_INT_PACK_END] = phi_i(LAST_INT_PACK)[LAST_INT_PACK_END];
      auto integrand_provider = [&](const int ilev)->Scalar {
        return PhysicalConstants::Rgas*tn1(ilev)*exner(ilev)/pnh(ilev);
      };
      ColumnOps::column_scan_mid_to_int<false>(kv,integrand_provider,fphi);

      // Finally, update forcing for theta and phi_i
      auto vtheta = Homme::subview(m_state.m_vtheta_dp,kv.ie,m_np1,igp,jgp);
      auto ftheta = Homme::subview(m_forcing.m_fvtheta,kv.ie,igp,jgp);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        ftheta(ilev) = (tn1(ilev) - vtheta(ilev)) / m_dt;
        fphi(ilev) = (fphi(ilev) - phi_i(ilev)) / m_dt;
      });

      // If NUM_LEV=NUM_LEV_P, the code above already took care of the last level
      if (NUM_LEV!=NUM_LEV_P) {
        // Last level
        Kokkos::single(Kokkos::PerThread(kv.team),[&](){
          fphi(LAST_INT_PACK)[LAST_INT_PACK_END] = (fphi(LAST_INT_PACK)[LAST_INT_PACK_END] - phi_i(LAST_INT_PACK)[LAST_INT_PACK_END]) / m_dt;
        });
      }
    });
  }

private:

  ElementsStateST<ST>     m_state;
  ElementsForcingST<ST>   m_forcing;
  ElementsGeometry        m_geometry;
  TracersST<ST>           m_tracers;

  HybridVCoord      m_hvcoord;
  ElementOps        m_elem_ops;
  EquationOfState<> m_eos;

  const int m_num_state_elems;
  const int m_num_tracer_elems;
  const int m_num_tracers;

  bool is_setup;

  ExecViewUnmanaged<PT*[NP][NP][NUM_LEV]>   m_Rstar;
  ExecViewUnmanaged<PT*[NP][NP][NUM_LEV]>   m_dp_adj;
  ExecViewUnmanaged<PT*[NP][NP][NUM_LEV]>   m_pnh;
  ExecViewUnmanaged<PT*[NP][NP][NUM_LEV_P]> m_pi_i;
  ExecViewUnmanaged<PT*[NP][NP][NUM_LEV]>   m_exner;
  ExecViewUnmanaged<PT*[NP][NP][NUM_LEV]>   m_tn1;

  int m_qsize;
  int m_np1;
  int m_np1_qdp;
  bool m_moist;
  bool m_hydrostatic;
  bool m_adjustment;
  bool m_adjust_ps;
  Real m_dt;

  Kokkos::TeamPolicy<ExecSpace,TagStates>        m_policy_states;
  Kokkos::TeamPolicy<ExecSpace,TagTracersPre>    m_policy_tracers_pre;
  Kokkos::TeamPolicy<ExecSpace,TagTracers>       m_policy_tracers;
  Kokkos::TeamPolicy<ExecSpace,TagTracersPost>   m_policy_tracers_post;
  TeamUtils<ExecSpace> m_tu_states, m_tu_tracers_pre, m_tu_tracers, m_tu_tracers_post;

  // Shallow aliases (reassigned by states_forcing_adj to whatever
  // lambda/dJ_dforcing were passed in for that call) rather than pointers,
  // since *this is copied by value to the device for the parallel_for launch
  // -- see states_forcing_adj for details.
  StateSnapshot m_lambda;
  ElementsForcingST<ST> m_dJ_dforcing;
};

using ForcingFunctor = ForcingFunctorST<ScalarValue>;

} // namespace Homme
