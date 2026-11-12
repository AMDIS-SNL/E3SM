#include "pyhommexx.hpp"
#include "pyhommexx_utils.hpp"

#include "Context.hpp"
#include "ElementsState.hpp"
#include "SimulationParams.hpp"
#include "StateSnapshot.hpp"
#include "Tape.hpp"
#include "TimeLevel.hpp"
#include "Types.hpp"
#include "prim_advance_exp.hpp"

// Note: prim_step_adj replays the fwd run with the Real-typed objects of the Context
#ifdef HOMMEXX_ENABLE_FAD_TYPES
#include "prim_advance_adj.hpp"
#include "prim_step_adj.hpp"
#endif

#include <ekat_assert.hpp>
#include <ekat_pack_kokkos.hpp>

#include <nanobind/ndarray.h>

namespace pyhommexx {

using namespace Homme;

namespace {

constexpr const char* adj_state_key = "pyhommexx_adj_state";

#ifdef HOMMEXX_ENABLE_FAD_TYPES
constexpr const char* adj_qdp_key = "pyhommexx_adj_qdp";
constexpr const char* prim_step_ckpt_key = "pyhommexx_prim_step_ckpt";

template<typename T>
T& get_from_any_map (const char* key, const char* what)
{
  auto& c = Context::singleton();
  EKAT_REQUIRE_MSG (c.any_map().find(key)!=c.any_map().end(),
      std::string("[pyhommexx] Error! ") + what + " not available.\n"
      " - call enable_adjoint() first (it requires qsize>0 and transport_alg=0 to set up the tracers adjoint).\n");
  return std::any_cast<T&>(c.any_map().at(key));
}
#endif

StateSnapshot& get_adj_state ()
{
  auto& c = Context::singleton();
  EKAT_REQUIRE_MSG (c.any_map().find(adj_state_key)!=c.any_map().end(),
      "[pyhommexx] Error! Adjoint not enabled. Call enable_adjoint() first.\n");
  return std::any_cast<StateSnapshot&>(c.any_map().at(adj_state_key));
}

template<bool ToPy>
void copy_adj_state_var (nb::ndarray<double>& arr, const nb::str& name)
{
  auto& s = get_adj_state();
  const int nelem = s.num_elems;

  std::vector<int> vector3dm_shape = {nelem,2,NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3dm_shape = {nelem,  NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3di_shape = {nelem,  NP,NP,NUM_INTERFACE_LEV};

  const std::string n (name.c_str());
  const bool is_vec = n=="uv";
  const bool is_int = n=="w" or n=="phi";
  if (n=="u" or n=="v" or n=="dp" or n=="vtheta_dp") {
    check_shape(arr,scalar3dm_shape);
  } else if (is_vec) {
    check_shape(arr,vector3dm_shape);
  } else if (is_int) {
    check_shape(arr,scalar3di_shape);
  } else {
    EKAT_ERROR_MSG("[pyhommexx] Unrecognized/unsupported adjoint state var '" + n + "'.\n"
                   " - valid names: u, v, uv, vtheta_dp, dp, w, phi\n");
  }
  EKAT_REQUIRE_MSG ((int)arr.dtype().bits==64,
      "[pyhommexx] Error! The input array must have dtype float64.\n");

  auto v   = ekat::scalarize(s.v);
  auto vth = ekat::scalarize(s.vtheta_dp);
  auto dp  = ekat::scalarize(s.dp3d);
  auto w   = ekat::scalarize(s.w_i);
  auto phi = ekat::scalarize(s.phinh_i);

  ExecViewUnmanaged<double*****> vec_v (vp2dp(arr.data()),nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  scl_m (vp2dp(arr.data()),nelem,  NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  scl_i (vp2dp(arr.data()),nelem,  NP,NP,NUM_INTERFACE_LEV);

  enum { U, V, UV, VTH, DP, W, PHI };
  const int which = n=="u" ? U : n=="v" ? V : n=="uv" ? UV : n=="vtheta_dp" ? VTH :
                    n=="dp" ? DP : n=="w" ? W : PHI;

  const int nlev = is_int ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  policy_t p ({0,0,0,0},{nelem,NP,NP,nlev});

  if constexpr (ToPy) {
    Kokkos::parallel_for(p,KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
      switch (which) {
        case U:   scl_m(ie,ip,jp,k)   = v(ie,0,ip,jp,k); break;
        case V:   scl_m(ie,ip,jp,k)   = v(ie,1,ip,jp,k); break;
        case UV:  vec_v(ie,0,ip,jp,k) = v(ie,0,ip,jp,k);
                  vec_v(ie,1,ip,jp,k) = v(ie,1,ip,jp,k); break;
        case VTH: scl_m(ie,ip,jp,k)   = vth(ie,ip,jp,k); break;
        case DP:  scl_m(ie,ip,jp,k)   = dp (ie,ip,jp,k); break;
        case W:   scl_i(ie,ip,jp,k)   = w  (ie,ip,jp,k); break;
        default:  scl_i(ie,ip,jp,k)   = phi(ie,ip,jp,k); break;
      }
    });
  } else {
    Kokkos::parallel_for(p,KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
      switch (which) {
        case U:   v(ie,0,ip,jp,k)   = scl_m(ie,ip,jp,k); break;
        case V:   v(ie,1,ip,jp,k)   = scl_m(ie,ip,jp,k); break;
        case UV:  v(ie,0,ip,jp,k)   = vec_v(ie,0,ip,jp,k);
                  v(ie,1,ip,jp,k)   = vec_v(ie,1,ip,jp,k); break;
        case VTH: vth(ie,ip,jp,k)   = scl_m(ie,ip,jp,k); break;
        case DP:  dp (ie,ip,jp,k)   = scl_m(ie,ip,jp,k); break;
        case W:   w  (ie,ip,jp,k)   = scl_i(ie,ip,jp,k); break;
        default:  phi(ie,ip,jp,k)   = scl_i(ie,ip,jp,k); break;
      }
    });
  }
  Kokkos::fence();
}

} // anonymous namespace

void update_dynamics_levels ()
{
  Context::singleton().get<TimeLevel>().update_dynamics_levels(UpdateType::LEAPFROG);
}

void enable_adjoint ()
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  auto& c = Context::singleton();
  auto& params = c.get<SimulationParams>();

  // Only supported configuration (for now)
  EKAT_REQUIRE_MSG (params.time_step_type==TimeStepType::ttype10_imex,
      "[pyhommexx::enable_adjoint] Error! Only time_step_type=10 (ttype10_imex) has an adjoint.\n");
  EKAT_REQUIRE_MSG (not params.prescribed_wind,
      "[pyhommexx::enable_adjoint] Error! 'prescribed_wind' is not supported.\n");

  const int nelem = c.get<ElementsState>().num_elems();

  params.store_fwd_state = true;
  ensure_imex_tape(nelem);
  c.any_map().try_emplace(adj_state_key,std::in_place_type<StateSnapshot>,nelem);

  // The tracers adjoint (needed by prim_step_adj) only exists for Eulerian transport.
  // The qdp adjoint state, and the checkpoint prim_step_adj replays the fwd run from.
  if (params.qsize>0 and params.transport_alg==0) {
    c.any_map().try_emplace(adj_qdp_key,std::in_place_type<PrimStepQdpView>,"pyhommexx adj qdp",nelem);
    c.any_map().try_emplace(prim_step_ckpt_key,std::in_place_type<PrimStepCheckpoint>,nelem,params.qsize);
  }
#else
  EKAT_ERROR_MSG("[pyhommexx::enable_adjoint] Error! Adjoint requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

void prim_advance_exp (const double dt, const bool compute_diagnostics, const nb::str& dtype)
{
  auto& tl = Context::singleton().get<TimeLevel>();
  dispatch_dtype(dtype,"prim_advance_exp",[&](auto tag) {
    using ST = typename decltype(tag)::type;
    Homme::prim_advance_exp<ST>(tl,dt,compute_diagnostics);
  });
}

void prim_advance_adj (const double dt)
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  Homme::prim_advance_adj(dt,get_adj_state());
#else
  (void)dt;
  EKAT_ERROR_MSG("[pyhommexx::prim_advance_adj] Error! Adjoint requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

void zero_adj_state ()
{
  get_adj_state().zero();
}

void prim_step_checkpoint_take ()
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  get_from_any_map<PrimStepCheckpoint>(prim_step_ckpt_key,"The prim_step checkpoint").take();
#else
  EKAT_ERROR_MSG("[pyhommexx::prim_step_checkpoint_take] Error! prim_step_adj requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

void prim_step_adj (const double dt)
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  auto& ckpt = get_from_any_map<PrimStepCheckpoint>(prim_step_ckpt_key,"The prim_step checkpoint");
  auto& qdp  = get_from_any_map<PrimStepQdpView>(adj_qdp_key,"The qdp adjoint state");
  Homme::prim_step_adj(dt,ckpt,get_adj_state(),qdp);
#else
  (void)dt;
  EKAT_ERROR_MSG("[pyhommexx::prim_step_adj] Error! prim_step_adj requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

#ifdef HOMMEXX_ENABLE_FAD_TYPES
namespace {
template<bool ToPy>
void copy_adj_qdp (const nb::ndarray<double>& arr)
{
  auto& adj_qdp = get_from_any_map<PrimStepQdpView>(adj_qdp_key,"The qdp adjoint state");
  const int nelem = adj_qdp.extent_int(0);
  const int qsize = Context::singleton().get<SimulationParams>().qsize;

  const std::vector<int> shape = {nelem,qsize,NP,NP,NUM_PHYSICAL_LEV};
  EKAT_REQUIRE_MSG (arr.ndim()==shape.size(),
      "[pyhommexx] Error! Wrong rank for the adjoint qdp array (expected (nelem,qsize,np,np,nlev)).\n");
  for (size_t i=0; i<shape.size(); ++i) {
    EKAT_REQUIRE_MSG (static_cast<int>(arr.shape(i))==shape[i],
        "[pyhommexx] Error! Wrong shape for the adjoint qdp array (expected (nelem,qsize,np,np,nlev)).\n");
  }
  EKAT_REQUIRE_MSG ((int)arr.dtype().bits==64,
      "[pyhommexx] Error! The input array must have dtype float64.\n");

  auto q = ekat::scalarize(adj_qdp);
  ExecViewUnmanaged<double*****> v (vp2dp(arr.data()),nelem,qsize,NP,NP,NUM_PHYSICAL_LEV);
  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>>;
  policy_t p ({0,0,0,0,0},{nelem,qsize,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(p,KOKKOS_LAMBDA (int ie, int iq, int ip, int jp, int k) {
    if constexpr (ToPy) {
      v(ie,iq,ip,jp,k) = q(ie,iq,ip,jp,k);
    } else {
      q(ie,iq,ip,jp,k) = v(ie,iq,ip,jp,k);
    }
  });
  Kokkos::fence();
}
} // anonymous namespace
#endif

void zero_adj_qdp ()
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  Kokkos::deep_copy(get_from_any_map<PrimStepQdpView>(adj_qdp_key,"The qdp adjoint state"),Real(0));
#else
  EKAT_ERROR_MSG("[pyhommexx::zero_adj_qdp] Error! Requires HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

void get_adj_qdp (nb::ndarray<double>& arr)
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  copy_adj_qdp<true>(arr);
#else
  (void)arr;
  EKAT_ERROR_MSG("[pyhommexx::get_adj_qdp] Error! Requires HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

void set_adj_qdp (const nb::ndarray<double>& arr)
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  copy_adj_qdp<false>(arr);
#else
  (void)arr;
  EKAT_ERROR_MSG("[pyhommexx::set_adj_qdp] Error! Requires HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

void get_adj_state_var (nb::ndarray<double>& arr, const nb::str& name)
{
  copy_adj_state_var<true>(arr,name);
}

void set_adj_state_var (nb::ndarray<double>& arr, const nb::str& name)
{
  copy_adj_state_var<false>(arr,name);
}

} // namespace pyhommexx
