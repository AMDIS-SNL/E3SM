#include "pyhommexx.hpp"
#include "pyhommexx_utils.hpp"

#include "Context.hpp"
#include "ElementsState.hpp"
#include "SimulationParams.hpp"
#include "StateSnapshot.hpp"
#include "Tape.hpp"
#include "TimeLevel.hpp"
#include "Types.hpp"

#ifdef HOMMEXX_ENABLE_FAD_TYPES
#include "prim_advance_adj.hpp"
#endif

#include <ekat_assert.hpp>

#include <nanobind/ndarray.h>

namespace Homme {
void prim_advance_exp (TimeLevel& tl, const Real dt, const bool compute_diagnostics);
} // namespace Homme

namespace pyhommexx {

using namespace Homme;

namespace {

constexpr const char* adj_state_key = "pyhommexx_adj_state";

// Checkpoints stored by one prim_advance_exp call (see ttype10_imex_timestep
// and the store_fwd_state block of prim_advance_exp.cpp). prim_advance_adj
// reads up to index 11 (post-HV state)
constexpr int imex_tape_size = 12;

StateSnapshot& get_adj_state ()
{
  auto& c = Context::singleton();
  EKAT_REQUIRE_MSG (c.any_map().find(adj_state_key)!=c.any_map().end(),
      "[pyhommexx] Error! Adjoint not enabled. Call enable_adjoint() first.\n");
  return std::any_cast<StateSnapshot&>(c.any_map().at(adj_state_key));
}

// Copy between a (scalarized) snapshot entry and a py array entry
template<bool ToPy>
KOKKOS_INLINE_FUNCTION void xfer (Real& snap, double& py)
{
  if constexpr (ToPy) py = snap; else snap = py;
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
  assert ((int)arr.dtype().bits==64);

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

  Kokkos::parallel_for(p,KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    switch (which) {
      case U:   xfer<ToPy>(v(ie,0,ip,jp,k),scl_m(ie,ip,jp,k)); break;
      case V:   xfer<ToPy>(v(ie,1,ip,jp,k),scl_m(ie,ip,jp,k)); break;
      case UV:  xfer<ToPy>(v(ie,0,ip,jp,k),vec_v(ie,0,ip,jp,k));
                xfer<ToPy>(v(ie,1,ip,jp,k),vec_v(ie,1,ip,jp,k)); break;
      case VTH: xfer<ToPy>(vth(ie,ip,jp,k),scl_m(ie,ip,jp,k)); break;
      case DP:  xfer<ToPy>(dp (ie,ip,jp,k),scl_m(ie,ip,jp,k)); break;
      case W:   xfer<ToPy>(w  (ie,ip,jp,k),scl_i(ie,ip,jp,k)); break;
      default:  xfer<ToPy>(phi(ie,ip,jp,k),scl_i(ie,ip,jp,k)); break;
    }
  });
  Kokkos::fence();
}

} // anonymous namespace

nb::dict get_time_levels ()
{
  const auto& tl = Context::singleton().get<TimeLevel>();
  nb::dict d;
  d["nm1"]   = tl.nm1;
  d["n0"]    = tl.n0;
  d["np1"]   = tl.np1;
  d["nstep"] = tl.nstep;
  return d;
}

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
  c.any_map().try_emplace("imex_tape",std::in_place_type<Tape<StateSnapshot>>,imex_tape_size,nelem);
  c.any_map().try_emplace(adj_state_key,std::in_place_type<StateSnapshot>,nelem);
#else
  EKAT_ERROR_MSG("[pyhommexx::enable_adjoint] Error! Adjoint requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

void prim_advance_exp (const double dt, const bool compute_diagnostics)
{
  auto& tl = Context::singleton().get<TimeLevel>();
  Homme::prim_advance_exp(tl,dt,compute_diagnostics);
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

void get_adj_state_var (nb::ndarray<double>& arr, const nb::str& name)
{
  copy_adj_state_var<true>(arr,name);
}

void set_adj_state_var (nb::ndarray<double>& arr, const nb::str& name)
{
  copy_adj_state_var<false>(arr,name);
}

} // namespace pyhommexx
