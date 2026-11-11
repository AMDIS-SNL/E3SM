#include "pyhommexx.hpp"
#include "pyhommexx_utils.hpp"

#include "Context.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "Tracers.hpp"
#include "Types.hpp"

#include <ekat_assert.hpp>
#include <ekat_pack_kokkos.hpp>

#include <nanobind/ndarray.h>

namespace pyhommexx {

using namespace Homme;

namespace {

// What we do with the qdp entries
enum class QdpOp {
  GetVal,   // arr = qdp
  SetVal,   // qdp = arr (derivatives are NOT touched)
  GetSens,  // arr(...,ider) = d(qdp)/dp_ider
  SetSens   // d(qdp)/dp_ider = arr(...,ider) (values are NOT touched)
};

constexpr bool op_is_sens (const QdpOp op) {
  return op==QdpOp::GetSens or op==QdpOp::SetSens;
}

// Number of deriv components of a scalar type (1 for non-Fad types, which have none,
// but whose sens views we still need to be able to build, since the access kernel
// is generic). Plus access to values/derivs, which, for non-Fad types, are no-ops.
template<typename ST>
struct DerivSzOrOne { static constexpr int value = 1; };

KOKKOS_INLINE_FUNCTION void setval  (double& x, const double v) { x = v; }
KOKKOS_INLINE_FUNCTION double getsens (const double&, const int) { return 0; }
KOKKOS_INLINE_FUNCTION void setsens (double&, const int, const double) {}

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template<typename T, int N>
struct DerivSzOrOne<SFadN<T,N>> { static constexpr int value = N; };

// Set only the value: plain assignment would reset the derivs to 0
template<typename T, int N>
KOKKOS_INLINE_FUNCTION void setval (SFadN<T,N>& x, const double v) { x.val() = v; }
template<typename T, int N>
KOKKOS_INLINE_FUNCTION double getsens (const SFadN<T,N>& x, const int ider) { return x.fastAccessDx(ider); }
template<typename T, int N>
KOKKOS_INLINE_FUNCTION void setsens (SFadN<T,N>& x, const int ider, const double v) { x.fastAccessDx(ider) = v; }
#endif

// Note: tracers have no 'current' time level, only a pair of slots, which flip every
// tracers step. So it's up to the caller to say which slot he wants (see get_prim_step_qdp_slots)
template<typename ST>
void qdp_access (const nb::ndarray<double>& arr, const int slot, const QdpOp op)
{
  const auto& c = Context::singleton();
  const auto& tracers = c.get<TracersST<ST>>();
  const int nelem = tracers.qdp.extent_int(0);
  const int qsize = tracers.num_tracers();

  EKAT_REQUIRE_MSG (slot>=0 and slot<Q_NUM_TIME_LEVELS,
      "[pyhommexx::qdp_access] Error! Invalid qdp slot.\n"
      " - slot: " + std::to_string(slot) + "\n"
      " - valid values: 0.." + std::to_string(Q_NUM_TIME_LEVELS-1) + "\n");
  EKAT_REQUIRE_MSG ((int)arr.dtype().bits==64,
      "[pyhommexx::qdp_access] Error! The input array must have dtype float64.\n");

  constexpr int nder = DerivSzOrOne<ST>::value;
  std::vector<int> shape = {nelem,qsize,NP,NP,NUM_PHYSICAL_LEV};
  if (op_is_sens(op)) {
    shape.push_back(nder);
  }
  EKAT_REQUIRE_MSG (arr.ndim()==shape.size(),
      "[pyhommexx::qdp_access] Error! Wrong rank for the input array.\n");
  for (size_t i=0; i<shape.size(); ++i) {
    EKAT_REQUIRE_MSG (static_cast<int>(arr.shape(i))==shape[i],
        "[pyhommexx::qdp_access] Error! Wrong shape for the input array.\n"
        " - expected shape: (nelem,qsize,np,np,nlev[,nder]) = (" + std::to_string(nelem) + "," +
                           std::to_string(qsize) + "," + std::to_string(NP) + "," +
                           std::to_string(NP) + "," + std::to_string(NUM_PHYSICAL_LEV) +
                           (op_is_sens(op) ? "," + std::to_string(nder) : std::string()) + ")\n");
  }

  auto qdp = ekat::scalarize(tracers.qdp);
  ExecViewUnmanaged<double*****> val_v  (vp2dp(arr.data()),nelem,qsize,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double******> sens_v (vp2dp(arr.data()),nelem,qsize,NP,NP,NUM_PHYSICAL_LEV,nder);

  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>>;
  policy_t p ({0,0,0,0,0},{nelem,qsize,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::parallel_for(p,KOKKOS_LAMBDA (int ie, int iq, int ip, int jp, int k) {
    switch (op) {
      case QdpOp::GetVal:
        val_v(ie,iq,ip,jp,k) = ADValue(qdp(ie,slot,iq,ip,jp,k));
        break;
      case QdpOp::SetVal:
        // Assigning a Real resets the derivs to 0 for Fad types: we want to preserve them
        setval(qdp(ie,slot,iq,ip,jp,k),val_v(ie,iq,ip,jp,k));
        break;
      case QdpOp::GetSens:
        for (int ider=0; ider<nder; ++ider) {
          sens_v(ie,iq,ip,jp,k,ider) = getsens(qdp(ie,slot,iq,ip,jp,k),ider);
        }
        break;
      case QdpOp::SetSens:
        for (int ider=0; ider<nder; ++ider) {
          setsens(qdp(ie,slot,iq,ip,jp,k),ider,sens_v(ie,iq,ip,jp,k,ider));
        }
        break;
    }
  });
  Kokkos::fence();
}

} // anonymous namespace

nb::tuple get_prim_step_qdp_slots ()
{
  const auto& c = Context::singleton();
  const auto& params = c.get<SimulationParams>();

  // Replicate the sequence of level updates of prim_step (see also
  // prim_step_final_time_level in prim_step_adj.hpp): K-1 dynamics levels
  // updates, and then the tracers levels are updated based on the nstep we got to.
  const int K = params.dt_tracer_factor;
  TimeLevel tl = c.get<TimeLevel>();
  for (int n=1; n<K; ++n) {
    tl.update_dynamics_levels(UpdateType::LEAPFROG);
  }
  tl.update_tracers_levels(K);
  return nb::make_tuple(tl.n0_qdp,tl.np1_qdp);
}

void get_qdp (nb::ndarray<double>& arr, const nb::str& dtype, const int slot)
{
  dispatch_dtype(dtype,"get_qdp",[&](auto tag) {
    using ST = typename decltype(tag)::type;
    qdp_access<ST>(arr,slot,QdpOp::GetVal);
  });
}

void set_qdp (const nb::ndarray<double>& arr, const nb::str& dtype, const int slot)
{
  dispatch_dtype(dtype,"set_qdp",[&](auto tag) {
    using ST = typename decltype(tag)::type;
    qdp_access<ST>(arr,slot,QdpOp::SetVal);
  });
}

void get_qdp_dp_sens (nb::ndarray<double>& arr, const int slot)
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  qdp_access<DpFadType>(arr,slot,QdpOp::GetSens);
#else
  (void)arr; (void)slot;
  EKAT_ERROR_MSG("[pyhommexx::get_qdp_dp_sens] Error! dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

void set_qdp_dp_sens (const nb::ndarray<double>& arr, const int slot)
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  qdp_access<DpFadType>(arr,slot,QdpOp::SetSens);
#else
  (void)arr; (void)slot;
  EKAT_ERROR_MSG("[pyhommexx::set_qdp_dp_sens] Error! dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
}

} // namespace pyhommexx
