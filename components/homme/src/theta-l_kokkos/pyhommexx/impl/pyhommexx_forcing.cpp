#include "pyhommexx.hpp"
#include "pyhommexx_utils.hpp"

#include "Context.hpp"
#include "ElementsForcing.hpp"
#include "ForcingFunctor.hpp"
#include "Hommexx_Session.hpp"
#include "TimeLevel.hpp"
#include "Types.hpp"

#include <ekat_assert.hpp>

#include <nanobind/ndarray.h>

#include <map>
#include <string>
#include <vector>

namespace pyhommexx {

using namespace Homme;

namespace {

constexpr int flag_fm     = 0;   // full [3] vector momentum forcing
constexpr int flag_fm_x   = 1;
constexpr int flag_fm_y   = 2;
constexpr int flag_fm_z   = 3;
constexpr int flag_fvtheta = 4;
constexpr int flag_fphi    = 5;

int name_to_flag (const std::string& n) {
  static const std::map<std::string,int> m = {
    {"fm",      flag_fm},
    {"fm_x",    flag_fm_x},
    {"fm_y",    flag_fm_y},
    {"fm_z",    flag_fm_z},
    {"fvtheta", flag_fvtheta},
    {"fphi",    flag_fphi},
  };
  auto it = m.find(n);
  EKAT_REQUIRE_MSG(it != m.end(),
      "[pyhommexx] Unrecognized/unsupported forcing var '" + n + "'.\n"
      " - valid names: fm, fm_x, fm_y, fm_z, fvtheta, fphi\n");
  return it->second;
}

template<typename T>
void check_forcing_shape (const nb::ndarray<T>& arr,
                          const int nelem, const int which)
{
  std::vector<int> vec_mid_shape = {nelem, 3, NP, NP, NUM_PHYSICAL_LEV};
  std::vector<int> scl_mid_shape = {nelem,    NP, NP, NUM_PHYSICAL_LEV};
  std::vector<int> scl_int_shape = {nelem,    NP, NP, NUM_INTERFACE_LEV};

  switch (which) {
    case flag_fm:
      check_shape(arr, vec_mid_shape); break;
    case flag_fm_x:
    case flag_fm_y:
    case flag_fm_z:
    case flag_fvtheta:
      check_shape(arr, scl_mid_shape); break;
    case flag_fphi:
      check_shape(arr, scl_int_shape); break;
    default:
      EKAT_ERROR_MSG("[pyhommexx] check_forcing_shape: bad which.\n");
  }
}

} // anonymous namespace

template<typename ST>
void get_forcing_impl (nb::ndarray<double>& arr, const nb::str& name)
{
  const auto& c = Context::singleton();
  const auto& forcing = c.get<ElementsForcingST<ST>>();
  const int nelem = forcing.num_elems();

  const std::string n(name.c_str());
  const int which = name_to_flag(n);
  check_forcing_shape(arr, nelem, which);
  assert ((int)arr.dtype().bits == 64);

  ExecViewUnmanaged<double*****> vec_mid (vp2dp(arr.data()), nelem, 3, NP, NP, NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  scl_mid (vp2dp(arr.data()), nelem,    NP, NP, NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  scl_int (vp2dp(arr.data()), nelem,    NP, NP, NUM_INTERFACE_LEV);

  auto fm      = forcing.m_fm;
  auto fvtheta = forcing.m_fvtheta;
  auto fphi    = forcing.m_fphi;

  const int nlev = (which == flag_fphi) ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
  policy_t p ({0,0,0,0}, {nelem, NP, NP, nlev});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    const int lev = k / VECTOR_SIZE;
    const int vec = k % VECTOR_SIZE;
    switch (which) {
      case flag_fm:
        vec_mid(ie,0,ip,jp,k) = ADValue(fm(ie,0,ip,jp,lev)[vec]);
        vec_mid(ie,1,ip,jp,k) = ADValue(fm(ie,1,ip,jp,lev)[vec]);
        vec_mid(ie,2,ip,jp,k) = ADValue(fm(ie,2,ip,jp,lev)[vec]); break;
      case flag_fm_x:
        scl_mid(ie,ip,jp,k) = ADValue(fm(ie,0,ip,jp,lev)[vec]); break;
      case flag_fm_y:
        scl_mid(ie,ip,jp,k) = ADValue(fm(ie,1,ip,jp,lev)[vec]); break;
      case flag_fm_z:
        scl_mid(ie,ip,jp,k) = ADValue(fm(ie,2,ip,jp,lev)[vec]); break;
      case flag_fvtheta:
        scl_mid(ie,ip,jp,k) = ADValue(fvtheta(ie,ip,jp,lev)[vec]); break;
      case flag_fphi:
        scl_int(ie,ip,jp,k) = ADValue(fphi(ie,ip,jp,lev)[vec]); break;
      default:
        Kokkos::abort("Unsupported value for 'which' in get_forcing.\n");
    }
  };
  Kokkos::parallel_for(p, copy);
}

template<typename ST>
void set_forcing_impl (const nb::ndarray<double>& arr, const nb::str& name)
{
  const auto& c = Context::singleton();
  const auto& forcing = c.get<ElementsForcingST<ST>>();
  const int nelem = forcing.num_elems();

  const std::string n(name.c_str());
  const int which = name_to_flag(n);
  check_forcing_shape(arr, nelem, which);
  assert ((int)arr.dtype().bits == 64);

  ExecViewUnmanaged<const double*****> vec_mid (vp2cdp(arr.data()), nelem, 3, NP, NP, NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  scl_mid (vp2cdp(arr.data()), nelem,    NP, NP, NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  scl_int (vp2cdp(arr.data()), nelem,    NP, NP, NUM_INTERFACE_LEV);

  auto fm      = forcing.m_fm;
  auto fvtheta = forcing.m_fvtheta;
  auto fphi    = forcing.m_fphi;

  const int nlev = (which == flag_fphi) ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
  policy_t p ({0,0,0,0}, {nelem, NP, NP, nlev});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    const int lev = k / VECTOR_SIZE;
    const int vec = k % VECTOR_SIZE;
    switch (which) {
      case flag_fm:
        fm(ie,0,ip,jp,lev)[vec] = vec_mid(ie,0,ip,jp,k);
        fm(ie,1,ip,jp,lev)[vec] = vec_mid(ie,1,ip,jp,k);
        fm(ie,2,ip,jp,lev)[vec] = vec_mid(ie,2,ip,jp,k); break;
      case flag_fm_x:
        fm(ie,0,ip,jp,lev)[vec] = scl_mid(ie,ip,jp,k); break;
      case flag_fm_y:
        fm(ie,1,ip,jp,lev)[vec] = scl_mid(ie,ip,jp,k); break;
      case flag_fm_z:
        fm(ie,2,ip,jp,lev)[vec] = scl_mid(ie,ip,jp,k); break;
      case flag_fvtheta:
        fvtheta(ie,ip,jp,lev)[vec] = scl_mid(ie,ip,jp,k); break;
      case flag_fphi:
        fphi(ie,ip,jp,lev)[vec] = scl_int(ie,ip,jp,k); break;
      default:
        Kokkos::abort("Unsupported value for 'which' in set_forcing.\n");
    }
  };
  Kokkos::parallel_for(p, copy);
}

template<typename ST>
void set_forcing_value_impl (const double value, const nb::str& name)
{
  const auto& c = Context::singleton();
  const auto& forcing = c.get<ElementsForcingST<ST>>();

  const std::string n(name.c_str());
  const int which = name_to_flag(n);

  auto fm      = forcing.m_fm;
  auto fvtheta = forcing.m_fvtheta;
  auto fphi    = forcing.m_fphi;

  const auto A = Kokkos::ALL();
  ST v (value);
  switch (which) {
    case flag_fm:
      Kokkos::deep_copy(fm, v); break;
    case flag_fm_x:
      Kokkos::deep_copy(Kokkos::subview(fm, A, 0, A, A, A), v); break;
    case flag_fm_y:
      Kokkos::deep_copy(Kokkos::subview(fm, A, 1, A, A, A), v); break;
    case flag_fm_z:
      Kokkos::deep_copy(Kokkos::subview(fm, A, 2, A, A, A), v); break;
    case flag_fvtheta:
      Kokkos::deep_copy(fvtheta, v); break;
    case flag_fphi:
      Kokkos::deep_copy(fphi, v); break;
    default:
      EKAT_ERROR_MSG("Unsupported value for 'which' in set_forcing_value.\n");
  }
}

template<typename ST>
void apply_dynamics_forcing_impl (const double dt)
{
  auto& c = Context::singleton();
  auto& ff = c.get<ForcingFunctorST<ST>>();
  const auto& tl = c.get<TimeLevel>();
  ff.states_forcing(dt, tl.n0);
}

void get_forcing (nb::ndarray<double>& arr, const nb::str& name, const nb::str& dtype)
{
  const std::string dtype_str(dtype.c_str());
  if (dtype_str == "real") {
    get_forcing_impl<Real>(arr, name);
  } else if (dtype_str == "dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    get_forcing_impl<DpFadType>(arr, name);
#else
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
    EKAT_ERROR_MSG("[get_forcing] Error! Unrecognized/unsupported dtype name.\n"
        " - input dtype: " + dtype_str + "\n"
        " - valid dtype(s): real, dpfad\n");
  }
}

void set_forcing (const nb::ndarray<double>& arr, const nb::str& name, const nb::str& dtype)
{
  const std::string dtype_str(dtype.c_str());
  if (dtype_str == "real") {
    set_forcing_impl<Real>(arr, name);
  } else if (dtype_str == "dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    set_forcing_impl<DpFadType>(arr, name);
#else
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
    EKAT_ERROR_MSG("[set_forcing] Error! Unrecognized/unsupported dtype name.\n"
        " - input dtype: " + dtype_str + "\n"
        " - valid dtype(s): real, dpfad\n");
  }
}

void set_forcing_value (const double value, const nb::str& name, const nb::str& dtype)
{
  const std::string dtype_str(dtype.c_str());
  if (dtype_str == "real") {
    set_forcing_value_impl<Real>(value, name);
  } else if (dtype_str == "dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    set_forcing_value_impl<DpFadType>(value, name);
#else
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
    EKAT_ERROR_MSG("[set_forcing_value] Error! Unrecognized/unsupported dtype name.\n"
        " - input dtype: " + dtype_str + "\n"
        " - valid dtype(s): real, dpfad\n");
  }
}

void apply_dynamics_forcing (const double dt, const nb::str& dtype)
{
  const std::string dtype_str(dtype.c_str());
  if (dtype_str == "real") {
    apply_dynamics_forcing_impl<Real>(dt);
  } else if (dtype_str == "dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    apply_dynamics_forcing_impl<DpFadType>(dt);
#else
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
    EKAT_ERROR_MSG("[apply_dynamics_forcing] Error! Unrecognized/unsupported dtype name.\n"
        " - input dtype: " + dtype_str + "\n"
        " - valid dtype(s): real, dpfad\n");
  }
}

} // namespace pyhommexx
