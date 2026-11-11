#ifndef PYHOMMEXX_UTILS_HPP
#define PYHOMMEXX_UTILS_HPP

#include "pyhommexx.hpp"

#include "Hommexx_Session.hpp"
#include "Types.hpp"

#include <ekat_assert.hpp>

#include <cassert>
#include <string>
#include <vector>

namespace pyhommexx {

inline double* vp2dp (void* p)
{
  return reinterpret_cast<double*>(p);
}
inline const double* vp2cdp (void* p)
{
  return reinterpret_cast<const double*>(p);
}

template<typename T>
void check_shape(const nb::ndarray<T>& arr, const std::vector<int>& shape)
{
  assert (arr.ndim()==shape.size());
  for (size_t i=0; i<shape.size(); ++i) {
    assert (static_cast<int>(arr.shape(i))==shape[i]);
  }
}

// Python has no templates, so each binding that must work on different scalar types takes
// a runtime 'dtype' string, and dispatches to the right instantiation of a templated impl.
// This helper does the string -> type translation once, so that bindings can do
//
//   dispatch_dtype(dtype,"my_fcn",[&](auto tag) {
//     using ST = typename decltype(tag)::type;
//     my_fcn_impl<ST>(...);
//   });
//
// It errors out if the dtype is unknown, if it's not available in this build
// (dpfad needs HOMMEXX_ENABLE_FAD_TYPES), or if it was not enabled in the session
// via enable_scalar_type() (since the data structures for that type would not exist).
template<typename ST>
struct StTag { using type = ST; };

template<typename F>
void dispatch_dtype (const nb::str& dtype, const std::string& caller, F&& f)
{
  const std::string d (dtype.c_str());

  if (d=="real") {
    EKAT_REQUIRE_MSG (Homme::Session::is_st_enabled<Homme::Real>(),
        "[pyhommexx::" + caller + "] Error! Scalar type 'real' is not enabled.\n"
        " - call enable_scalar_type('real') before model_init().\n");
    f(StTag<Homme::Real>{});
  } else if (d=="dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    EKAT_REQUIRE_MSG (Homme::Session::is_st_enabled<Homme::DpFadType>(),
        "[pyhommexx::" + caller + "] Error! Scalar type 'dpfad' is not enabled.\n"
        " - call enable_scalar_type('dpfad') before model_init().\n");
    f(StTag<Homme::DpFadType>{});
#else
    EKAT_ERROR_MSG ("[pyhommexx::" + caller + "] Error! dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
    EKAT_ERROR_MSG ("[pyhommexx::" + caller + "] Error! Unrecognized/unsupported dtype name.\n"
                    " - input dtype: " + d + "\n"
                    " - valid dtype(s): real, dpfad\n");
  }
}

} // namespace pyhommexx

#endif // PYHOMMEXX_UTILS_HPP
