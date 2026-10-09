#ifndef HOMMEXX_THETAL_UT_UTILS_HPP
#define HOMMEXX_THETAL_UT_UTILS_HPP

// Small utilities shared by the theta-l unit tests (random numbers, view copies,
// comparison of numbers, ...), so that each test doesn't have to redefine them.
//
// Everything lives in the Homme::ut namespace, NOT in Homme. The tests do
// 'using namespace Homme', so adding these names directly to Homme could make them
// ambiguous with test-specific helpers (e.g., a test needing a different default
// tolerance in 'equal'). Tests can do 'using namespace Homme::ut', or just pull in
// the names they need.

#include "Types.hpp"

#include <catch2/catch.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <type_traits>

namespace Homme {
namespace ut {

// Host views with right-most index fastest (same layout as f90), of rank 1-5
using CA1d = Kokkos::View<Real*,     Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA2d = Kokkos::View<Real**,    Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA3d = Kokkos::View<Real***,   Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA4d = Kokkos::View<Real****,  Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA5d = Kokkos::View<Real*****, Kokkos::LayoutRight, Kokkos::HostSpace>;

// Create a host mirror of v, and deep copy v into it
template <typename V>
decltype(Kokkos::create_mirror_view(V())) cmvdc (const V& v) {
  const auto h = Kokkos::create_mirror_view(v);
  deep_copy(h, v);
  return h;
}

// Random number generator, seeded by the seed Catch2 was given,
// or by a random seed (if Catch2's seed is 0)
class Random {
  using rngalg = std::mt19937_64;
  using rpdf = std::uniform_real_distribution<Real>;
  using ipdf = std::uniform_int_distribution<int>;
  std::random_device rd;
  unsigned int seed;
  rngalg engine;
public:
  Random (unsigned int seed_ = Catch::rngSeed()) : seed(seed_ == 0 ? rd() : seed_), engine(seed) {}
  unsigned int gen_seed () { return seed; }
  Real urrng (const Real lo = 0, const Real hi = 1) { return rpdf(lo, hi)(engine); }
  int  uirng (const int lo, const int hi) { return ipdf(lo, hi)(engine); }
};

// Fill a (pack) view with random numbers in [-scale,scale]
template <typename V>
void fill (Random& r, const V& a, const Real scale = 1,
           typename std::enable_if<V::rank == 3>::type* = 0) {
  const auto am = cmvdc(a);
  for (int i = 0; i < a.extent_int(0); ++i)
    for (int j = 0; j < a.extent_int(1); ++j)
      for (int k = 0; k < a.extent_int(2); ++k)
        for (int s = 0; s < VECTOR_SIZE; ++s)
          am(i,j,k)[s] = scale*r.urrng(-1,1);
  deep_copy(a, am);
}

// Fill a (pack) view with random numbers in [-1,1]
template <typename V>
void fill (Random& r, const V& a,
           typename std::enable_if<V::rank == 4>::type* = 0) {
  const auto am = cmvdc(a);
  for (int i = 0; i < a.extent_int(0); ++i)
    for (int j = 0; j < a.extent_int(1); ++j)
      for (int k = 0; k < a.extent_int(2); ++k)
        for (int l = 0; l < a.extent_int(3); ++l)
          for (int s = 0; s < VECTOR_SIZE; ++s)
            am(i,j,k,l)[s] = r.urrng(-1,1);
  deep_copy(a, am);
}

// Check that a and b are equal, up to the given relative tolerance
inline bool almost_equal (const Real& a, const Real& b,
                          const Real tol = 0) {
  const auto re = std::abs(a-b)/(1 + std::abs(a));
  const bool good = re <= tol;
  if ( ! good)
    printf("equal: a,b = %23.16e %23.16e re = %23.16e tol %9.2e\n",
           a, b, re, tol);
  return good;
}

// Check that a and b are equal: exactly, if BFB testing is on,
// up to the given relative tolerance otherwise.
inline bool equal (const Real& a, const Real& b,
                   // Used only if not defined HOMMEXX_BFB_TESTING.
                   const Real tol = 0) {
#ifdef HOMMEXX_BFB_TESTING
  if (a != b)
    printf("equal: a,b = %23.16e %23.16e re = %23.16e\n",
           a, b, std::abs((a-b)/a));
  return a == b;
#else
  return almost_equal(a, b, tol);
#endif
}

#ifdef HOMMEXX_ENABLE_FWD_SENS
inline bool almost_equal (const ScalarValue& a, const ScalarValue& b,
                          // Used only if not defined HOMMEXX_BFB_TESTING.
                          const Real tol = 0) {
  return almost_equal(ADValue(a),ADValue(b),tol);
}

inline bool equal (const ScalarValue& a, const ScalarValue& b,
                   // Used only if not defined HOMMEXX_BFB_TESTING.
                   const Real tol = 0) {
  return equal(ADValue(a),ADValue(b),tol);
}
#endif

} // namespace ut
} // namespace Homme

#endif // HOMMEXX_THETAL_UT_UTILS_HPP
