/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_TEST_UTILS_HPP
#define HOMMEXX_TEST_UTILS_HPP

#include "Types.hpp"
#include "ErrorDefs.hpp"
#include <functional>

namespace Homme {

template <typename rngAlg, typename PDF>
void genRandArray(int *const x, int length, rngAlg &engine, PDF &&pdf) {
  for (int i = 0; i < length; ++i) {
    x[i] = pdf(engine);
  }
}

template <typename rngAlg, typename PDF>
void genRandArray(Real *const x, int length, rngAlg &engine, PDF &&pdf) {
  for (int i = 0; i < length; ++i) {
    x[i] = pdf(engine);
  }
}

template <typename ST, typename rngAlg, typename PDF>
void genRandArray(PackType<ST> *const x, int length, rngAlg &engine, PDF &&pdf) {
  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      x[i][j] = pdf(engine);
    }
  }
}

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template <typename FadType, typename rngAlg, typename PDF>
void genRandArray(FadType *const x, int length, rngAlg &engine, PDF &&pdf) {
  for (int i = 0; i < length; ++i) {
    x[i] = pdf(engine);
    for (int k=0; k<x[i].size(); ++k)
      x[i].fastAccessDx(k) = pdf(engine);
  }
}
#endif

template <typename ViewType, typename rngAlg, typename PDF>
typename std::enable_if<Kokkos::is_view<ViewType>::value, void>::type
genRandArray(ViewType view, rngAlg &engine, PDF &&pdf,
             std::function<bool(typename ViewType::HostMirror)> constraint,
             const int max_attempts = 1000) {
  typename ViewType::HostMirror mirror = Kokkos::create_mirror_view(view);
  int iter=0;
  bool success;
  do {
    genRandArray(mirror.data(), view.size(), engine, pdf);
    ++iter;
    success = constraint(mirror);
  } while (!success && iter<max_attempts);
  if (!success) {
    EKAT_ERROR_MSG("Error! Failed to randomly initialize Kokkos view '" +
                           view.label() + "' in " + std::to_string(max_attempts) + " attempts.\n");
  }
  Kokkos::deep_copy(view, mirror);
}

template <typename ViewType, typename rngAlg, typename PDF>
typename std::enable_if<Kokkos::is_view<ViewType>::value, void>::type
genRandArray(ViewType view, rngAlg &engine, PDF &&pdf) {
  genRandArray(view, engine, pdf,
               [](typename ViewType::HostMirror) { return true; });
}

inline Real compare_answers(Real target, Real computed,
                            Real relative_coeff = 1) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target - computed) / denom;
}
#ifdef HOMMEXX_ENABLE_FWD_SENS
inline Real compare_answers(Real target, ScalarValue computed,
                Real relative_coeff = 1) {
  return compare_answers(target,ADValue(computed),relative_coeff);
}
#endif

// If last_extent=-1, views last extent must match. Otherwise we compare
// entries (along last dim) only up to the provided extent.
template<typename V1, typename V2>
std::enable_if_t<Kokkos::is_view_v<V1> and Kokkos::is_view_v<V2>, bool>
views_are_equal(const V1& v1, const V2& v2, int last_extent = 0)
{
  if constexpr (v1.rank!=v2.rank)
    return false;

  constexpr int N = v1.rank;
  for (int i=0; i<N-1; ++i)
    if (v1.extent(i)!=v2.extent(i))
      return false;

  if (last_extent>0) {
    /// If views are smaller than provided extent, something is off
    if (v1.extent(N-1)<last_extent or v2.extent(N-1)<last_extent)
      return false;
  } else {
    if (v1.extent(N-1)!=v2.extent(N-1))
    return false;
  }

  auto v1h = Kokkos::create_mirror_view(v1);
  auto v2h = Kokkos::create_mirror_view(v2);
  Kokkos::deep_copy(v1h,v1);
  Kokkos::deep_copy(v2h,v2);

  if constexpr (N==0) {
    return v1h()==v2h();
  } else if constexpr(N==1) {
    for (int i=0; i<last_extent; ++i)
      if (v1h(i)!=v2h(i))
        return false;
  } else if constexpr(N==2) {
    for (int i=0; i<v1.extent(0); ++i)
      for (int j=0; j<last_extent; ++j)
        if (v1h(i,j)!=v2h(i,j))
          return false;
  } else if constexpr(N==3) {
    for (int i=0; i<v1.extent(0); ++i)
      for (int j=0; j<v1.extent(1); ++j)
        for (int k=0; k<last_extent; ++k)
          if (v1h(i,j,k)!=v2h(i,j,k))
            return false;
  } else if constexpr(N==4) {
    for (int i=0; i<v1.extent(0); ++i)
      for (int j=0; j<v1.extent(1); ++j)
        for (int k=0; k<v1.extent(2); ++k)
          for (int l=0; l<last_extent; ++l)
            if (v1h(i,j,k,l)!=v2h(i,j,k,l))
              return false;
  } else if constexpr(N==5) {
    for (int i=0; i<v1.extent(0); ++i)
      for (int j=0; j<v1.extent(1); ++j)
        for (int k=0; k<v1.extent(2); ++k)
          for (int l=0; l<v1.extent(3); ++l)
            for (int m=0; m<last_extent; ++m)
              if (v1h(i,j,k,l,m)!=v2h(i,j,k,l,m))
                return false;
  } else if constexpr(N==6) {
    for (int i=0; i<v1.extent(0); ++i)
      for (int j=0; j<v1.extent(1); ++j)
        for (int k=0; k<v1.extent(2); ++k)
          for (int l=0; l<v1.extent(3); ++l)
            for (int m=0; m<v1.extent(4); ++m)
              for (int n=0; n<last_extent; ++n)
                if (v1h(i,j,k,l,m,n)!=v2h(i,j,k,l,m,n))
                  return false;
  } else {
    throw std::runtime_error("[views_are_equal] Unsupported rank (" + std::to_string(N) + ").\n");
  }

  return true;
}

// Dot product of two same-shaped views (rank 3, 4, or 5), via a host-mirror
// round trip. If last_dim>0, only the first last_dim entries along the last
// dimension are summed (e.g. to skip padding past the physical extent);
// otherwise the full last dimension is used.
template<typename ViewT>
double dot (ViewT v1, ViewT v2, int last_dim = -1) {
  auto v1h = Kokkos::create_mirror_view(v1);
  Kokkos::deep_copy(v1h,v1);
  auto v2h = Kokkos::create_mirror_view(v2);
  Kokkos::deep_copy(v2h,v2);

  EKAT_REQUIRE_MSG (v1.rank==v2.rank, "Error! Views have different rank.\n");
  EKAT_REQUIRE_MSG (v1.rank==3 or v1.rank==4 or v1.rank==5, "Error! Unsupported rank.\n");
  double prod = 0;
  if constexpr(v1.rank==3) {
    auto last_ext = last_dim > 0 ? last_dim : v1.extent_int(2);
    for (int i=0; i<v1.extent_int(0); ++i)
      for (int j=0; j<v1.extent_int(1); ++j)
        for (int k=0; k<last_ext; ++k)
          prod += v1h(i,j,k)*v2h(i,j,k);
  } else if constexpr(v1.rank==4) {
    auto last_ext = last_dim > 0 ? last_dim : v1.extent_int(3);
    for (int i=0; i<v1.extent_int(0); ++i)
      for (int j=0; j<v1.extent_int(1); ++j)
        for (int k=0; k<v1.extent_int(2); ++k)
          for (int l=0; l<last_ext; ++l)
            prod += v1h(i,j,k,l)*v2h(i,j,k,l);
  } else if constexpr(v1.rank==5) {
    auto last_ext = last_dim > 0 ? last_dim : v1.extent_int(4);
    for (int i=0; i<v1.extent_int(0); ++i)
      for (int j=0; j<v1.extent_int(1); ++j)
        for (int k=0; k<v1.extent_int(2); ++k)
          for (int l=0; l<v1.extent_int(3); ++l)
            for (int m=0; m<last_ext; ++m)
              prod += v1h(i,j,k,l,m)*v2h(i,j,k,l,m);
  }

  return prod;
}

} // namespace Homme

#endif // HOMMEXX_TEST_UTILS_HPP
