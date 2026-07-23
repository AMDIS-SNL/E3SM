/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_HASH_HPP
#define HOMMEXX_HASH_HPP

#include <cstdint>

#include "Types.hpp"

/* Utilities to calculate a hash for a given model state. Hash values can be
   compared between runs to find instances of non-BFBness.
 */

namespace Homme {

typedef std::uint64_t HashType;

// Each hash function accumulates v into accum using a hash of its bits.

KOKKOS_INLINE_FUNCTION void hash (const HashType v, HashType& accum) {
  constexpr auto first_bit = 1ULL << 63;
  accum += ~first_bit & v; // no overflow
  accum ^=  first_bit & v; // handle most significant bit  
}

// For Kokkos::parallel_reduce.
template <typename ExecSpace = Kokkos::HostSpace>
struct HashReducer {
  typedef HashReducer reducer;
  typedef HashType value_type;
  typedef Kokkos::View<value_type*, ExecSpace, Kokkos::MemoryUnmanaged> result_view_type;

  KOKKOS_INLINE_FUNCTION HashReducer (value_type& value_) : value(value_) {}
  KOKKOS_INLINE_FUNCTION void join (value_type& dest, const value_type& src) const { hash(src, dest); }
  KOKKOS_INLINE_FUNCTION void init (value_type& val) const { val = 0; }
  KOKKOS_INLINE_FUNCTION value_type& reference () const { return value; }
  KOKKOS_INLINE_FUNCTION bool references_scalar () const { return true; }
  KOKKOS_INLINE_FUNCTION result_view_type view () const { return result_view_type(&value, 1); }

private:
  value_type& value;
};

KOKKOS_INLINE_FUNCTION void hash (const double v_, HashType& accum) {
  HashType v;
  std::memcpy(&v, &v_, sizeof(HashType));
  hash(v, accum);
}

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template<typename T, int N>
KOKKOS_INLINE_FUNCTION
void hash (const SFadN<T,N> v_, HashType& accum) {
  hash(v_.val(), accum);
}
#endif

// Overloads for views: dispatch || reduce and hash all entries.
// The last integer specifies where to stop along the last dimension
template<typename DT,typename... Props>
std::enable_if_t<Kokkos::View<DT,Props...>::rank==6>
hash (int itl, const Kokkos::View<DT,Props...>& v, int n5, HashType& accum_out)
{
  using value_t = decltype(*v.data());
  using scalar_t = typename ekat::ScalarTraits<value_t>::scalar_type;

  HashType accum;
  int beg[5] = {0, 0, 0, 0, 0};
  int end[5] = {v.extent_int(0), v.extent_int(2), v.extent_int(3), v.extent_int(4), n5};
  MDRangePolicy<ExecSpace, 5> policy(beg,end);
  auto lambda = KOKKOS_LAMBDA(int i0, int i2, int i3, int i4, int i5, HashType& accum) {
    const auto* vcol = reinterpret_cast<const scalar_t*>(&v(i0,itl,i2,i3,i4,0));
    Homme::hash(vcol[i5], accum);
  };
  Kokkos::parallel_reduce(policy,lambda,HashReducer<>(accum));
  hash(accum, accum_out);
}

template<typename DT,typename... Props>
std::enable_if_t<Kokkos::View<DT,Props...>::rank==5>
hash (const Kokkos::View<DT,Props...>& v, int n4, HashType& accum_out)
{
  using value_t = decltype(*v.data());
  using scalar_t = typename ekat::ScalarTraits<value_t>::scalar_type;

  HashType accum;
  int beg[4] = {0, 0, 0, 0};
  int end[4] = {v.extent_int(0), v.extent_int(2), v.extent_int(3), n4};
  MDRangePolicy<ExecSpace, 4> policy(beg,end);
  auto lambda = KOKKOS_LAMBDA(int i0, int i2, int i3, int i4, HashType& accum) {
    const auto* vcol = reinterpret_cast<const scalar_t*>(&v(i0,i2,i3,0));
    Homme::hash(vcol[i4], accum);
  };
  Kokkos::parallel_reduce(policy,lambda,HashReducer<>(accum));
  hash(accum, accum_out);
}

template<typename DT,typename... Props>
std::enable_if_t<Kokkos::View<DT,Props...>::rank==4>
hash (const Kokkos::View<DT,Props...>& v, HashType& accum_out) {
  HashType accum;
  int beg[3] = {0, 0, 0};
  int end[3] = {v.extent_int(0), v.extent_int(2), v.extent_int(3)};
  MDRangePolicy<ExecSpace, 3> policy(beg,end);
  auto lambda = KOKKOS_LAMBDA(int i0, int i2, int i3, HashType& accum) {
    Homme::hash(v(i0,i2,i3), accum);
  };
  Kokkos::parallel_reduce(policy,lambda,HashReducer<>(accum));
  hash(accum, accum_out);
}

} // Homme

#endif // HASH_HPP
