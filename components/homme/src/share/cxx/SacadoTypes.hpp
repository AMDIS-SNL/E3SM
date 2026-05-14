#ifndef HOMMEXX_SACADO_TYPES_HPP
#define HOMMEXX_SACADO_TYPES_HPP

#include "Hommexx_config.h"
#include "Dimensions.hpp"

#ifdef HOMMEXX_ENABLE_FAD_TYPES

#include <ekat_scalar_traits.hpp>

// Disable view specializations
#define SACADO_DISABLE_FAD_VIEW_SPEC

#include <Sacado.hpp>

namespace Homme {

// TODO: decide what to do in terms of which Fad to support (SFad, SLFad, DFad, etc)
template<typename T,int N>
using SFadN = Sacado::Fad::SFad<T,N>;

constexpr int DpFadSize = HOMMEXX_DP_SFAD_SIZE;
using DpFadType = SFadN<double,DpFadSize>;

// The fad type for the deriv w.r.t. state vars of some functors
using DxFadTypeCaar = SFadN<double,19*NP*NP>; // for JV
//using DxFadTypeCaar = SFadN<double,1568>; // for JV_full

// The fad type for the deriv w.r.t. state vars of the DIRK functor.
// DIRK has no horizontal (GaussPoint) coupling: each column (ip,jp) is independent.
// The stencil encodes {u,v,vtheta_dp,dp3d} at NUM_PHYSICAL_LEV midpoints plus
// {w_i,phinh_i} at NUM_INTERFACE_LEV interface levels, all per column.
using DxFadTypeDirk = SFadN<double, NUM_PHYSICAL_LEV*4 + NUM_INTERFACE_LEV*2>;

template<typename T, int N>
KOKKOS_INLINE_FUNCTION
T ADValue(const SFadN<T,N>& v) { return v.val(); }

template<typename Expr>
KOKKOS_INLINE_FUNCTION
auto ADValue(const Expr& e)
 -> std::enable_if_t<Sacado::IsExpr<Expr>::value,decltype(e.val())>
{
  return e.val();
}

template<typename T>
struct DerivSz {
  static constexpr int value = 0;
};

template<typename T, int N>
struct DerivSz<SFadN<T,N>> {
  static constexpr int value = N;
};

} // namespace Homme

namespace ekat {
// Specialization of ScalarTraits struct for Sacado SFad types
template<typename T, int N>
struct ScalarTraits<Homme::SFadN<T,N>>
{
  using value_type  = Homme::SFadN<T,N>;
  using scalar_type = value_type;

  static constexpr bool is_simd = false;

  static constexpr bool is_floating_point = ekat::ScalarTraits<T>::is_floating_point;

  static constexpr bool specialized = true;
};
} // namespace ekat

#endif // HOMMEXX_ENABLE_FAD_TYPES

#endif // HOMMEXX_SACADO_TYPES_HPP
