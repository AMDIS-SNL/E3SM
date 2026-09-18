#ifndef HOMME_ELEMENTS_FORCING_DEF_HPP
#define HOMME_ELEMENTS_FORCING_DEF_HPP

#include "ElementsForcing.hpp"

#include "utilities/TestUtils.hpp"
#include <random>

namespace Homme {

template<typename ST>
void ElementsForcingST<ST>::init (const int num_elems) {
  // Sanity check
  assert (num_elems>0);

  m_num_elems = num_elems;

  m_fm      = ExecViewManaged<PT * [3][NP][NP][NUM_LEV]  >("F_Momentum",    m_num_elems);
  m_fvtheta = ExecViewManaged<PT *    [NP][NP][NUM_LEV]  >("F_VirtualPotentialTemperature", m_num_elems);
  m_ft      = ExecViewManaged<PT *    [NP][NP][NUM_LEV]  >("F_Temperature", m_num_elems);
  m_fphi    = ExecViewManaged<PT *    [NP][NP][NUM_LEV_P]>("F_Phi",         m_num_elems);
}

template<typename ST>
void ElementsForcingST<ST>::randomize (const int seed, const Real min_f, const Real max_f) {
  // Check forcing was inited
  assert (m_num_elems>0);
  assert (max_f > min_f);

  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<Real> random_dist(min_f, max_f);

  genRandArray(m_fm,      engine, random_dist);
  genRandArray(m_fvtheta, engine, random_dist);
  genRandArray(m_ft,      engine, random_dist);
  genRandArray(m_fphi,    engine, random_dist);
}

template<typename ST>
void ElementsForcingST<ST>::zero () {
  // Kokkos::deep_copy(view,0) needs an implicit conversion from int all the
  // way to the view's value_type (PT = PackType<ST>). For ST=Real that's a
  // single user-defined conversion (int->Real->PT) and compiles fine, but
  // for a Fad-typed ST (see ElementsForcing.cpp's explicit instantiations
  // under HOMMEXX_ENABLE_FAD_TYPES) it would need two chained user-defined
  // conversions (int->ST->PT), which C++ won't do implicitly. Constructing
  // ST(0) explicitly first leaves only the one (ST->PT) conversion.
  Kokkos::deep_copy(m_fm,ST(0));
  Kokkos::deep_copy(m_fvtheta,ST(0));
  Kokkos::deep_copy(m_ft,ST(0));
  Kokkos::deep_copy(m_fphi,ST(0));
}

} // namespace Homme

#endif // HOMME_ELEMENTS_FORCING_DEF_HPP
