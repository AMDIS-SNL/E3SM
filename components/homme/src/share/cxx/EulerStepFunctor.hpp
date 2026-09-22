/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_EULER_STEP_FUNCTOR_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_HPP

#include "Types.hpp"
#include "SimulationParams.hpp"

#include <any>

namespace Homme {

struct FunctorsBuffersManager;

template<typename ST>
class EulerStepFunctorST {
public:
  EulerStepFunctorST(const int num_elems);

  bool setup_needed() { return !is_setup; }
  void setup();

  void reset(const SimulationParams& params);

  int requested_buffer_size () const;
  void init_buffers    (const FunctorsBuffersManager& fbm);
  void init_boundary_exchanges();

  void precompute_divdp();

  void qdp_time_avg(const int n0_qdp, const int np1_qdp);

  void euler_step(const int np1_qdp, const int n0_qdp, const Real dt,
                  const Real rhs_multiplier, const DSSOption DSSopt);

  KOKKOS_INLINE_FUNCTION
  static bool is_quasi_monotone (const int& limiter_option) {
    return limiter_option == 8 || limiter_option == 9;
  }

  // Exposes the concrete impl object, but still wrapped inside a std::any
  std::any& get_impl () { return m_impl; }

private:
  std::any m_impl;

  bool is_setup = false;
};

using EulerStepFunctor = EulerStepFunctorST<ScalarValue>;

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
