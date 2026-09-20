/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_EULER_STEP_FUNCTOR_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_HPP

#include <memory>

#include "Types.hpp"
#include "SimulationParams.hpp"

namespace Homme {

template<typename ST>
class EulerStepFunctorImplST;

struct FunctorsBuffersManager;

template<typename ST>
class EulerStepFunctorST {
  std::shared_ptr<EulerStepFunctorImplST<ST>> p_;

public:
  EulerStepFunctorST();

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

  // Exposes the concrete impl object directly, for callers that need
  // functionality not forwarded above (e.g. euler_step_adj/
  // set_tape_for_adjoint, battleplan Step 4's adjoint of euler_step, which
  // -- like HyperviscosityFunctorImplST::run_JtV -- is Real-only and
  // reached this way rather than via a pimpl-forwarded wrapper method, to
  // avoid the explicit-instantiation-doesn't-cover-member-templates pitfall
  // that SFINAE'd wrapper methods would hit). Callers must include
  // "EulerStepFunctorImpl.hpp" themselves to use the returned pointer.
  std::shared_ptr<EulerStepFunctorImplST<ST>> get_impl () { return p_; }

private:
  bool is_setup;
};

using EulerStepFunctor = EulerStepFunctorST<ScalarValue>;

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
