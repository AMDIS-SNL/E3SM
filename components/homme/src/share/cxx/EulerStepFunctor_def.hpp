/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_EULER_STEP_FUNCTOR_DEF_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_DEF_HPP

#include "EulerStepFunctorImpl.hpp"

namespace Homme {

template<typename ST>
EulerStepFunctorST<ST>::
EulerStepFunctorST(const int num_elems)
{
  m_impl = std::make_any<EulerStepFunctorImplST<ST>>(num_elems);
}

template<typename ST>
void EulerStepFunctorST<ST>::setup()
{
  // Sanity check
  assert (!is_setup);

  auto& impl = std::any_cast<EulerStepFunctorImplST<ST>&>(m_impl);
  impl.setup();
  is_setup = true;
}

template<typename ST>
void EulerStepFunctorST<ST>::reset (const SimulationParams& params)
{
  auto& impl = std::any_cast<EulerStepFunctorImplST<ST>&>(m_impl);
  impl.reset(params);
}

template<typename ST>
int EulerStepFunctorST<ST>::requested_buffer_size () const
{
  const auto& impl = std::any_cast<const EulerStepFunctorImplST<ST>&>(m_impl);
  return impl.requested_buffer_size();
}
template<typename ST>
void EulerStepFunctorST<ST>::init_buffers (const FunctorsBuffersManager& fbm)
{
  auto& impl = std::any_cast<EulerStepFunctorImplST<ST>&>(m_impl);
  impl.init_buffers(fbm);
}

template<typename ST>
void EulerStepFunctorST<ST>::init_boundary_exchanges ()
{
  // The Functor needs to be fully setup to use this function
  assert (is_setup);

  auto& impl = std::any_cast<EulerStepFunctorImplST<ST>&>(m_impl);
  impl.init_boundary_exchanges();
}

template<typename ST>
void EulerStepFunctorST<ST>::precompute_divdp ()
{
  // The Functor needs to be fully setup to use this function
  assert (is_setup);

  auto& impl = std::any_cast<EulerStepFunctorImplST<ST>&>(m_impl);
  impl.precompute_divdp();
}

template<typename ST>
void EulerStepFunctorST<ST>::
euler_step (const int np1_qdp, const int n0_qdp, const Real dt,
              const Real rhs_multiplier, const DSSOption DSSopt)
{
  // The Functor needs to be fully setup to use this function
  assert (is_setup);

  auto& impl = std::any_cast<EulerStepFunctorImplST<ST>&>(m_impl);
  impl.euler_step(np1_qdp, n0_qdp, dt, rhs_multiplier, DSSopt);
}

template<typename ST>
void EulerStepFunctorST<ST>::
qdp_time_avg (const int n0_qdp, const int np1_qdp)
{
  // The Functor needs to be fully setup to use this function
  assert (is_setup);

  auto& impl = std::any_cast<EulerStepFunctorImplST<ST>&>(m_impl);
  impl.qdp_time_avg(n0_qdp, np1_qdp);
}

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_DEF_HPP
