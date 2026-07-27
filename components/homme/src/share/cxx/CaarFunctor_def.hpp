/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMME_CAAR_FUNCTOR_DEF_HPP
#define HOMME_CAAR_FUNCTOR_DEF_HPP

#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"
#include "Context.hpp"
#include "ErrorDefs.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "ReferenceElement.hpp"
#include "Tracers.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "profiling.hpp"

#include <assert.h>
#include <type_traits>

namespace Homme {

template<typename ST>
CaarFunctorST<ST>::
CaarFunctorST()
  : is_setup(false)
{
  auto& geo     = Context::singleton().get<ElementsGeometry>();
  auto& params  = Context::singleton().get<SimulationParams>();

  // Build functor impl
  m_caar_impl = std::make_any<CaarFunctorImplST<ST>>(geo.num_elems(),params);
}

template<typename ST>
CaarFunctorST<ST>::
CaarFunctorST(const ElementsST<ST> &elements, const TracersST<ST> &tracers,
              const ReferenceElement &ref_FE,
              const HybridVCoord &hvcoord,
              const SphereOperatorsST<ST> &sphere_ops,
              const SimulationParams& params)
  : is_setup(true)
{
  // Build functor impl
  m_caar_impl = std::make_any<CaarFunctorImplST<ST>>(elements, tracers, ref_FE, hvcoord, sphere_ops, params);
}

// This constructor is useful for using buffer functionality without
// having all other Functor information available.
// If this constructor is used, the setup() function must be called
// before using any other CaarFunctorST functions.
template<typename ST>
CaarFunctorST<ST>::
CaarFunctorST(const int num_elems, const SimulationParams& params)
  : is_setup(false)
{
  // Build functor impl
  m_caar_impl = std::make_any<CaarFunctorImplST<ST>>(num_elems,params);
}

template<typename ST>
void CaarFunctorST<ST>::
setup(const ElementsST<ST> &elements, const TracersST<ST> &tracers,
      const ReferenceElement &ref_FE, const HybridVCoord &hvcoord,
      const SphereOperatorsST<ST> &sphere_ops)
{
  // Sanity check
  assert (!is_setup);

  auto impl = std::any_cast<CaarFunctorImplST<ST>>(&m_caar_impl);
  impl->setup(elements, tracers, ref_FE, hvcoord, sphere_ops);
  is_setup = true;
}

template<typename ST>
int CaarFunctorST<ST>::requested_buffer_size () const {
  assert (is_setup);
  auto impl = std::any_cast<CaarFunctorImplST<ST>>(&m_caar_impl);
  return impl->requested_buffer_size();
}

template<typename ST>
void CaarFunctorST<ST>::init_buffers(const FunctorsBuffersManager& fbm) {
  assert (is_setup);
  auto impl = std::any_cast<CaarFunctorImplST<ST>>(&m_caar_impl);
  impl->init_buffers(fbm);
}

template<typename ST>
void CaarFunctorST<ST>::init_boundary_exchanges (const std::shared_ptr<MpiBuffersManager>& bm_exchange) {
  // The Functor needs to be fully setup to use this function
  assert (is_setup);
  auto impl = std::any_cast<CaarFunctorImplST<ST>>(&m_caar_impl);
  impl->init_boundary_exchanges(bm_exchange);
}

template<typename ST>
void CaarFunctorST<ST>::set_rk_stage_data (const RKStageData& data)
{
  assert (is_setup);
  auto impl = std::any_cast<CaarFunctorImplST<ST>>(&m_caar_impl);
  impl->set_rk_stage_data(data);
}

template<typename ST>
void CaarFunctorST<ST>::run (const RKStageData& data)
{
  assert (is_setup);
  auto impl = std::any_cast<CaarFunctorImplST<ST>>(&m_caar_impl);
  impl->run(data);
}

} // Namespace Homme

#endif // HOMME_CAAR_FUNCTOR_DEF_HPP
