/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_HYPERVISCOSITY_FUNCTOR_DEF_HPP
#define HOMMEXX_HYPERVISCOSITY_FUNCTOR_DEF_HPP

#include "HyperviscosityFunctor.hpp"
#include "HyperviscosityFunctorImpl.hpp"
#include "FunctorsBuffersManager.hpp"

#include "Context.hpp"


namespace Homme
{

template<typename ST>
HyperviscosityFunctorST<ST>::HyperviscosityFunctorST ()
  : is_setup(true)
{
  auto& c = Context::singleton();
  auto& params   = c.get<SimulationParams>();
  auto& geometry = c.get<ElementsGeometry>();
  auto& state    = c.get<ElementsStateST<ST>>();
  auto& derived  = c.get<ElementsDerivedStateST<ST>>();

  m_hvf_impl = std::make_any<HyperviscosityFunctorImplST<ST>>(params,geometry,state,derived);
}

// This constructor is useful for using buffer functionality without
// having all other Functor information available.
// If this constructor is used, the setup() function must be called
// before using any other HyperviscosityFunctor functions.
template<typename ST>
HyperviscosityFunctorST<ST>::HyperviscosityFunctorST(const int num_elems, const SimulationParams& params)
  : is_setup(false)
{
  // Build functor impl
  m_hvf_impl = std::make_any<HyperviscosityFunctorImplST<ST>>(num_elems,params);
}

template<typename ST>
void HyperviscosityFunctorST<ST>::
setup(const ElementsGeometry &geometry,
      const ElementsStateST<ST> &state,
      const ElementsDerivedStateST<ST> &derived)
{
  // Sanity check
  assert (!is_setup);

  auto impl = std::any_cast<HyperviscosityFunctorImplST<ST>>(&m_hvf_impl);
  impl->setup(geometry, state, derived);
  is_setup = true;
}

template<typename ST>
int HyperviscosityFunctorST<ST>::requested_buffer_size () const {
  auto impl = std::any_cast<HyperviscosityFunctorImplST<ST>>(&m_hvf_impl);
  return impl->requested_buffer_size();
}

template<typename ST>
void HyperviscosityFunctorST<ST>::init_buffers (const FunctorsBuffersManager& fbm) {
  auto impl = std::any_cast<HyperviscosityFunctorImplST<ST>>(&m_hvf_impl);
  impl->init_buffers(fbm);
}

template<typename ST>
void HyperviscosityFunctorST<ST>::init_boundary_exchanges () {
  auto impl = std::any_cast<HyperviscosityFunctorImplST<ST>>(&m_hvf_impl);
  impl->init_boundary_exchanges();
}

template<typename ST>
void HyperviscosityFunctorST<ST>::run (const int np1, const Real dt, const Real eta_ave_w)
{
  auto impl = std::any_cast<HyperviscosityFunctorImplST<ST>>(&m_hvf_impl);
  impl->run(np1,dt,eta_ave_w);
}

} // namespace Homme

#endif // HOMMEXX_HYPERVISCOSITY_FUNCTOR_DEF_HPP
