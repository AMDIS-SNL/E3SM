/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP
#define HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP

#include "Types.hpp"
#include "ElementsGeometry.hpp"
#include "ElementsState.hpp"
#include "ElementsDerivedState.hpp"
#include "SimulationParams.hpp"

#include <memory>

namespace Homme
{

template<typename ST>
class HyperviscosityFunctorImplST;

struct FunctorsBuffersManager;
class SimulationParams;

template<typename ST>
class HyperviscosityFunctorST
{
public:

  HyperviscosityFunctorST ();

  HyperviscosityFunctorST (const int num_elems, const SimulationParams& params);

  ~HyperviscosityFunctorST ();

  bool setup_needed() { return !is_setup; }
  void setup(const ElementsGeometry&           geometry,
             const ElementsStateST<ST>&        state,
             const ElementsDerivedStateST<ST>& derived);

  int requested_buffer_size () const;
  void init_buffers    (const FunctorsBuffersManager& fbm);

  void init_boundary_exchanges();

  void run (const int np1, const Real dt, const Real eta_ave_w);

  // Access to the impl, needed by tangent/adjoint callers (init_J/run_JV/run_JtV
  // only live on the impl, not on this thin wrapper).
  HyperviscosityFunctorImplST<ST>& impl ();

private:

  std::unique_ptr<HyperviscosityFunctorImplST<ST>>  m_hvf_impl;
  bool is_setup;
};

using HyperviscosityFunctor = HyperviscosityFunctorST<ScalarValue>;

} // namespace Homme

#endif // HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP
