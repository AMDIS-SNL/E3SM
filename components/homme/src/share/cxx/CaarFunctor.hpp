/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_CAAR_FUNCTOR_HPP
#define HOMMEXX_CAAR_FUNCTOR_HPP

#include "Elements.hpp"
#include "Tracers.hpp"
#include "SphereOperators.hpp"
#include "Types.hpp"
#include "RKStageData.hpp"

#include <memory>
#include <any>

namespace Homme {

class ReferenceElement;
class SimulationParams;
class HybridVCoord;

class MpiBuffersManager;
struct FunctorsBuffersManager;

template<typename ST>
class CaarFunctorST {
public:
  CaarFunctorST();
  CaarFunctorST(const ElementsST<ST> &elements, const TracersST<ST> &tracers,
                const ReferenceElement &ref_FE, const HybridVCoord &hvcoord,
                const SphereOperatorsST<ST> &sphere_ops, const SimulationParams& params);
  CaarFunctorST(const int num_elems, const SimulationParams& params);
  CaarFunctorST(const CaarFunctorST &) = delete;

  ~CaarFunctorST() = default;

  CaarFunctorST &operator=(const CaarFunctorST &) = delete;

  bool setup_needed() { return !is_setup; }
  void setup(const ElementsST<ST> &elements, const TracersST<ST> &tracers,
             const ReferenceElement &ref_FE, const HybridVCoord &hvcoord,
             const SphereOperatorsST<ST> &sphere_ops);

  int requested_buffer_size () const;
  void init_buffers(const FunctorsBuffersManager& fbm);
  void init_boundary_exchanges(const std::shared_ptr<MpiBuffersManager> &bm_exchange);

  void set_rk_stage_data(const RKStageData& data);

  void run(const RKStageData& data);

  std::any& impl () { return m_caar_impl; }
private:
  std::any m_caar_impl;
  bool is_setup;
};

using CaarFunctor = CaarFunctorST<ScalarValue>;

} // Namespace Homme

#endif // HOMMEXX_CAAR_FUNCTOR_HPP
