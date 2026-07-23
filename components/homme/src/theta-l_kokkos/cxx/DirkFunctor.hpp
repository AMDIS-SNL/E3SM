/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_DIRK_FUNCTOR_HPP
#define HOMMEXX_DIRK_FUNCTOR_HPP

#include "Elements.hpp"
#include "Types.hpp"
#include <memory>
#include <any>

namespace Homme {

class FunctorsBuffersManager;

class HybridVCoord;

template<typename ST>
class DirkFunctorST {
public:
  DirkFunctorST(const int nelem);
  DirkFunctorST(const DirkFunctorST &) = delete;
  DirkFunctorST &operator=(const DirkFunctorST &) = delete;

  ~DirkFunctorST();

  int requested_buffer_size() const;
  void init_buffers(const FunctorsBuffersManager& fbm);

  // Top-level interface, equivalent to compute_stage_value_dirk.
  void run(int nm1, Real alphadt_nm1, int n0, Real alphadt_n0, int np1, Real dt2,
           const ElementsST<ST>& elements, const HybridVCoord& hvcoord);

  std::any& impl () { return m_dirk_impl; }
private:
  std::any m_dirk_impl;
};

using DirkFunctor = DirkFunctorST<ScalarValue>;

} // Namespace Homme

#endif // HOMMEXX_DIRK_FUNCTOR_HPP
