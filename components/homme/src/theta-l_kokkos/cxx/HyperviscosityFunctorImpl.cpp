/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "HyperviscosityFunctorImpl.hpp"
#include "HyperviscosityFunctorImpl_def.hpp"

#include "Types.hpp"

namespace Homme {

// ETI for the commonly used scalar type
template class HyperviscosityFunctorImplST<Real>;

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template class HyperviscosityFunctorImplST<DpFadType>;
#endif

// HVF is self adjoint after vtheta=vtheta_dp/dp and before scaling back.
// But dure to that scaling/rescaling, the derivs contain new terms.
// Since we can compute those derivs terms manually, there's no need to instantiate
// HVF on a "DxFad" type (like we do for CAAR/DIRK). Also, since Jac-related
// methods don't require Fad types, let's instantiate them only for Real scalar type
template void HyperviscosityFunctorImplST<Real>::init_J<Real>(const int);
template void HyperviscosityFunctorImplST<Real>::run_JV<Real>(const int, const StateSnapshot&, StateSnapshot&);
template void HyperviscosityFunctorImplST<Real>::run_JtV<Real>(const int, const StateSnapshot&, StateSnapshot&);

} // namespace Homme
