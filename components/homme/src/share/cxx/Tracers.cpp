/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Tracers.hpp"
#include "Tracers_def.hpp"

namespace Homme {

// ETI for the commonly used scalar type
template class TracersST<Real>;

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template class TracersST<DpFadType>;
template class TracersST<DxFadTypeCaar>;
template class TracersST<DxFadTypeRemap>;
#endif

} // namespace Homme
