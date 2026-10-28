/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ElementsState.hpp"
#include "ElementsState_def.hpp"

#include "Types.hpp"

namespace Homme {

void RefStates::init(const int num_elems) {
  dp_ref       = decltype(dp_ref)      ("dp_ref",      num_elems);
  phi_i_ref    = decltype(phi_i_ref)   ("phi_i_ref",   num_elems);
  theta_ref    = decltype(theta_ref)   ("theta_ref",   num_elems);
  nu_scale_top = decltype(nu_scale_top)("nu_scale_top");
  nu_scale_top_ilev_pack_lim = 0;

  m_num_elems = num_elems;

  m_policy = get_default_team_policy<ExecSpace>(num_elems);
  m_tu     = TeamUtils<ExecSpace>(m_policy);
}

// ETI for the commonly used scalar type
template class ElementsStateST<Real>;

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template class ElementsStateST<DpFadType>;
template class ElementsStateST<DxFadTypeCaar>;
template class ElementsStateST<DxFadTypeDirk>;
#endif

} // namespace Homme
