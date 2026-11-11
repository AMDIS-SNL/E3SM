/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMME_PRIM_ADVANCE_EXP_HPP
#define HOMME_PRIM_ADVANCE_EXP_HPP

#include "TimeLevel.hpp"
#include "Types.hpp"

namespace Homme
{

// Advance the dynamics by one time step dt (defined in prim_advance_exp.cpp).
// Templated on the scalar type, to match the theta-l model (so that code shared among
// models, like prim_step, can call prim_advance_exp<ST>). Only ST=ScalarValue is available
// (checked at runtime, so that the shared code can be instantiated for other types too).
template<typename ST = ScalarValue>
void prim_advance_exp (TimeLevel& tl, const Real dt, const bool compute_diagnostics);

} // namespace Homme

#endif // HOMME_PRIM_ADVANCE_EXP_HPP
