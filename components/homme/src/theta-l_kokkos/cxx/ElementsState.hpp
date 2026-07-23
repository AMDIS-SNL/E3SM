/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_ELEMENTS_STATE_HPP
#define HOMMEXX_ELEMENTS_STATE_HPP

#include "StateSnapshot.hpp"
#include "Types.hpp"
#include "kokkos_utils.hpp"
#include "utilities/Hash.hpp"

#include <ekat_pack_kokkos.hpp>
#include <ekat_assert.hpp>

namespace Homme {

class HybridVCoord;

// Reference states, needed in HV and vert remap
struct RefStates {
  ExecViewManaged<RPack * [NP][NP][NUM_LEV_P]> phi_i_ref;
  ExecViewManaged<RPack * [NP][NP][NUM_LEV  ]> theta_ref;
  ExecViewManaged<RPack * [NP][NP][NUM_LEV  ]> dp_ref;

  RefStates () = default;

  void init (const int num_elems);

  int num_elems () const { return m_num_elems; }
private:
  int m_num_elems;
};

/* Per element data - specific velocity, temperature, pressure, etc. */
template<typename ST>
class ElementsStateST {
public:
  using PT = PackType<ST>;

  ExecViewManaged<PT * [2][NP][NP][NUM_LEV  ]> m_v;          // Horizontal velocity
  ExecViewManaged<PT *    [NP][NP][NUM_LEV_P]> m_w_i;        // Vertical velocity at interfaces
  ExecViewManaged<PT *    [NP][NP][NUM_LEV  ]> m_vtheta_dp;  // Virtual potential temperature (mass)
  ExecViewManaged<PT *    [NP][NP][NUM_LEV_P]> m_phinh_i;    // Geopotential used by NH model at interfaces
  ExecViewManaged<PT *    [NP][NP][NUM_LEV  ]> m_dp3d;       // Delta p on levels

  ExecViewManaged<ST *    [NP][NP]           > m_ps_v;       // Surface pressure

  ElementsStateST() = default;

  void init(const int num_elems);

  void randomize(const int seed);
  void randomize(const int seed, const Real max_pressure);
  void randomize(const int seed, const Real max_pressure, const Real ps0, const Real hyai0);
  void randomize(const int seed, const Real max_pressure, const Real ps0, const Real hyai0,
                 const ExecViewUnmanaged<const Real*[NP][NP]>& phis);
  void randomize(const int seed, const HybridVCoord& hvcoord);

  KOKKOS_INLINE_FUNCTION
  int num_elems() const { return m_num_elems; }

  // Fill the exec space views with data coming from F90 pointers
  void pull_from_f90_pointers(CF90Ptr& state_v,         CF90Ptr& state_w_i,
                              CF90Ptr& state_vtheta_dp, CF90Ptr& state_phinh_i,
                              CF90Ptr& state_dp3d,      CF90Ptr& state_ps_v,
                              int itl);

  // Push the results from the exec space views to the F90 pointers
  void push_to_f90_pointers(F90Ptr& state_v, F90Ptr& state_w_i, F90Ptr& state_vtheta_dp,
                            F90Ptr& state_phinh_i, F90Ptr& state_dp,
                            int itl) const;

  HashType hash() const;

  // Copy values from one ElementStateST struct to another. All derivs get set to 0.
  void deep_copy(const ElementsStateST<ST>& rhs);

  // Check ElementsState for NaN or incorrectly signed values. The initial check
  // is fast and on device. If everything is fine, the routine returns
  // immediately. If there is a bad value, a subsequent check is run on the host,
  // and this check prints detailed information to a file called
  // hommexx.errlog.${rank}. Then EKAT_ERROR_MSG is called with a message pointing
  // to this file.
  void check_print_abort_on_bad_elems(const std::string& label) const;    // string to ID call site

#ifdef HOMMEXX_ENABLE_FAD_TYPES
  void randomize_derivs(const int seed);

  ElementsStateST<Real> take_deriv_snapshot (int ider);
  void take_deriv_snapshot (ElementsStateST<Real>& snap, int ider);
  ElementsStateST<Real> take_value_snapshot ();
  void take_value_snapshot (ElementsStateST<Real>& snap);
#endif

private:
  int m_num_elems;
};

using ElementsState = ElementsStateST<ScalarValue>;

} // Homme

#endif // HOMMEXX_ELEMENTS_STATE_HPP
