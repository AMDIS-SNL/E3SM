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

namespace Homme {

class HybridVCoord;

// Reference states, needed in HV and vert remap
struct RefStates {
  ExecViewManaged<RPack * [NP][NP][NUM_LEV_P]> phi_i_ref;
  ExecViewManaged<RPack * [NP][NP][NUM_LEV  ]> theta_ref;
  ExecViewManaged<RPack * [NP][NP][NUM_LEV  ]> dp_ref;

  RefStates () :
    m_num_elems(0)
    , m_policy(get_default_team_policy<ExecSpace>(1))
    , m_tu(m_policy)
  {}

  void init (const int num_elems);

  int num_elems () const { return m_num_elems; }
private:
  int m_num_elems;
  Kokkos::TeamPolicy<ExecSpace> m_policy;
  TeamUtils<ExecSpace> m_tu;
};

/* Per element data - specific velocity, temperature, pressure, etc. */
template<typename ST>
class ElementsStateST {
public:
  using PT = PackType<ST>;

  RefStates m_ref_states;

  ExecViewManaged<PT * [NUM_TIME_LEVELS][2][NP][NP][NUM_LEV  ]> m_v;          // Horizontal velocity
  ExecViewManaged<PT * [NUM_TIME_LEVELS]   [NP][NP][NUM_LEV_P]> m_w_i;        // Vertical velocity at interfaces
  ExecViewManaged<PT * [NUM_TIME_LEVELS]   [NP][NP][NUM_LEV  ]> m_vtheta_dp;  // Virtual potential temperature (mass)
  ExecViewManaged<PT * [NUM_TIME_LEVELS]   [NP][NP][NUM_LEV_P]> m_phinh_i;    // Geopotential used by NH model at interfaces
  ExecViewManaged<PT * [NUM_TIME_LEVELS]   [NP][NP][NUM_LEV  ]> m_dp3d;       // Delta p on levels

  ExecViewManaged<ST * [NUM_TIME_LEVELS]   [NP][NP]           > m_ps_v;       // Surface pressure

  ElementsStateST() :
    m_num_elems(0)
    , m_policy(get_default_team_policy<ExecSpace>(1))
    , m_tu(m_policy)
  {}

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
                              CF90Ptr& state_dp3d,      CF90Ptr& state_ps_v);

  // Push the results from the exec space views to the F90 pointers
  void push_to_f90_pointers(F90Ptr& state_v, F90Ptr& state_w_i, F90Ptr& state_vtheta_dp,
                            F90Ptr& state_phinh_i, F90Ptr& state_dp) const;

  HashType hash(const int time_level) const;

  // Copy values from one ElementStateST struct to another. All derivs get set to 0.
  template<typename RST>
  void import_values (const ElementsStateST<RST>& rhs, int tl);

  StateSnapshot export_values (int tl, bool do_ps = false);

  // Check ElementsState for NaN or incorrectly signed values. The initial check
  // is fast and on device. If everything is fine, the routine returns
  // immediately. If there is a bad value, a subsequent check is run on the host,
  // and this check prints detailed information to a file called
  // hommexx.errlog.${rank}. Then EKAT_ERROR_MSG is called with a message pointing
  // to this file.
  void check_print_abort_on_bad_elems(const std::string& label,    // string to ID call site
                                      const int time_level) const; // time level index in state arrays


#ifdef HOMMEXX_ENABLE_FAD_TYPES
  void randomize_derivs(const int seed, const int itl);

  template<typename RST>
  std::enable_if_t<Sacado::IsFad<RST>::value>
  import_values_from_deriv (const ElementsStateST<RST>& rhs, int tl, int ider);
#endif

private:
  int m_num_elems;
  Kokkos::TeamPolicy<ExecSpace> m_policy;
  TeamUtils<ExecSpace> m_tu;
};

using ElementsState = ElementsStateST<ScalarValue>;

// Copy values from one ElementStateST struct to another. All derivs get set to 0.
template<typename ST>
template<typename RST>
void ElementsStateST<ST>::import_values (const ElementsStateST<RST>& rhs, int tl)
{
  if constexpr (std::is_same_v<ST,RST>) {
    const void* lhs_ptr = this;
    const void* rhs_ptr = &rhs;
    if (lhs_ptr==rhs_ptr)
      return;
  }
  auto lhs_v = ekat::scalarize(m_v);
  auto lhs_dp = ekat::scalarize(m_dp3d);
  auto lhs_phi = ekat::scalarize(m_phinh_i);
  auto lhs_vth = ekat::scalarize(m_vtheta_dp);
  auto lhs_w = ekat::scalarize(m_w_i);
  auto lhs_ps = ekat::scalarize(m_ps_v);

  auto rhs_v = ekat::scalarize(rhs.m_v);
  auto rhs_dp = ekat::scalarize(rhs.m_dp3d);
  auto rhs_phi = ekat::scalarize(rhs.m_phinh_i);
  auto rhs_vth = ekat::scalarize(rhs.m_vtheta_dp);
  auto rhs_w = ekat::scalarize(rhs.m_w_i);
  auto rhs_ps = ekat::scalarize(rhs.m_ps_v);

  int nlev = NUM_PHYSICAL_LEV;
  auto copy = KOKKOS_LAMBDA(int ie, int ip, int jp, int k) {
    lhs_v(ie,tl,0,ip,jp,k) = ADValue(rhs_v(ie,tl,0,ip,jp,k));
    lhs_v(ie,tl,1,ip,jp,k) = ADValue(rhs_v(ie,tl,1,ip,jp,k));

    lhs_w(ie,tl,ip,jp,k)   = ADValue(rhs_w(ie,tl,ip,jp,k));
    lhs_dp(ie,tl,ip,jp,k)  = ADValue(rhs_dp(ie,tl,ip,jp,k));
    lhs_vth(ie,tl,ip,jp,k) = ADValue(rhs_vth(ie,tl,ip,jp,k));
    lhs_phi(ie,tl,ip,jp,k) = ADValue(rhs_phi(ie,tl,ip,jp,k));

    if (k==0) {
      lhs_ps(ie,tl,ip,jp) = ADValue(rhs_ps(ie,tl,ip,jp));
      lhs_w(ie,tl,ip,jp,nlev) = ADValue(rhs_w(ie,tl,ip,jp,nlev));
      lhs_phi(ie,tl,ip,jp,nlev) = ADValue(rhs_phi(ie,tl,ip,jp,nlev));
    }
  };
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> p({0,0,0,0},{m_num_elems,NP,NP,nlev});
  Kokkos::parallel_for(p,copy);
}

#ifdef HOMMEXX_ENABLE_FAD_TYPES
// Extract derivs from an ElementStateST templated on a Fad type into one that has ST=Real
template<typename ST>
template<typename RST>
std::enable_if_t<Sacado::IsFad<RST>::value>
ElementsStateST<ST>::import_values_from_deriv (const ElementsStateST<RST>& rhs, int tl, int ider)
{
  EKAT_REQUIRE_MSG (ider>=0 and ider<DerivSz<RST>::value,
      "[ElementsStateST::import_values_from_deriv] Error! Derivative index (" << ider << ") out of bounds.\n");
  auto lhs_v = ekat::scalarize(m_v);
  auto lhs_dp = ekat::scalarize(m_dp3d);
  auto lhs_phi = ekat::scalarize(m_phinh_i);
  auto lhs_vth = ekat::scalarize(m_vtheta_dp);
  auto lhs_w = ekat::scalarize(m_w_i);
  auto lhs_ps = ekat::scalarize(m_ps_v);

  auto rhs_v = ekat::scalarize(rhs.m_v);
  auto rhs_dp = ekat::scalarize(rhs.m_dp3d);
  auto rhs_phi = ekat::scalarize(rhs.m_phinh_i);
  auto rhs_vth = ekat::scalarize(rhs.m_vtheta_dp);
  auto rhs_w = ekat::scalarize(rhs.m_w_i);
  auto rhs_ps = ekat::scalarize(rhs.m_ps_v);

  int nlev = NUM_PHYSICAL_LEV;
  auto copy = KOKKOS_LAMBDA(int ie, int ip, int jp, int k) {
    lhs_v(ie,tl,0,ip,jp,k) = rhs_v(ie,tl,0,ip,jp,k).fastAccessDx(ider);
    lhs_v(ie,tl,1,ip,jp,k) = rhs_v(ie,tl,1,ip,jp,k).fastAccessDx(ider);

    lhs_w(ie,tl,ip,jp,k)   = rhs_w(ie,tl,ip,jp,k).fastAccessDx(ider);
    lhs_dp(ie,tl,ip,jp,k)  = rhs_dp(ie,tl,ip,jp,k).fastAccessDx(ider);
    lhs_vth(ie,tl,ip,jp,k) = rhs_vth(ie,tl,ip,jp,k).fastAccessDx(ider);
    lhs_phi(ie,tl,ip,jp,k) = rhs_phi(ie,tl,ip,jp,k).fastAccessDx(ider);

    if (k==0) {
      lhs_ps(ie,tl,ip,jp) = rhs_ps(ie,tl,ip,jp).fastAccessDx(ider);
      lhs_w(ie,tl,ip,jp,nlev) = rhs_w(ie,tl,ip,jp,nlev).fastAccessDx(ider);
      lhs_phi(ie,tl,ip,jp,nlev) = rhs_phi(ie,tl,ip,jp,nlev).fastAccessDx(ider);
    }
  };
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> p({0,0,0,0},{m_num_elems,NP,NP,nlev});
  Kokkos::parallel_for(p,copy);
}
#endif // HOMMEXX_ENABLE_FAD_TYPES

} // Homme

#endif // HOMMEXX_ELEMENTS_STATE_HPP
