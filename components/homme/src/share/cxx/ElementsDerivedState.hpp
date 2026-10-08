/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_ELEMENTS_DERIVED_STATE_HPP
#define HOMMEXX_ELEMENTS_DERIVED_STATE_HPP

#include "Types.hpp"

namespace Homme {

// Per element derived data
template<typename ST>
class ElementsDerivedStateST {
public:
  using PT = PackType<ST>;

  ExecViewManaged<PT * [NP][NP][NUM_LEV]>     m_omega_p;  // Scaled 'pressure vertical velocity' (omega=(1/p)*Dp/Dt)
  ExecViewManaged<PT * [2][NP][NP][NUM_LEV]>  m_vn0;      // weighted velocity flux for consistency
  ExecViewManaged<PT * [2][NP][NP][NUM_LEV]>  m_vstar;    // velocity at start of tracer time step

  // eta=$\eta$ is the vertical coordinate
  // eta_dot_dpdn = $\dot{eta}\frac{dp}{d\eta}$
  // Note: the last level (surface) is always 0.
  ExecViewManaged<PT * [NP][NP][NUM_LEV_P]>   m_eta_dot_dpdn;

  ExecViewManaged<PT * [NP][NP][NUM_LEV]>     m_dp;                // for dp_tracers at physics timestep
  ExecViewManaged<PT * [NP][NP][NUM_LEV]>     m_divdp;             // divergence of dp
  ExecViewManaged<PT * [NP][NP][NUM_LEV]>     m_divdp_proj;        // DSSed divdp
  ExecViewManaged<PT * [NP][NP][NUM_LEV]>     m_dpdiss_biharmonic; // mean dp dissipation tendency, if nu_p>0
  ExecViewManaged<PT * [NP][NP][NUM_LEV]>     m_dpdiss_ave;        // mean dp used to compute psdiss_tens

  ElementsDerivedStateST() : m_num_elems(0) {}

  void init (const int num_elems);

  void randomize(const int seed, const Real dp3d_min);

  // Copy values from one ElementsDerivedStateST struct to another. All derivs get set to 0.
  template<typename RST>
  void import_values (const ElementsDerivedStateST<RST>& rhs);

  KOKKOS_INLINE_FUNCTION
  int num_elems() const { return m_num_elems; }

private:
  int m_num_elems;
};

using ElementsDerivedState = ElementsDerivedStateST<ScalarValue>;

// Copy values from one ElementStateST struct to another. All derivs get set to 0.
template<typename ST>
template<typename RST>
void ElementsDerivedStateST<ST>::import_values (const ElementsDerivedStateST<RST>& rhs)
{
  if constexpr (std::is_same_v<ST,RST>) {
    const void* lhs_ptr = this;
    const void* rhs_ptr = &rhs;
    if (lhs_ptr==rhs_ptr)
      return;
  }
  auto lhs_ome = ekat::scalarize(m_omega_p);
  auto lhs_vn0 = ekat::scalarize(m_vn0);
  auto lhs_vst = ekat::scalarize(m_vstar);
  auto lhs_eta = ekat::scalarize(m_eta_dot_dpdn);
  auto lhs_dp  = ekat::scalarize(m_dp);
  auto lhs_div = ekat::scalarize(m_divdp);
  auto lhs_pro = ekat::scalarize(m_divdp_proj);
  auto lhs_bih = ekat::scalarize(m_dpdiss_biharmonic);
  auto lhs_ave = ekat::scalarize(m_dpdiss_ave);

  auto rhs_ome = ekat::scalarize(rhs.m_omega_p);
  auto rhs_vn0 = ekat::scalarize(rhs.m_vn0);
  auto rhs_vst = ekat::scalarize(rhs.m_vstar);
  auto rhs_eta = ekat::scalarize(rhs.m_eta_dot_dpdn);
  auto rhs_dp  = ekat::scalarize(rhs.m_dp);
  auto rhs_div = ekat::scalarize(rhs.m_divdp);
  auto rhs_pro = ekat::scalarize(rhs.m_divdp_proj);
  auto rhs_bih = ekat::scalarize(rhs.m_dpdiss_biharmonic);
  auto rhs_ave = ekat::scalarize(rhs.m_dpdiss_ave);

  int nlev = NUM_PHYSICAL_LEV;
  auto copy = KOKKOS_LAMBDA(int ie, int ip, int jp, int k) {
    lhs_ome(ie,ip,jp,k)   = ADValue(rhs_ome(ie,ip,jp,k));
    lhs_vn0(ie,0,ip,jp,k) = ADValue(rhs_vn0(ie,0,ip,jp,k));
    lhs_vn0(ie,1,ip,jp,k) = ADValue(rhs_vn0(ie,1,ip,jp,k));
    lhs_vst(ie,0,ip,jp,k) = ADValue(rhs_vst(ie,0,ip,jp,k));
    lhs_vst(ie,1,ip,jp,k) = ADValue(rhs_vst(ie,1,ip,jp,k));
    lhs_eta(ie,ip,jp,k)   = ADValue(rhs_eta(ie,ip,jp,k));
    lhs_dp (ie,ip,jp,k)   = ADValue(rhs_dp (ie,ip,jp,k));
    lhs_div(ie,ip,jp,k)   = ADValue(rhs_div(ie,ip,jp,k));
    lhs_pro(ie,ip,jp,k)   = ADValue(rhs_pro(ie,ip,jp,k));
    lhs_bih(ie,ip,jp,k)   = ADValue(rhs_bih(ie,ip,jp,k));
    lhs_ave(ie,ip,jp,k)   = ADValue(rhs_ave(ie,ip,jp,k));

    if (k==0) {
      lhs_eta(ie,ip,jp,nlev) = ADValue(rhs_eta(ie,ip,jp,nlev));
    }
  };
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> p({0,0,0,0},{m_num_elems,NP,NP,nlev});
  Kokkos::parallel_for(p,copy);
}

} // Homme

#endif // HOMMEXX_ELEMENTS_DERIVED_STATE_HPP
