/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_PRIM_STEP_HPP
#define HOMMEXX_PRIM_STEP_HPP

#include "Elements.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "Types.hpp"

namespace Homme
{

// Defined elsewhere (prim_advance_exp.cpp, prim_advec_tracers_remap.cpp, vertical_remap.cpp)
void prim_advance_exp (TimeLevel& tl, const Real dt, const bool compute_diagnostics);
void prim_advec_tracers_observe_velocity (const int step);
void prim_advec_tracers_remap (const Real dt);
void vertical_remap (const Real dt);

// Defined in prim_step.cpp
void prim_step (const Real dt, const bool compute_diagnostics);
void prim_step_flexible (const Real dt, const bool compute_diagnostics);

// Initialize the mean flux accumulation variables, and save some variables at n0,
// for use by advection. Templated on the scalar type so that it can be used on
// Fad-typed elements too (e.g., in adjoint unit tests).
template<typename ST>
void set_tracer_transport_derived_values (const SimulationParams& params,
                                          const ElementsST<ST>& elements,
                                          const TimeLevel& tl)
{
  const auto eta_dot_dpdn = elements.m_derived.m_eta_dot_dpdn;
  const auto derived_vn0 = elements.m_derived.m_vn0;
  const auto omega_p = elements.m_derived.m_omega_p;
  const auto derived_dpdiss_ave = elements.m_derived.m_dpdiss_ave;
  const auto derived_dpdiss_biharmonic = elements.m_derived.m_dpdiss_biharmonic;
  const auto derived_dp = elements.m_derived.m_dp;
  const auto dp3d = elements.m_state.m_dp3d;
  const auto vstar = elements.m_derived.m_vstar;
  const auto v = elements.m_state.m_v;
  const auto n0 = tl.n0;
  const bool nu_p_pos = params.nu_p>0;
  const bool transport_alg_pos = params.transport_alg>0;
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace> (0,elements.num_elems()*NP*NP*NUM_LEV),
                       KOKKOS_LAMBDA(const int idx) {
    const int ie   = ((idx / NUM_LEV) / NP) / NP;
    const int igp  = ((idx / NUM_LEV) / NP) % NP;
    const int jgp  =  (idx / NUM_LEV) % NP;
    const int ilev =   idx % NUM_LEV;
    eta_dot_dpdn(ie,igp,jgp,ilev) = 0;
    derived_vn0(ie,0,igp,jgp,ilev) = 0;
    derived_vn0(ie,1,igp,jgp,ilev) = 0;
    omega_p(ie,igp,jgp,ilev) = 0;
    if (nu_p_pos) {
      derived_dpdiss_ave(ie,igp,jgp,ilev) = 0;
      derived_dpdiss_biharmonic(ie,igp,jgp,ilev) = 0;
    }
    derived_dp(ie,igp,jgp,ilev) = dp3d(ie,n0,igp,jgp,ilev);
    if (transport_alg_pos) {
      vstar(ie,0,igp,jgp,ilev) = v(ie,n0,0,igp,jgp,ilev);
      vstar(ie,1,igp,jgp,ilev) = v(ie,n0,1,igp,jgp,ilev);
    }
  });
  Kokkos::fence();
}

} // namespace Homme

#endif // HOMMEXX_PRIM_STEP_HPP
