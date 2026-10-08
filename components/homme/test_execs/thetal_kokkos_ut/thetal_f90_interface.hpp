#ifndef HOMMEXX_THETAL_F90_INTERFACE_HPP
#define HOMMEXX_THETAL_F90_INTERFACE_HPP

#include "Types.hpp"

// Declarations of the bind(c) fortran routines, shared by the theta-l unit tests.
// These must match the interfaces defined in thetal_test_interface.F90
// (and hv_interface.F90 for init_hv_f90).
// Routines used by a single test are declared in that test's own source file.

extern "C" {

// Initialize the mesh, the connectivity and the f90 data structures (cubed sphere).
// Fills dvv and mp (the reference element derivative and mass matrix)
void init_f90 (const int& ne,
               const Homme::Real* hyai_ptr, const Homme::Real* hybi_ptr,
               const Homme::Real* hyam_ptr, const Homme::Real* hybm_ptr,
               Homme::Real* dvv, Homme::Real* mp,
               const Homme::Real& ps0);

// Same as init_f90, but also inits the hyperviscosity-related f90 data.
// HV's boundary exchange assumes every element has 4 real edge connections,
// so a real (cube-sphere) mesh/connectivity must be built on the F90 side
// (a hand-rolled, connection-less Connectivity segfaults in unpack()).
void init_hv_f90 (const int& ne,
                  const Homme::Real* hyai_ptr, const Homme::Real* hybi_ptr,
                  const Homme::Real* hyam_ptr, const Homme::Real* hybm_ptr,
                  Homme::Real* dvv, Homme::Real* mp,
                  const Homme::Real& ps0, const int& hypervis_subcycle,
                  const Homme::Real& nu, const Homme::Real& nu_div, const Homme::Real& nu_top,
                  const Homme::Real& nu_p, const Homme::Real& nu_s);

// Get pointers to the f90 geometry data (to init the C++ geometry from)
void init_geo_views_f90 (Homme::Real*& d_ptr, Homme::Real*& dinv_ptr,
                         const Homme::Real*& phis_ptr, const Homme::Real*& gradphis_ptr,
                         Homme::Real*& fcor_ptr,
                         Homme::Real*& sphmp_ptr, Homme::Real*& rspmp_ptr,
                         Homme::Real*& tVisc_ptr, Homme::Real*& sph2c_ptr,
                         Homme::Real*& metdet_ptr, Homme::Real*& metinv_ptr);

// Free the f90 data structures, so that the next TEST_CASE can call init_f90 again
void cleanup_f90 ();

} // extern "C"

#endif // HOMMEXX_THETAL_F90_INTERFACE_HPP
