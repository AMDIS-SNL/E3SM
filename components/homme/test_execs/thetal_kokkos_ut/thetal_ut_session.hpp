#ifndef HOMMEXX_THETAL_UT_SESSION_HPP
#define HOMMEXX_THETAL_UT_SESSION_HPP

#include "Types.hpp"
#include "ElementsGeometry.hpp"
#include "HybridVCoord.hpp"
#include "ReferenceElement.hpp"

#include <functional>
#include <vector>

namespace Homme {

// Common setup/cleanup of the unit tests that rely on the F90 side to create
// a cubed-sphere mesh (grid, decomposition, connectivity, geometry).
//
// The constructor
//  - creates a HybridVCoord and a ReferenceElement in the Context singleton,
//  - randomly inits the HybridVCoord (see random_init_hvcoord),
//  - calls the F90 init routine (by default, init_f90), which also creates
//    the Connectivity in the Context,
//  - inits the ReferenceElement mass and derivative matrices from the F90 ones.
// The destructor frees the F90 data structures and finalizes the Context
// singleton. Since it runs even if a CHECK/REQUIRE throws, the next TEST_CASE
// can safely start from scratch.
//
// Other objects (ekat::Comm, SimulationParams, ...) are NOT created by the session;
// the caller is responsible for creating them in the Context, as needed.
// Since the Context is finalized in the destructor, the session must outlive
// all the objects obtained from the Context. In particular, declare the session
// right after getting the Context singleton.
class CubeSphereTestSession {
public:
  // Signature of a custom F90 init routine. It is called with the session,
  // so that it can use its (already initialized) data members as inputs
  // (hyai, hybi, hyam, hybm, ps0) and outputs (dvv, mp).
  using InitFn = std::function<void(CubeSphereTestSession&)>;

  // Default F90 init and cleanup: init_f90/cleanup_f90
  CubeSphereTestSession (const int ne, const unsigned int seed);

  // Custom F90 init and cleanup (e.g., init_dirk_f90/cleanup_f90)
  CubeSphereTestSession (const int ne, const unsigned int seed,
                         const InitFn& f90_init,
                         const std::function<void()>& f90_cleanup);

  ~CubeSphereTestSession ();

  CubeSphereTestSession (const CubeSphereTestSession&) = delete;
  CubeSphereTestSession& operator= (const CubeSphereTestSession&) = delete;

  // Number of elements on this rank (the Connectivity must have been created by the F90 init)
  int num_elems () const;

  // Pull from F90 the geometry of the mesh (d, dinv, spheremp, rspheremp, metdet,
  // metinv, tensorvisc, vec_sph2cart, fcor) and store it in geo (which must already
  // be init-ed with the proper number of elements).
  // Note: the F90 side also reads phis and gradphis (they're inputs). If zero_phis=true,
  //       they are set to 0 both in F90 and in geo; otherwise, the values stored
  //       in geo are passed to F90.
  void init_geometry (ElementsGeometry& geo, const bool zero_phis = true) const;

  // Mesh resolution (number of elements per cube edge)
  const int ne;

  // Random seed used to init the hvcoord
  const unsigned int seed;

  // Inputs/outputs of the F90 init routines. Note: hyam/hybm are the
  // "unpacked" mid-point coefficients (NUM_PHYSICAL_LEV entries)
  Real ps0;
  std::vector<Real> hyai, hybi, hyam, hybm;
  std::vector<Real> dvv, mp;

  HybridVCoord&     hvcoord;
  ReferenceElement& ref_FE;

private:
  // Randomly inits hvcoord, and copies hyai, hybi, hyam, hybm, ps0 in the host-side members
  void random_init_hvcoord ();

  std::function<void()> m_f90_cleanup;
};

} // namespace Homme

#endif // HOMMEXX_THETAL_UT_SESSION_HPP
