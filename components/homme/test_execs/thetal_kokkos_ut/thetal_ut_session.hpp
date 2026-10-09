#ifndef HOMMEXX_THETAL_UT_SESSION_HPP
#define HOMMEXX_THETAL_UT_SESSION_HPP

#include "Types.hpp"
#include "ElementsGeometry.hpp"
#include "HybridVCoord.hpp"
#include "ReferenceElement.hpp"

#include <vector>

namespace Homme {

// Size and offset (in meters) of a planar domain: [sx,sx+lx] x [sy,sy+ly]
struct PlanarDomain {
  Real lx = 10000;
  Real ly = 5000;
  Real sx = -5000;
  Real sy = 2500;
};

// Common setup/cleanup of the unit tests that rely on the F90 side to create
// a mesh (grid, decomposition, connectivity, geometry), either a cubed sphere or a planar one.
//
// The constructor
//  - creates a HybridVCoord and a ReferenceElement in the Context singleton,
//  - inits the HybridVCoord (randomly, by default) and copies it to the host-side members,
//  - calls the F90 init routine (init_f90 or init_planar_f90), which also creates
//    the Connectivity in the Context,
//  - inits the ReferenceElement mass and derivative matrices from the F90 ones.
// The destructor frees the F90 data structures (cleanup_f90) and finalizes the
// Context singleton. Since it runs even if a CHECK/REQUIRE throws, the next
// TEST_CASE can safely start from scratch.
//
// Test-specific F90 initialization (e.g., edge buffers) is NOT done by the session:
// tests should call their own F90 init routine right after creating the session.
// Other objects (ekat::Comm, SimulationParams, ...) are NOT created by the session
// either; the caller is responsible for creating them in the Context, as needed.
// Since the Context is finalized in the destructor, the session must outlive
// all the objects obtained from the Context.
class ThetalUnitTestSession {
public:
  // How to init the hybrid vertical coordinate
  enum class VCoord {
    Random,  // HybridVCoord::random_init(seed): random, monotone, but layers may vary a lot in thickness
    Smooth   // HybridVCoord::smooth_init(): deterministic, uniform in eta, realistic model top
  };

  // Cubed sphere mesh, with ne x ne elements per cube face
  ThetalUnitTestSession (const int ne, const unsigned int seed,
                         const VCoord vcoord_type = VCoord::Random);

  // Planar mesh, with nex x ney elements
  ThetalUnitTestSession (const int nex, const int ney, const unsigned int seed,
                         const VCoord vcoord_type = VCoord::Random,
                         const PlanarDomain& domain = PlanarDomain());

  ~ThetalUnitTestSession ();

  ThetalUnitTestSession (const ThetalUnitTestSession&) = delete;
  ThetalUnitTestSession& operator= (const ThetalUnitTestSession&) = delete;

  // Number of elements on this rank
  int num_elems () const;

  // Pull from F90 the geometry of the mesh (d, dinv, spheremp, rspheremp, metdet,
  // metinv, tensorvisc, vec_sph2cart, fcor) and store it in geo (which must already
  // be init-ed with the proper number of elements).
  // Note: the F90 side also reads phis and gradphis (they're inputs). If zero_phis=true,
  //       they are set to 0 both in F90 and in geo; otherwise, the values stored
  //       in geo are passed to F90.
  void init_geometry (ElementsGeometry& geo, const bool zero_phis = true) const;

  // Mesh resolution: number of elements in each direction (for a cubed sphere,
  // along the edge of each cube face; so nex=ney=ne)
  const int nex;
  const int ney;
  const bool is_planar;

  // Random seed used to init the hvcoord (not used if vcoord_type=VCoord::Smooth)
  const unsigned int seed;

  // Host-side copy of hvcoord. Note: hyam/hybm are the "unpacked"
  // mid-point coefficients (NUM_PHYSICAL_LEV entries)
  Real ps0;
  std::vector<Real> hyai, hybi, hyam, hybm;

  HybridVCoord&     hvcoord;
  ReferenceElement& ref_FE;

private:
  enum class Mesh { CubedSphere, Planar };

  // The actual constructor, which all the public ones forward to
  ThetalUnitTestSession (const Mesh mesh, const int nex, const int ney, const unsigned int seed,
                         const VCoord vcoord_type, const PlanarDomain& domain);

  void update_host_hvcoord ();
};

} // namespace Homme

#endif // HOMMEXX_THETAL_UT_SESSION_HPP
