/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ElementsGeometry.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/TestUtils.hpp"
#include "HybridVCoord.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include <limits>
#include <random>
#include <assert.h>

namespace Homme {

void ElementsGeometry::init(const int num_elems, const bool consthv, const bool alloc_gradphis,
                            const Real scale_factor, const Real laplacian_rigid_factor,
                            const bool alloc_sphere_coords) {
  // Sanity check
  assert (num_elems>0);

  m_num_elems = num_elems;
  m_consthv   = consthv;

  assert(scale_factor > 0);
  m_scale_factor = scale_factor;
  m_laplacian_rigid_factor = laplacian_rigid_factor < 0 ? 1/scale_factor : laplacian_rigid_factor;
  
  // Coriolis force
  m_fcor = ExecViewManaged<Real * [NP][NP]>("FCOR", m_num_elems);

  // Mass on the sphere
  m_spheremp  = ExecViewManaged<Real * [NP][NP]>("SPHEREMP",  m_num_elems);
  m_rspheremp = ExecViewManaged<Real * [NP][NP]>("RSPHEREMP", m_num_elems);

  // Metric
  m_metinv = ExecViewManaged<Real * [2][2][NP][NP]>("METINV", m_num_elems);
  m_metdet = ExecViewManaged<Real * [NP][NP]>("METDET", m_num_elems);

  if(!consthv){
    m_tensorvisc   = ExecViewManaged<Real * [2][2][NP][NP]>("TENSORVISC",   m_num_elems);
  }
  m_vec_sph2cart = ExecViewManaged<Real * [2][3][NP][NP]>("VEC_SPH2CART", m_num_elems);

  m_phis     = ExecViewManaged<Real *    [NP][NP]>("PHIS",          m_num_elems);

  //matrix D and its derivatives 
  m_d    = ExecViewManaged<Real * [2][2][NP][NP]>("matrix D",                   m_num_elems);
  m_dinv = ExecViewManaged<Real * [2][2][NP][NP]>("DInv - inverse of matrix D", m_num_elems);

  if (alloc_gradphis) {
    m_gradphis = decltype(m_gradphis) ("gradient of geopotential at surface", m_num_elems);
  }

  if (alloc_sphere_coords) {
    m_sphere_cart = ExecViewManaged<Real * [NP][NP][3]>("sphere_cart", m_num_elems);
    m_sphere_latlon = ExecViewManaged<Real * [NP][NP][2]>("sphere_latlon", m_num_elems);
  }
}

void ElementsGeometry::
set_elem_data (const int ie,
               CF90Ptr& D, CF90Ptr& Dinv, CF90Ptr& fcor,
               CF90Ptr& spheremp, CF90Ptr& rspheremp,
               CF90Ptr& metdet, CF90Ptr& metinv,
               CF90Ptr& tensorvisc, CF90Ptr& vec_sph2cart, const bool consthv,
               const Real* sphere_cart, const Real* sphere_latlon) {
  // Check geometry was inited
  assert (m_num_elems>0);

  // Check input
  assert (ie>=0 && ie<m_num_elems);

  using ScalarView   = ExecViewUnmanaged<Real [NP][NP]>;
  using TensorView   = ExecViewUnmanaged<Real [2][2][NP][NP]>;
  using Tensor23View = ExecViewUnmanaged<Real [2][3][NP][NP]>;

  using ScalarViewF90   = HostViewUnmanaged<const Real [NP][NP]>;
  using TensorViewF90   = HostViewUnmanaged<const Real [2][2][NP][NP]>;
  using Tensor23ViewF90 = HostViewUnmanaged<const Real [2][3][NP][NP]>;

  ScalarView::HostMirror h_fcor      = Kokkos::create_mirror_view(Homme::subview(m_fcor,ie));
  ScalarView::HostMirror h_metdet    = Kokkos::create_mirror_view(Homme::subview(m_metdet,ie));
  ScalarView::HostMirror h_spheremp  = Kokkos::create_mirror_view(Homme::subview(m_spheremp,ie));
  ScalarView::HostMirror h_rspheremp = Kokkos::create_mirror_view(Homme::subview(m_rspheremp,ie));
  TensorView::HostMirror h_metinv    = Kokkos::create_mirror_view(Homme::subview(m_metinv,ie));
  TensorView::HostMirror h_d         = Kokkos::create_mirror_view(Homme::subview(m_d,ie));
  TensorView::HostMirror h_dinv      = Kokkos::create_mirror_view(Homme::subview(m_dinv,ie));

  TensorView::HostMirror h_tensorvisc;
  Tensor23View::HostMirror h_vec_sph2cart;
  if( !consthv ){
    h_tensorvisc   = Kokkos::create_mirror_view(Homme::subview(m_tensorvisc,ie));
  }
  h_vec_sph2cart = Kokkos::create_mirror_view(Homme::subview(m_vec_sph2cart,ie));

  ScalarViewF90 h_fcor_f90         (fcor);
  ScalarViewF90 h_metdet_f90       (metdet);
  ScalarViewF90 h_spheremp_f90     (spheremp);
  ScalarViewF90 h_rspheremp_f90    (rspheremp);
  TensorViewF90 h_metinv_f90       (metinv);
  TensorViewF90 h_d_f90            (D);
  TensorViewF90 h_dinv_f90         (Dinv);
  TensorViewF90 h_tensorvisc_f90   (tensorvisc);
  Tensor23ViewF90 h_vec_sph2cart_f90 (vec_sph2cart);
  
  // 2d scalars
  for (int igp = 0; igp < NP; ++igp) {
    for (int jgp = 0; jgp < NP; ++jgp) {
      h_fcor      (igp, jgp) = h_fcor_f90      (igp,jgp);
      h_spheremp  (igp, jgp) = h_spheremp_f90  (igp,jgp);
      h_rspheremp (igp, jgp) = h_rspheremp_f90 (igp,jgp);
      h_metdet    (igp, jgp) = h_metdet_f90    (igp,jgp);
    }
  }

  // 2d tensors
  for (int idim = 0; idim < 2; ++idim) {
    for (int jdim = 0; jdim < 2; ++jdim) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          h_d      (idim,jdim,igp,jgp) = h_d_f90      (idim,jdim,igp,jgp);
          h_dinv   (idim,jdim,igp,jgp) = h_dinv_f90   (idim,jdim,igp,jgp);
          h_metinv (idim,jdim,igp,jgp) = h_metinv_f90 (idim,jdim,igp,jgp);
        }
      }
    }
  }
  
  if(!consthv) {
    for (int idim = 0; idim < 2; ++idim) {
      for (int jdim = 0; jdim < 2; ++jdim) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            h_tensorvisc   (idim,jdim,igp,jgp) = h_tensorvisc_f90   (idim,jdim,igp,jgp);
          }
        }
      }
    }
  }//end if consthv
  for (int idim = 0; idim < 2; ++idim) {
    for (int jdim = 0; jdim < 3; ++jdim) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          h_vec_sph2cart (idim,jdim,igp,jgp) = h_vec_sph2cart_f90 (idim,jdim,igp,jgp);
        }
      }
    }
  }

  Kokkos::deep_copy(Homme::subview(m_fcor,ie), h_fcor);
  Kokkos::deep_copy(Homme::subview(m_metinv,ie), h_metinv);
  Kokkos::deep_copy(Homme::subview(m_metdet,ie), h_metdet);
  Kokkos::deep_copy(Homme::subview(m_spheremp,ie), h_spheremp);
  Kokkos::deep_copy(Homme::subview(m_rspheremp,ie), h_rspheremp);
  Kokkos::deep_copy(Homme::subview(m_d,ie), h_d);
  Kokkos::deep_copy(Homme::subview(m_dinv,ie), h_dinv);
  if( !consthv ) {
    Kokkos::deep_copy(Homme::subview(m_tensorvisc,ie), h_tensorvisc);
  }
  Kokkos::deep_copy(Homme::subview(m_vec_sph2cart,ie), h_vec_sph2cart);

  if (sphere_cart && m_sphere_cart.size() != 0) {
    const auto fsc = HostViewUnmanaged<const Real [NP][NP][3]>(sphere_cart);
    Kokkos::deep_copy(Homme::subview(m_sphere_cart, ie), fsc);
  }
  if (sphere_latlon && m_sphere_latlon.size() != 0) {
    const auto fsl = HostViewUnmanaged<const Real [NP][NP][2]>(sphere_latlon);
    Kokkos::deep_copy(Homme::subview(m_sphere_latlon, ie), fsl);
  }
}

void ElementsGeometry::
set_phis (const int ie, CF90Ptr& phis) {
  // Check geometry was inited
  assert (m_num_elems>0);

  // Check input
  assert (ie>=0 && ie<m_num_elems);

  using ScalarView    = ExecViewUnmanaged<Real [NP][NP]>;
  using ScalarViewF90 = HostViewUnmanaged<const Real [NP][NP]>;

  ScalarViewF90           h_phis_f90 (phis);
  ScalarView::HostMirror  h_phis = Kokkos::create_mirror_view(Homme::subview(m_phis,ie));

  for (int igp = 0; igp < NP; ++igp) {
    for (int jgp = 0; jgp < NP; ++jgp) {
      h_phis (igp, jgp) = h_phis_f90 (igp,jgp);
    }
  }

  Kokkos::deep_copy(Homme::subview(m_phis,ie), h_phis);
}

void ElementsGeometry::randomize(const int seed) {
  // Check geometry was inited
  assert (m_num_elems>0);

  // Arbitrary minimum value to generate.
  constexpr const Real min_value = 0.1;

  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<Real> random_dist(min_value, 1.0 / min_value);

  // For the metric, to ensure we don't get an ill-conditioned metric
  std::uniform_real_distribution<Real> diag_pdf(0.5, 2.0);
  std::uniform_real_distribution<Real> offdiag_pdf(-0.1, 0.1);

  genRandArray(m_fcor,         engine, random_dist);

  genRandArray(m_spheremp,     engine, random_dist);
  genRandArray(m_tensorvisc,   engine, random_dist);
  genRandArray(m_vec_sph2cart, engine, random_dist);

  genRandArray(m_phis,         engine, random_dist);
  if (m_gradphis.size()!=0) {
    genRandArray(m_gradphis, engine, random_dist);
  }

  // 2d tensors
  // Generating lots of matrices with reasonable determinants can be difficult
  // So instead of generating them all at once and verifying they're correct,
  // generate them one at a time, verifying them individually
  HostViewManaged<Real[2][2]> h_matrix("single host metric matrix");

  auto h_d    = Kokkos::create_mirror_view(m_d);
  auto h_dinv = Kokkos::create_mirror_view(m_dinv);
  auto h_metinv = Kokkos::create_mirror_view(m_metinv);
  auto h_metdet = Kokkos::create_mirror_view(m_metdet);
  auto h_rspheremp = Kokkos::create_mirror_view(m_rspheremp);
  auto h_spheremp  = Kokkos::create_mirror_view(m_spheremp);
  Kokkos::deep_copy(h_spheremp, m_spheremp);

  for (int ie = 0; ie < m_num_elems; ++ie) {
    // Because this constraint is difficult to satisfy for all of the tensors,
    // incrementally generate the view
    for (int ip = 0; ip < NP; ++ip) {
      for (int jp = 0; jp < NP; ++jp) {

        // To avoid physical inconsistencies (which may trigger errors in the EOS):
        // rspheremp = 1/spheremp (locally. For the dss-ed version, use the overload
        // that accepts connectivity)
        h_rspheremp(ie,ip,jp) = 1./h_spheremp(ie,ip,jp);

        const Real a = diag_pdf(engine), d = diag_pdf(engine);
        const Real b = offdiag_pdf(engine), c = offdiag_pdf(engine);
        const Real det = a*d - b*c; // >= 0.5*0.5 - 0.1*0.1 = 0.24, safely away from 0

        h_d(ie,0,0,ip,jp) = a; h_d(ie,0,1,ip,jp) = b;
        h_d(ie,1,0,ip,jp) = c; h_d(ie,1,1,ip,jp) = d;
        h_dinv(ie,0,0,ip,jp) =  d/det; h_dinv(ie,0,1,ip,jp) = -b/det;
        h_dinv(ie,1,0,ip,jp) = -c/det; h_dinv(ie,1,1,ip,jp) =  a/det;

        const Real a2 = diag_pdf(engine), d2 = diag_pdf(engine);
        const Real b2 = offdiag_pdf(engine), c2 = offdiag_pdf(engine);
        h_metdet(ie,ip,jp) = a2*d2 - b2*c2;
        h_metinv(ie,0,0,ip,jp) = a2; h_metinv(ie,0,1,ip,jp) = b2;
        h_metinv(ie,1,0,ip,jp) = c2; h_metinv(ie,1,1,ip,jp) = d2;
      }
    }
  }

  Kokkos::deep_copy(m_d,    h_d);
  Kokkos::deep_copy(m_dinv, h_dinv);
  Kokkos::deep_copy(m_metinv, h_metinv);
  Kokkos::deep_copy(m_metdet, h_metdet);
  Kokkos::deep_copy(m_rspheremp, h_rspheremp);
}

void ElementsGeometry::randomize (const int seed, const std::shared_ptr<Connectivity>& connectivity,
                                   const std::shared_ptr<MpiBuffersManager>& buffers_manager) {
  randomize(seed);

  // Make rspheremp equal to 1/BE(spheremp) (needed to ensure adjoint works):
  // spheremp itself stays local (it is the per-element weak-form/quadrature
  // weight), but rspheremp -- used by BoundaryExchange's weighted
  // exchange(rspheremp) to divide by the *assembled* mass after DSS-summing
  // local (spheremp-weighted) contributions -- must be the same value at
  // every owning element of a shared DOF for that exchange to be self-adjoint.
  // randomize(seed) alone sets rspheremp = 1/spheremp locally, which does not
  // satisfy this.
  Kokkos::deep_copy(m_rspheremp, m_spheremp);
  BoundaryExchangeST<Real> be_mass(connectivity, buffers_manager);
  be_mass.set_num_fields(0,1,0);
  be_mass.register_field(m_rspheremp);
  be_mass.registration_completed();
  be_mass.exchange();
  const auto rspheremp = m_rspheremp;
  const Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>> p3({0,0,0},{m_num_elems,NP,NP});
  Kokkos::parallel_for(p3, KOKKOS_LAMBDA(const int ie, const int ip, const int jp) {
    rspheremp(ie,ip,jp) = 1 / rspheremp(ie,ip,jp);
  });
}

} // namespace Homme
