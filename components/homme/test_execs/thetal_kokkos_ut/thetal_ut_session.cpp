#include "thetal_ut_session.hpp"
#include "thetal_f90_interface.hpp"

#include "Context.hpp"
#include "mpi/Connectivity.hpp"

namespace Homme {

ThetalUnitTestSession::
ThetalUnitTestSession (const int ne_in, const unsigned int seed_in,
                       const bool is_sphere, const VCoord vcoord_type)
 : ne (ne_in)
 , seed (seed_in)
 , hvcoord (Context::singleton().create<HybridVCoord>())
 , ref_FE  (Context::singleton().create<ReferenceElement>())
{
  // Init hvcoord, and copy it to the host-side members
  if (vcoord_type==VCoord::Smooth) {
    hvcoord.smooth_init();
  } else {
    hvcoord.random_init(seed);
  }
  update_host_hvcoord();

  // This also creates the Connectivity in the Context
  std::vector<Real> dvv(NP*NP), mp(NP*NP);
  if (is_sphere) {
    init_f90(ne, hyai.data(), hybi.data(), hyam.data(), hybm.data(),
             dvv.data(), mp.data(), ps0);
  } else {
    init_planar_f90(ne+1, ne, hyai.data(), hybi.data(), hyam.data(), hybm.data(),
                    dvv.data(), mp.data(), ps0);
  }

  ref_FE.init_mass(mp.data());
  ref_FE.init_deriv(dvv.data());
}

ThetalUnitTestSession::~ThetalUnitTestSession ()
{
  cleanup_f90();
  Context::finalize_singleton();
}

int ThetalUnitTestSession::num_elems () const
{
  return Context::singleton().get<Connectivity>().get_num_local_elements();
}

void ThetalUnitTestSession::update_host_hvcoord ()
{
  ps0 = hvcoord.ps0;

  auto h_ai = Kokkos::create_mirror(hvcoord.hybrid_ai);
  auto h_bi = Kokkos::create_mirror(hvcoord.hybrid_bi);
  auto h_am = Kokkos::create_mirror(hvcoord.hybrid_am);
  auto h_bm = Kokkos::create_mirror(hvcoord.hybrid_bm);
  Kokkos::deep_copy(h_ai,hvcoord.hybrid_ai);
  Kokkos::deep_copy(h_bi,hvcoord.hybrid_bi);
  Kokkos::deep_copy(h_am,hvcoord.hybrid_am);
  Kokkos::deep_copy(h_bm,hvcoord.hybrid_bm);

  hyai.resize(NUM_INTERFACE_LEV);
  hybi.resize(NUM_INTERFACE_LEV);
  for (int i=0; i<NUM_INTERFACE_LEV; ++i) {
    hyai[i] = h_ai(i);
    hybi[i] = h_bi(i);
  }

  hyam.resize(NUM_PHYSICAL_LEV);
  hybm.resize(NUM_PHYSICAL_LEV);
  for (int i=0; i<NUM_PHYSICAL_LEV; ++i) {
    const int ilev = i / VECTOR_SIZE;
    const int ivec = i % VECTOR_SIZE;
    hyam[i] = h_am(ilev)[ivec];
    hybm[i] = h_bm(ilev)[ivec];
  }
}

void ThetalUnitTestSession::
init_geometry (ElementsGeometry& geo, const bool zero_phis) const
{
  auto d        = Kokkos::create_mirror(geo.m_d);
  auto dinv     = Kokkos::create_mirror(geo.m_dinv);
  auto phis     = Kokkos::create_mirror(geo.m_phis);
  auto gradphis = Kokkos::create_mirror(geo.m_gradphis);
  auto fcor     = Kokkos::create_mirror(geo.m_fcor);
  auto spmp     = Kokkos::create_mirror(geo.m_spheremp);
  auto rspmp    = Kokkos::create_mirror(geo.m_rspheremp);
  auto tVisc    = Kokkos::create_mirror(geo.m_tensorvisc);
  auto sph2c    = Kokkos::create_mirror(geo.m_vec_sph2cart);
  auto mdet     = Kokkos::create_mirror(geo.m_metdet);
  auto minv     = Kokkos::create_mirror(geo.m_metinv);

  if (zero_phis) {
    Kokkos::deep_copy(phis,    Real(0));
    Kokkos::deep_copy(gradphis,Real(0));
  } else {
    Kokkos::deep_copy(phis,    geo.m_phis);
    Kokkos::deep_copy(gradphis,geo.m_gradphis);
  }

  Real*       d_ptr        = d.data();
  Real*       dinv_ptr     = dinv.data();
  const Real* phis_ptr     = phis.data();
  const Real* gradphis_ptr = gradphis.data();
  Real*       fcor_ptr     = fcor.data();
  Real*       spmp_ptr     = spmp.data();
  Real*       rspmp_ptr    = rspmp.data();
  Real*       tVisc_ptr    = tVisc.data();
  Real*       sph2c_ptr    = sph2c.data();
  Real*       mdet_ptr     = mdet.data();
  Real*       minv_ptr     = minv.data();

  init_geo_views_f90(d_ptr, dinv_ptr, phis_ptr, gradphis_ptr, fcor_ptr,
                     spmp_ptr, rspmp_ptr, tVisc_ptr,
                     sph2c_ptr, mdet_ptr, minv_ptr);

  Kokkos::deep_copy(geo.m_d,           d);
  Kokkos::deep_copy(geo.m_dinv,        dinv);
  Kokkos::deep_copy(geo.m_spheremp,    spmp);
  Kokkos::deep_copy(geo.m_rspheremp,   rspmp);
  Kokkos::deep_copy(geo.m_tensorvisc,  tVisc);
  Kokkos::deep_copy(geo.m_vec_sph2cart,sph2c);
  Kokkos::deep_copy(geo.m_metdet,      mdet);
  Kokkos::deep_copy(geo.m_metinv,      minv);
  Kokkos::deep_copy(geo.m_fcor,        fcor);
  Kokkos::deep_copy(geo.m_phis,        phis);
  Kokkos::deep_copy(geo.m_gradphis,    gradphis);
}

} // namespace Homme
