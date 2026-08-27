#include <catch2/catch.hpp>

#include "Types.hpp"
#include "Tape.hpp"
#include "Context.hpp"
#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"
#include "DirkFunctor.hpp"
#include "DirkFunctorImpl.hpp"
#include "LimiterFunctor.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "PhysicalConstants.hpp"
#include "TimeLevel.hpp"
#include "prim_advance_exp.hpp"
#include "prim_advance_adj.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_string_utils.hpp>
#include <ekat_comm.hpp>

#include <iomanip>

namespace Homme
{

extern "C" {
// Even if we don't run the f90 code in this unit test, it is easier to
// init from f90, which takes care of creating the grid and decomposing it
void init_f90 (const int& ne,
               const Real* hyai_ptr, const Real* hybi_ptr,
               const Real* hyam_ptr, const Real* hybm_ptr,
               Real* dvv, Real* mp,
               const Real& ps0);
void init_geo_views_f90 (Real*& d_ptr, Real*& dinv_ptr,
               const Real*& phis_ptr, const Real*& gradphis_ptr,
               Real*& fcor_ptr,
               Real*& sphmp_ptr, Real*& rspmp_ptr,
               Real*& tVisc_ptr, Real*& sph2c_ptr,
               Real*& metdet_ptr, Real*& metinv_ptr);
void cleanup_f90();
}

template<typename ViewT>
double dot (ViewT v1, ViewT v2, int last_dim = -1) {
  auto v1h = Kokkos::create_mirror_view(v1);
  Kokkos::deep_copy(v1h,v1);
  auto v2h = Kokkos::create_mirror_view(v2);
  Kokkos::deep_copy(v2h,v2);

  EKAT_REQUIRE_MSG (v1.rank==v2.rank, "Error! Views have different rank.\n");
  EKAT_REQUIRE_MSG (v1.rank==3 or v1.rank==4 or v1.rank==5, "Error! Unsupported rank.\n");
  double prod = 0;
  if constexpr(v1.rank==3) {
    auto last_ext = last_dim > 0 ? last_dim : v1.extent_int(2);
    for (int i=0; i<v1.extent_int(0); ++i)
      for (int j=0; j<v1.extent_int(1); ++j)
        for (int k=0; k<last_ext; ++k)
          prod += v1h(i,j,k)*v2h(i,j,k);
  } else if constexpr(v1.rank==4) {
    auto last_ext = last_dim > 0 ? last_dim : v1.extent_int(3);
    for (int i=0; i<v1.extent_int(0); ++i)
      for (int j=0; j<v1.extent_int(1); ++j)
        for (int k=0; k<v1.extent_int(2); ++k)
          for (int l=0; l<last_ext; ++l)
            prod += v1h(i,j,k,l)*v2h(i,j,k,l);
  } else if constexpr(v1.rank==5) {
    auto last_ext = last_dim > 0 ? last_dim : v1.extent_int(4);
    for (int i=0; i<v1.extent_int(0); ++i)
      for (int j=0; j<v1.extent_int(1); ++j)
        for (int k=0; k<v1.extent_int(2); ++k)
          for (int l=0; l<v1.extent_int(3); ++l)
            for (int m=0; m<last_ext; ++m)
              prod += v1h(i,j,k,l,m)*v2h(i,j,k,l,m);
  }

  return prod;
}

void init_geo_views (ElementsGeometry& geo)
{
  auto d        = Kokkos::create_mirror_view(geo.m_d);
  auto dinv     = Kokkos::create_mirror_view(geo.m_dinv);
  auto phis     = Kokkos::create_mirror_view(geo.m_phis);
  auto gradphis = Kokkos::create_mirror_view(geo.m_gradphis);
  auto fcor     = Kokkos::create_mirror_view(geo.m_fcor);
  auto spmp     = Kokkos::create_mirror_view(geo.m_spheremp);
  auto rspmp    = Kokkos::create_mirror_view(geo.m_rspheremp);
  auto tVisc    = Kokkos::create_mirror_view(geo.m_tensorvisc);
  auto sph2c    = Kokkos::create_mirror_view(geo.m_vec_sph2cart);
  auto mdet     = Kokkos::create_mirror_view(geo.m_metdet);
  auto minv     = Kokkos::create_mirror_view(geo.m_metinv);

  // Aquaplanet: zero phis/gradphis before passing to f90
  Kokkos::deep_copy(phis,    Real(0));
  Kokkos::deep_copy(gradphis,Real(0));

  Real*        d_ptr        = d.data();
  Real*        dinv_ptr     = dinv.data();
  const Real*  phis_ptr     = phis.data();
  const Real*  gradphis_ptr = gradphis.data();
  Real*        fcor_ptr     = fcor.data();
  Real*        spmp_ptr     = spmp.data();
  Real*        rspmp_ptr    = rspmp.data();
  Real*        tVisc_ptr    = tVisc.data();
  Real*        sph2c_ptr    = sph2c.data();
  Real*        mdet_ptr     = mdet.data();
  Real*        minv_ptr     = minv.data();

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

TEST_CASE("ttype10_imex_adjoint")
{
  constexpr int ne = 2;
  constexpr int nlevs = NUM_PHYSICAL_LEV;
  const auto A = Kokkos::ALL();

  // The random numbers generator
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  // Use stuff from Context, to increase similarity with actual runs
  auto& c = Context::singleton();
  auto& comm = c.create<ekat::Comm>(MPI_COMM_WORLD);
  auto& tl = c.create<TimeLevel>();

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.dp3d_thresh = 0; // don't let the limiter do anything, for now
  params.vtheta_thresh = 0; // don't let the limiter do anything, for now
  params.params_set = true;
  params.qsplit = 1;
  params.rsplit = 1;
  params.store_fwd_state = true;
  params.theta_hydrostatic_mode = true;
  params.scale_factor = PhysicalConstants::rearth0;
  params.laplacian_rigid_factor = PhysicalConstants::rrearth0;

  // Create and init hvcoord and ref_elem, needed to init the fortran interface
  auto& hvcoord = c.create<HybridVCoord>();
  auto& ref_FE  = c.create<ReferenceElement>();
  hvcoord.random_init(seed);

  auto hyai = Kokkos::create_mirror_view(hvcoord.hybrid_ai);
  auto hybi = Kokkos::create_mirror_view(hvcoord.hybrid_bi);
  auto hyam = Kokkos::create_mirror_view(hvcoord.hybrid_am);
  auto hybm = Kokkos::create_mirror_view(hvcoord.hybrid_bm);
  Kokkos::deep_copy(hyai,hvcoord.hybrid_ai);
  Kokkos::deep_copy(hybi,hvcoord.hybrid_bi);
  Kokkos::deep_copy(hyam,hvcoord.hybrid_am);
  Kokkos::deep_copy(hybm,hvcoord.hybrid_bm);
  HostViewManaged<Real[NUM_PHYSICAL_LEV]> hyam_r(""),hybm_r("");
  for (int i=0;i<NUM_PHYSICAL_LEV;++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyam_r(i) = ADValue(hyam(ilev)[ivec]);
    hybm_r(i) = ADValue(hybm(ilev)[ivec]);
  }

  std::vector<Real> dvv(NP*NP);
  std::vector<Real> mp(NP*NP);
  init_f90(ne,hyai.data(),hybi.data(),hyam_r.data(),hybm_r.data(),dvv.data(),mp.data(),hvcoord.ps0);

  ref_FE.init_mass(mp.data());
  ref_FE.init_deriv(dvv.data());

  const int num_elems = c.get<Connectivity>().get_num_local_elements();
  const auto max_pressure = 1000.0 + hvcoord.ps0; // This ensures max_p > ps0

  // Init geometry views once (same for all elements structs)
  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);

  init_geo_views(geo);

  // Create elements for FWD/BWD integration
  auto& elems_dp = c.create<ElementsST<DpFadType>>();
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dp.m_geometry = geo;

  auto& elems_dx_caar = c.create<ElementsST<DxFadTypeCaar>>();
  elems_dx_caar.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx_caar.m_geometry = geo; // Use same views for geometry

  auto& elems_dx_dirk = c.create<ElementsST<DxFadTypeDirk>>();
  elems_dx_dirk.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx_dirk.m_geometry = geo; // Use same views for geometry

  // Create auxiliary structures
  auto& bmm = c.create<MpiBuffersManagerMap>();
  bmm[MPI_EXCHANGE]->set_connectivity(c.get_ptr<Connectivity>());

  auto& sphop_dp = c.create<SphereOperatorsST<DpFadType>>();
  auto& sphop_dx_caar = c.create<SphereOperatorsST<DxFadTypeCaar>>();

  sphop_dp.setup(geo,ref_FE);
  sphop_dx_caar.setup(geo,ref_FE);

  auto conn  = c.get_ptr<Connectivity>();
  auto  bm   = bmm[MPI_EXCHANGE];

  int nm1 = tl.nm1 = 0;
  int n0  = tl.n0  = 1;
  int np1 = tl.np1 = 2;

  // Create and init Caar/Limiter/Dirk functors
  auto& caar_dp = c.create<CaarFunctorST<DpFadType>>();
  auto& caar_dx = c.create<CaarFunctorST<DxFadTypeCaar>>();
  caar_dp.setup(elems_dp,ref_FE,hvcoord,sphop_dp);
  caar_dx.setup(elems_dx_caar,ref_FE,hvcoord,sphop_dx_caar);
  caar_dp.init_boundary_exchanges(bmm[MPI_EXCHANGE]);
  caar_dx.init_boundary_exchanges(bmm[MPI_EXCHANGE]);
  std::any_cast<CaarFunctorImplST<DpFadType>&>(caar_dp.impl()).m_run_limiter = false;
  std::any_cast<CaarFunctorImplST<DxFadTypeCaar>&>(caar_dx.impl()).m_run_limiter = false;

  auto& limiter_dp = c.create<LimiterFunctorST<DpFadType>>(elems_dp,hvcoord,params);
  auto& limiter_dx = c.create<LimiterFunctorST<DxFadTypeCaar>>(elems_dx_caar,hvcoord,params);
  limiter_dp.m_verbose = false;
  limiter_dx.m_verbose = false;

  auto& dirk_dp = c.create<DirkFunctorST<DpFadType>>(elems_dp.num_elems());
  auto& dirk_dx = c.create<DirkFunctorST<DxFadTypeDirk>>(elems_dx_dirk.num_elems());

  // Setup scratch buffers
  FunctorsBuffersManager fbm;
  fbm.request_size(caar_dp.requested_buffer_size());
  fbm.request_size(caar_dx.requested_buffer_size());
  fbm.request_size(dirk_dp.requested_buffer_size());
  fbm.request_size(dirk_dx.requested_buffer_size());
  fbm.request_size(limiter_dp.requested_buffer_size());
  fbm.request_size(limiter_dx.requested_buffer_size());

  fbm.allocate();

  caar_dp.init_buffers(fbm);
  caar_dx.init_buffers(fbm);
  dirk_dp.init_buffers(fbm);
  dirk_dx.init_buffers(fbm);
  limiter_dp.init_buffers(fbm);
  limiter_dx.init_buffers(fbm);

  // Scalar params
  double dt = 10;
  double eta_ave_w = 0.25;

  // Create the Imex tape
  c.any_map().try_emplace("imex_tape",std::in_place_type<Tape<StateSnapshot>>,11,num_elems);

  // Initial state
  StateSnapshot state_t0(num_elems);
  state_t0.randomize(seed,1e5,1e3,hvcoord.hybrid_ai0,geo.m_phis);

  // Run FWD sweep
  elems_dp.m_state.import_snapshot(state_t0,n0);
  elems_dp.m_state.randomize_derivs(seed,n0);
  auto du0 = elems_dp.m_state.take_deriv_snapshot(n0,0);
  printf(" -> Run forward problem...\n");
  ttype10_imex_timestep<DpFadType>(tl,dt,eta_ave_w);
  printf(" -> Run forward problem...done!\n");
  auto duN = elems_dp.m_state.take_deriv_snapshot(np1,0);

  // Run BWD pass
  StateSnapshot lambda(num_elems);
  lambda.randomize(seed,1.0,1.0/100,0.0);
  auto lambdaN = lambda.clone(true);
  printf(" -> Run adjoint problem...\n");
  ttype10_imex_adjoint(dt,eta_ave_w,lambda);
  printf(" -> Run adjoint problem...done!\n");
  auto lambda0 = lambda.clone(true);

  // Compare du0*lambda0 with duN*lambdaN
  auto v_dot0 = dot(ekat::scalarize(du0.v),ekat::scalarize(lambda0.v),nlevs);
  auto v_dotN = dot(ekat::scalarize(duN.v),ekat::scalarize(lambdaN.v),nlevs);

  auto vth_dot0 = dot(ekat::scalarize(du0.vtheta_dp),ekat::scalarize(lambda0.vtheta_dp),nlevs);
  auto vth_dotN = dot(ekat::scalarize(duN.vtheta_dp),ekat::scalarize(lambdaN.vtheta_dp),nlevs);

  auto dp_dot0 = dot(ekat::scalarize(du0.dp3d),ekat::scalarize(lambda0.dp3d),nlevs);
  auto dp_dotN = dot(ekat::scalarize(duN.dp3d),ekat::scalarize(lambdaN.dp3d),nlevs);

  auto w_dot0 = dot(ekat::scalarize(du0.w_i),ekat::scalarize(lambda0.w_i),nlevs+1);
  auto w_dotN = dot(ekat::scalarize(duN.w_i),ekat::scalarize(lambdaN.w_i),nlevs+1);

  auto phi_dot0 = dot(ekat::scalarize(du0.phinh_i),ekat::scalarize(lambda0.phinh_i),nlevs+1);
  auto phi_dotN = dot(ekat::scalarize(duN.phinh_i),ekat::scalarize(lambdaN.phinh_i),nlevs+1);

  constexpr auto tol = std::numeric_limits<double>::epsilon()*1e4;
  {
    using namespace Catch::Matchers;

    auto full_dot0 = v_dot0 + vth_dot0 + dp_dot0 + w_dot0 + phi_dot0;
    auto full_dotN = v_dotN + vth_dotN + dp_dotN + w_dotN + phi_dotN;
    CHECK_THAT (full_dot0, WithinRel(full_dotN,tol));

    if (comm.am_i_root())
      std::cout << std::setprecision(15)
                << "   <du0, lambda0> = " << full_dot0
                << ",  <duN, lambdaN> = " << full_dotN << "\n";
  }

  cleanup_f90();
  c.finalize_singleton();
}

TEST_CASE("ttype5_imex_adjoint")
{
  constexpr int ne = 2;
  constexpr int nlevs = NUM_PHYSICAL_LEV;
  const auto A = Kokkos::ALL();

  // The random numbers generator
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  // Use stuff from Context, to increase similarity with actual runs
  auto& c = Context::singleton();
  auto& comm = c.create<ekat::Comm>(MPI_COMM_WORLD);
  auto& tl = c.create<TimeLevel>();

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.dp3d_thresh = 0; // don't let the limiter do anything, for now
  params.vtheta_thresh = 0; // don't let the limiter do anything, for now
  params.params_set = true;
  params.qsplit = 1;
  params.rsplit = 1;
  params.store_fwd_state = true;
  params.theta_hydrostatic_mode = false;
  params.scale_factor = PhysicalConstants::rearth0;
  params.laplacian_rigid_factor = PhysicalConstants::rrearth0;

  // Create and init hvcoord and ref_elem, needed to init the fortran interface
  auto& hvcoord = c.create<HybridVCoord>();
  auto& ref_FE  = c.create<ReferenceElement>();
  hvcoord.random_init(seed);

  auto hyai = Kokkos::create_mirror_view(hvcoord.hybrid_ai);
  auto hybi = Kokkos::create_mirror_view(hvcoord.hybrid_bi);
  auto hyam = Kokkos::create_mirror_view(hvcoord.hybrid_am);
  auto hybm = Kokkos::create_mirror_view(hvcoord.hybrid_bm);
  Kokkos::deep_copy(hyai,hvcoord.hybrid_ai);
  Kokkos::deep_copy(hybi,hvcoord.hybrid_bi);
  Kokkos::deep_copy(hyam,hvcoord.hybrid_am);
  Kokkos::deep_copy(hybm,hvcoord.hybrid_bm);
  HostViewManaged<Real[NUM_PHYSICAL_LEV]> hyam_r(""),hybm_r("");
  for (int i=0;i<NUM_PHYSICAL_LEV;++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyam_r(i) = ADValue(hyam(ilev)[ivec]);
    hybm_r(i) = ADValue(hybm(ilev)[ivec]);
  }

  std::vector<Real> dvv(NP*NP);
  std::vector<Real> mp(NP*NP);
  init_f90(ne,hyai.data(),hybi.data(),hyam_r.data(),hybm_r.data(),dvv.data(),mp.data(),hvcoord.ps0);

  ref_FE.init_mass(mp.data());
  ref_FE.init_deriv(dvv.data());

  const int num_elems = c.get<Connectivity>().get_num_local_elements();
  const auto max_pressure = 1000.0 + hvcoord.ps0; // This ensures max_p > ps0

  // Init geometry views once (same for all elements structs)
  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  
  init_geo_views(geo);

  // Create elements for FWD/BWD integration
  auto& elems_dp = c.create<ElementsST<DpFadType>>();
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dp.m_geometry = geo;

  auto& elems_dx_caar = c.create<ElementsST<DxFadTypeCaar>>();
  elems_dx_caar.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx_caar.m_geometry = geo; // Use same views for geometry

  auto& elems_dx_dirk = c.create<ElementsST<DxFadTypeDirk>>();
  elems_dx_dirk.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx_dirk.m_geometry = geo; // Use same views for geometry

  // Create auxiliary structures
  auto& bmm = c.create<MpiBuffersManagerMap>();
  bmm[MPI_EXCHANGE]->set_connectivity(c.get_ptr<Connectivity>());

  auto& sphop_dp = c.create<SphereOperatorsST<DpFadType>>();
  auto& sphop_dx_caar = c.create<SphereOperatorsST<DxFadTypeCaar>>();

  sphop_dp.setup(geo,ref_FE);
  sphop_dx_caar.setup(geo,ref_FE);

  auto conn  = c.get_ptr<Connectivity>();
  auto  bm   = bmm[MPI_EXCHANGE];

  int nm1 = tl.nm1 = 0;
  int n0  = tl.n0  = 1;
  int np1 = tl.np1 = 2;

  // Create and init Caar/Limiter/Dirk functors
  auto& caar_dp = c.create<CaarFunctorST<DpFadType>>();
  auto& caar_dx = c.create<CaarFunctorST<DxFadTypeCaar>>();
  caar_dp.setup(elems_dp,ref_FE,hvcoord,sphop_dp);
  caar_dx.setup(elems_dx_caar,ref_FE,hvcoord,sphop_dx_caar);
  caar_dp.init_boundary_exchanges(bmm[MPI_EXCHANGE]);
  caar_dx.init_boundary_exchanges(bmm[MPI_EXCHANGE]);
  std::any_cast<CaarFunctorImplST<DpFadType>&>(caar_dp.impl()).m_run_limiter = false;
  std::any_cast<CaarFunctorImplST<DxFadTypeCaar>&>(caar_dx.impl()).m_run_limiter = false;

  auto& limiter_dp = c.create<LimiterFunctorST<DpFadType>>(elems_dp,hvcoord,params);
  auto& limiter_dx = c.create<LimiterFunctorST<DxFadTypeCaar>>(elems_dx_caar,hvcoord,params);
  limiter_dp.m_verbose = false;
  limiter_dx.m_verbose = false;

  auto& dirk_dp = c.create<DirkFunctorST<DpFadType>>(elems_dp.num_elems());
  auto& dirk_dx = c.create<DirkFunctorST<DxFadTypeDirk>>(elems_dx_dirk.num_elems());

  // Setup scratch buffers
  FunctorsBuffersManager fbm;
  fbm.request_size(caar_dp.requested_buffer_size());
  fbm.request_size(caar_dx.requested_buffer_size());
  fbm.request_size(dirk_dp.requested_buffer_size());
  fbm.request_size(dirk_dx.requested_buffer_size());
  fbm.request_size(limiter_dp.requested_buffer_size());
  fbm.request_size(limiter_dx.requested_buffer_size());

  fbm.allocate();

  caar_dp.init_buffers(fbm);
  caar_dx.init_buffers(fbm);
  dirk_dp.init_buffers(fbm);
  dirk_dx.init_buffers(fbm);
  limiter_dp.init_buffers(fbm);
  limiter_dx.init_buffers(fbm);

  // Scalar params
  double dt = 10;
  double eta_ave_w = 0.25;

  // Create the Imex tape
  c.any_map().try_emplace("imex_tape",std::in_place_type<Tape<StateSnapshot>>,11,num_elems);

  // Initial state
  StateSnapshot state_t0(num_elems);
  state_t0.randomize(seed,1e5,1e3,hvcoord.hybrid_ai0,geo.m_phis);

  // Run FWD sweep
  elems_dp.m_state.import_snapshot(state_t0,n0);
  elems_dp.m_state.randomize_derivs(seed,n0);
  auto du0 = elems_dp.m_state.take_deriv_snapshot(n0,0);
  printf(" -> Run forward problem...\n");
  ttype5_imex_timestep<DpFadType>(tl,dt,eta_ave_w);
  printf(" -> Run forward problem...done!\n");
  auto duN = elems_dp.m_state.take_deriv_snapshot(np1,0);

  // Run BWD pass
  StateSnapshot lambda(num_elems);
  lambda.randomize(seed,1.0,1.0/100,0.0);
  auto lambdaN = lambda.clone(true);
  printf(" -> Run adjoint problem...\n");
  ttype5_imex_adjoint(dt,eta_ave_w,lambda);
  printf(" -> Run adjoint problem...done!\n");
  auto lambda0 = lambda.clone(true);

  // Compare du0*lambda0 with duN*lambdaN
  auto v_dot0 = dot(ekat::scalarize(du0.v),ekat::scalarize(lambda0.v),nlevs);
  auto v_dotN = dot(ekat::scalarize(duN.v),ekat::scalarize(lambdaN.v),nlevs);

  auto vth_dot0 = dot(ekat::scalarize(du0.vtheta_dp),ekat::scalarize(lambda0.vtheta_dp),nlevs);
  auto vth_dotN = dot(ekat::scalarize(duN.vtheta_dp),ekat::scalarize(lambdaN.vtheta_dp),nlevs);

  auto dp_dot0 = dot(ekat::scalarize(du0.dp3d),ekat::scalarize(lambda0.dp3d),nlevs);
  auto dp_dotN = dot(ekat::scalarize(duN.dp3d),ekat::scalarize(lambdaN.dp3d),nlevs);

  auto w_dot0 = dot(ekat::scalarize(du0.w_i),ekat::scalarize(lambda0.w_i),nlevs+1);
  auto w_dotN = dot(ekat::scalarize(duN.w_i),ekat::scalarize(lambdaN.w_i),nlevs+1);

  auto phi_dot0 = dot(ekat::scalarize(du0.phinh_i),ekat::scalarize(lambda0.phinh_i),nlevs+1);
  auto phi_dotN = dot(ekat::scalarize(duN.phinh_i),ekat::scalarize(lambdaN.phinh_i),nlevs+1);

  constexpr auto tol = std::numeric_limits<double>::epsilon()*1e4;
  {
    using namespace Catch::Matchers;

    auto full_dot0 = v_dot0 + vth_dot0 + dp_dot0 + w_dot0 + phi_dot0;
    auto full_dotN = v_dotN + vth_dotN + dp_dotN + w_dotN + phi_dotN;
    CHECK_THAT (full_dot0, WithinRel(full_dotN,tol));

    if (comm.am_i_root())
      std::cout << std::setprecision(15)
                << "   <du0, lambda0> = " << full_dot0
                << ",  <duN, lambdaN> = " << full_dotN << "\n";
  }

  cleanup_f90();
  c.finalize_singleton();
}

} // namespace Homme
