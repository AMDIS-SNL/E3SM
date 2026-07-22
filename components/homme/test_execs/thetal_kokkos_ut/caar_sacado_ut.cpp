#include <catch2/catch.hpp>

#include "Types.hpp"
#include "Context.hpp"
#include "CaarFunctorImpl.hpp"
#include "SimulationParams.hpp"
#include "Tracers.hpp"
#include "PhysicalConstants.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_string_utils.hpp>
#include <ekat_comm.hpp>

#include <iomanip>

using namespace Homme;

extern "C" {
// Even if we don't run the f90 code in this unit test, it is easier to
// init from f90, which takes care of creating the grid and decomposing it
void init_f90 (const int& ne,
               const Real* hyai_ptr, const Real* hybi_ptr,
               const Real* hyam_ptr, const Real* hybm_ptr,
               Real* dvv, Real* mp,
               const Real& ps0);
void cleanup_f90();
}

TEST_CASE("caar_dp_check") {

  constexpr int ne = 2;
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
  c.create<ekat::Comm>(MPI_COMM_WORLD);

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.dp3d_thresh = 0; // don't let the limiter do anything, for now
  params.vtheta_thresh = 0; // don't let the limiter do anything, for now
  params.params_set = true;

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

  // Create elements with Real scalar type
  auto& elems = c.create<ElementsST<Real>>();
  elems.init(num_elems,false,true,PhysicalConstants::rearth0);
  auto& geo = elems.m_geometry;
  geo.randomize(seed);

  // Create elements with DpFadType scalar type
  auto& elems_dp = c.create<ElementsST<DpFadType>>();
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dp.m_geometry = geo; // Use same views for geometry

  // Get or create and init other structures needed
  auto& bm = c.create<MpiBuffersManager>();
  auto& sphop = c.create<SphereOperatorsST<Real>>();
  auto& sphop_dp = c.create<SphereOperatorsST<DpFadType>>();
  auto& tracers = c.create<TracersST<Real>>();
  auto& tracers_dp = c.create<TracersST<DpFadType>>();

  // The Caar functor also runs the limiter functor (bad design, imho)
  auto& limiter = c.create<LimiterFunctorST<Real>>(elems,hvcoord,params);
  auto& limiter_dp = c.create<LimiterFunctorST<DpFadType>>(elems_dp,hvcoord,params);
  limiter.m_verbose = false;
  limiter_dp.m_verbose = false;

  sphop_dp.setup(geo,ref_FE);
  sphop.setup(geo,ref_FE);
  if (!bm.is_connectivity_set ()) {
    bm.set_connectivity(c.get_ptr<Connectivity>());
  }

  // We compute sens w.r.t. the earth radius multiplying factor: R=R0*alpha, so dR/dalpha = R0
  DpFadType rearth0 = PhysicalConstants::rearth0;
  rearth0.fastAccessDx(0) = PhysicalConstants::rearth0;
  sphop_dp.m_scale_factor_inv = 1/rearth0;

  auto& comm = c.get<ekat::Comm>();
  const int rank = comm.rank();

  std::vector<Real> dp_factor = {1, 1e-2, 1e-4, 1e-6, 1e-8};

  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>> policy({0,0,0,0,0},{num_elems,2,NP,NP,NUM_PHYSICAL_LEV});
  for (const bool hydrostatic : {true,false}) {
    if (comm.am_i_root()) {
      std::cout << " -> " << (hydrostatic ? "Hydrostatic\n" : "Non-Hydrostatic\n");
    }
    params.theta_hydrostatic_mode = hydrostatic;
    limiter.m_theta_hydrostatic_mode = hydrostatic;
    limiter_dp.m_theta_hydrostatic_mode = hydrostatic;
    auto adv_forms = {AdvectionForm::Conservative, AdvectionForm::NonConservative};
    for (const AdvectionForm adv_form : adv_forms) {
      if (comm.am_i_root()) {
        std::cout << "  -> " << (adv_form==AdvectionForm::Conservative ? "Conservative" : "Non-Conservative") << " theta advection\n";
      }
      for (int rsplit : {3,0}) {
        if (comm.am_i_root()) {
          std::cout << "   -> rsplit = " << rsplit << "\n";
        }
        for (const int pgrad : {1,0}) {
          if (comm.am_i_root()) {
            std::cout << "    -> pgrad_correction = " << pgrad << "\n";
          }
          // Set the parameters
          params.theta_hydrostatic_mode = hydrostatic;
          params.theta_adv_form = adv_form;
          params.rsplit = rsplit;
          params.pgrad_correction = (pgrad != 0);

          // Generate RK stage data
          Real dt = RPDF(1.0,10.0)(engine);
          Real eta_ave_w = RPDF(0.1,1.0)(engine);
          Real scale1 = RPDF(1.0,2.0)(engine);
          Real scale2 = RPDF(1.0,2.0)(engine);
          Real scale3 = RPDF(1.0,2.0)(engine);

          // Sync scalars across ranks (only np1 is *really* necessary, but might as well...)
          auto mpi_comm = c.get<ekat::Comm>().mpi_comm();
          MPI_Bcast(&dt,1,MPI_DOUBLE,0,mpi_comm);
          MPI_Bcast(&scale1,1,MPI_DOUBLE,0,mpi_comm);
          MPI_Bcast(&scale2,1,MPI_DOUBLE,0,mpi_comm);
          MPI_Bcast(&scale3,1,MPI_DOUBLE,0,mpi_comm);
          MPI_Bcast(&eta_ave_w,1,MPI_DOUBLE,0,mpi_comm);

          const int  nm1 = 0;
          const int  n0  = 1;
          const int  np1 = 2;

          RKStageData data (nm1, n0, np1, 0, dt, eta_ave_w, scale1, scale2, scale3);

          // Randomize state, set derived stuff to 0
          elems.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,geo.m_phis);
          elems_dp.m_state.import_values(elems.m_state,n0);

          Kokkos::deep_copy(elems.m_derived.m_vn0,0);
          Kokkos::deep_copy(elems_dp.m_derived.m_vn0,DpFadType(0));
          Kokkos::deep_copy(elems.m_derived.m_omega_p,0);
          Kokkos::deep_copy(elems_dp.m_derived.m_omega_p,DpFadType(0));

          // Create the Caar functors
          CaarFunctorImplST<Real> caar(elems,tracers,ref_FE,hvcoord,sphop,params);
          CaarFunctorImplST<DpFadType> caar_dp(elems_dp,tracers_dp,ref_FE,hvcoord,sphop_dp,params);

          FunctorsBuffersManager fbm;
          fbm.request_size( caar.requested_buffer_size() );
          fbm.request_size( caar_dp.requested_buffer_size() );
          fbm.request_size( limiter.requested_buffer_size() );
          fbm.request_size( limiter_dp.requested_buffer_size() );
          fbm.allocate();
          caar.init_buffers(fbm);
          caar_dp.init_buffers(fbm);
          limiter.init_buffers(fbm);
          limiter_dp.init_buffers(fbm);
          caar.init_boundary_exchanges(c.get_ptr<MpiBuffersManager>());
          caar_dp.init_boundary_exchanges(c.get_ptr<MpiBuffersManager>());

          // RUN caar for ST=DpFadType
          caar_dp.run(data);
          auto dvdp = ekat::scalarize(elems_dp.m_state.m_v);

          // RUN caar for ST=Real, with exact rearth
          caar.run(data);

          // Back up v0 otherwise it will be overwritten in the loop below
          ExecViewManaged<Real*[NUM_TIME_LEVELS][2][NP][NP][NUM_LEV*VECTOR_SIZE]> v0("v0",num_elems);
          Kokkos::deep_copy(v0,ekat::scalarize(elems.m_state.m_v));
          
          // RUN caar again, with different values of rearth perturbation
          std::vector<Real> h;
          std::vector<Real> eh;
          for (auto factor : dp_factor) {
            auto rearth = PhysicalConstants::rearth0 * (1+factor);
            auto hval = h.emplace_back(factor);
            caar.m_sphere_ops.m_scale_factor_inv = 1/rearth;
            caar.run(data);

            auto vh = ekat::scalarize(elems.m_state.m_v);

            auto linf = KOKKOS_LAMBDA(int ie, int icmp, int ip, int jp, int ilev, Real& accum) {
              Real dvdp_fd = (vh(ie,np1,icmp,ip,jp,ilev) - v0(ie,np1,icmp,ip,jp,ilev)) / hval;
              Real dvdp_ex = dvdp(ie,np1,icmp,ip,jp,ilev).fastAccessDx(0);
              Real lcl_err = (dvdp_fd-dvdp_ex)*(dvdp_fd-dvdp_ex);
              accum += lcl_err;
            };
            Kokkos::parallel_reduce(policy,linf,eh.emplace_back(0));
            eh.back() = std::sqrt(eh.back());
          }

          // Compute and print error and order of conv
          std::vector<Real> order;
          for (size_t i=1; i<dp_factor.size(); ++i) {
            auto err_reduction = eh[i] / eh[i-1];
            auto h_reduction = h[i] / h[i-1];

            order.push_back(std::log(err_reduction)/std::log(h_reduction));
          }

          if (comm.am_i_root()) {
            std::cout << "      h = [" << ekat::join(h,",") << "]\n";
            std::cout << "      |dv/dp - FD(dvdp)|_inf = [" << ekat::join(eh,",") << "]\n";
            std::cout << "      order: [" << ekat::join(order,",") << "\n";
          }

          auto min_err = *std::min_element(eh.begin(),eh.end());

          // Not sure how to test that the FD deriv approaches the Sacado one before diverging due to finite precision errors.
          // For now, I simply check that at some point, it is 3 orders of magnitues lower than the initial error
          REQUIRE (min_err < eh[0]/1e3);
        }
      }
    }
  }

  cleanup_f90();
  c.finalize_singleton();
}

// Compute Dx/Dp two ways, and compare:
//  - Use ST=DpFadType to compute Dx_new/Dp from Dx_old/Dp
//  - Use product rule: Dx_new/Dp = Dx_new/Dx_old * Dx_old/Dp
TEST_CASE("caar_dx_check") {
  constexpr int ne = 2;
  const auto A = Kokkos::ALL();
  using DxFadType = DxFadTypeCaar;

  // The random numbers generator to init the state
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  // Use stuff from Context, to increase similarity with actual runs
  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.dp3d_thresh = 0; // don't let the limiter do anything, for now
  params.vtheta_thresh = 0; // don't let the limiter do anything, for now
  params.params_set = true;
  params.rsplit = 3;

  // Create and init hvcoord and ref_elem
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

  // Create elements with Real scalar type
  auto& elems = c.create<ElementsST<Real>>();
  elems.init(num_elems,false,true,PhysicalConstants::rearth0);
  auto& geo = elems.m_geometry;
  geo.randomize(seed);

  // Create elements with DpFadType scalar type
  auto& elems_dp = c.create<ElementsST<DpFadType>>();
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dp.m_geometry = geo; // Use same views for geometry

  // Create elements with DxFadType scalar type
  auto& elems_dx = c.create<ElementsST<DxFadType>>();
  elems_dx.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx.m_geometry = geo; // Use same views for geometry

  // Get or create and init other structures needed
  auto& bm = c.create<MpiBuffersManager>();
  auto& sphop = c.create<SphereOperatorsST<Real>>();
  auto& sphop_dp = c.create<SphereOperatorsST<DpFadType>>();
  auto& sphop_dx = c.create<SphereOperatorsST<DxFadType>>();
  auto& tracers = c.create<TracersST<Real>>();
  auto& tracers_dp = c.create<TracersST<DpFadType>>();
  auto& tracers_dx = c.create<TracersST<DxFadType>>();

  // The Caar functor also runs the limiter functor (bad design, imho)
  auto& limiter = c.create<LimiterFunctorST<Real>>(elems,hvcoord,params);
  auto& limiter_dp = c.create<LimiterFunctorST<DpFadType>>(elems_dp,hvcoord,params);
  auto& limiter_dx = c.create<LimiterFunctorST<DxFadType>>(elems_dx,hvcoord,params);
  limiter.m_verbose = false;
  limiter_dp.m_verbose = false;
  limiter_dx.m_verbose = false;

  sphop_dx.setup(geo,ref_FE);
  sphop_dp.setup(geo,ref_FE);
  sphop.setup(geo,ref_FE);
  if (!bm.is_connectivity_set ()) {
    bm.set_connectivity(c.get_ptr<Connectivity>());
  }

  auto& comm = c.get<ekat::Comm>();
  const int rank = comm.rank();

  auto rtol = 1e-10;
  auto atol = 1e-10;

  auto& catch_capture = Catch::getResultCapture();
  // NOTE: cannot use hydrostatic=true, since it requires a scan sum
  for (const bool hydrostatic : {false}) {
    if (comm.am_i_root()) {
      std::cout << " -> " << (hydrostatic ? "Hydrostatic\n" : "Non-Hydrostatic\n");
    }
    params.theta_hydrostatic_mode = hydrostatic;
    limiter.m_theta_hydrostatic_mode = hydrostatic;
    limiter_dp.m_theta_hydrostatic_mode = hydrostatic;
    auto adv_forms = {AdvectionForm::Conservative, AdvectionForm::NonConservative};
    for (const AdvectionForm adv_form : adv_forms) {
      if (comm.am_i_root()) {
        std::cout << "  -> " << (adv_form==AdvectionForm::Conservative ? "Conservative" : "Non-Conservative") << " theta advection\n";
      }
      for (const int pgrad : {1,0}) {
        if (comm.am_i_root()) {
          std::cout << "    -> pgrad_correction = " << pgrad << "\n";
        }
        // Set the parameters
        params.theta_hydrostatic_mode = hydrostatic;
        params.theta_adv_form = adv_form;
        params.pgrad_correction = (pgrad != 0);

        // Generate RK stage data
        Real dt = RPDF(1.0,10.0)(engine);
        Real eta_ave_w = RPDF(0.1,1.0)(engine);
        Real scale1 = RPDF(1.0,2.0)(engine);
        Real scale2 = RPDF(1.0,2.0)(engine);
        Real scale3 = RPDF(1.0,2.0)(engine);

        // Sync scalars across ranks (only np1 is *really* necessary, but might as well...)
        auto mpi_comm = Context::singleton().get<ekat::Comm>().mpi_comm();
        MPI_Bcast(&dt,1,MPI_DOUBLE,0,mpi_comm);
        MPI_Bcast(&scale1,1,MPI_DOUBLE,0,mpi_comm);
        MPI_Bcast(&scale2,1,MPI_DOUBLE,0,mpi_comm);
        MPI_Bcast(&scale3,1,MPI_DOUBLE,0,mpi_comm);
        MPI_Bcast(&eta_ave_w,1,MPI_DOUBLE,0,mpi_comm);

        const int  nm1 = 0;
        const int  n0  = 1;
        const int  np1 = 2;

        RKStageData data (nm1, n0, np1, 0, dt, eta_ave_w, scale1, scale2, scale3);

        // Randomize state, set derived stuff to 0
        elems_dp.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,geo.m_phis);

        // We initialize also dp derivs of elems_dp at slice tl.n0 with random values
        elems_dp.m_state.randomize_derivs(seed,n0);

        // Extract derivs into a non-fad type for the Dp = Dx*Dp_old test
        auto dxdp0 = elems_dp.m_state.take_deriv_snapshot(n0,0);

        // Init d/dx state
        elems_dx.m_state.import_values(elems_dp.m_state,n0);
        elems_dx.m_state.import_values(elems_dp.m_state,nm1);

        Kokkos::deep_copy(elems_dp.m_derived.m_vn0,DpFadType(0));
        Kokkos::deep_copy(elems_dp.m_derived.m_omega_p,DpFadType(0));
        Kokkos::deep_copy(elems_dx.m_derived.m_vn0,DxFadType(0));
        Kokkos::deep_copy(elems_dx.m_derived.m_omega_p,DxFadType(0));

        // Create the Caar functors
        CaarFunctorImplST<DpFadType> caar_dp(elems_dp,tracers_dp,ref_FE,hvcoord,sphop_dp,params);
        CaarFunctorImplST<DxFadType> caar_dx(elems_dx,tracers_dx,ref_FE,hvcoord,sphop_dx,params);

        FunctorsBuffersManager fbm;
        fbm.request_size( caar_dp.requested_buffer_size() );
        fbm.request_size( caar_dx.requested_buffer_size() );
        fbm.request_size( limiter_dp.requested_buffer_size() );
        fbm.request_size( limiter_dx.requested_buffer_size() );
        fbm.allocate();
        caar_dp.init_buffers(fbm);
        caar_dx.init_buffers(fbm);
        limiter_dp.init_buffers(fbm);
        limiter_dx.init_buffers(fbm);
        caar_dp.init_boundary_exchanges(c.get_ptr<MpiBuffersManager>());
        caar_dx.init_boundary_exchanges(c.get_ptr<MpiBuffersManager>());

        // RUN caar for ST=DpFadType
        caar_dp.run_pre_exchange(data);

        // RUN caar's J*V functor for ST=DxFadType
        caar_dx.init_J(data);
        caar_dx.run_pre_exchange(data);
        elems.m_state.import_snapshot(dxdp0,n0,0);
        caar_dx.run_JV(data,elems.m_state);

        // Check that dXnew/dp = dXnew/dXold * dXold/dp. dXnew/dp is in elems_dp.m_state at slice np1
        // while dXnew/dXold is in elems_dx.m_state at slice np1

        auto v_dp   = ekat::scalarize(elems_dp.m_state.m_v);
        auto vth_dp = ekat::scalarize(elems_dp.m_state.m_vtheta_dp);
        auto dp_dp  = ekat::scalarize(elems_dp.m_state.m_dp3d);
        auto phi_dp = ekat::scalarize(elems_dp.m_state.m_phinh_i);
        auto w_dp   = ekat::scalarize(elems_dp.m_state.m_w_i);
        auto v_dp_h   = Kokkos::create_mirror_view(v_dp);
        auto vth_dp_h = Kokkos::create_mirror_view(vth_dp);
        auto dp_dp_h  = Kokkos::create_mirror_view(dp_dp);
        auto phi_dp_h = Kokkos::create_mirror_view(phi_dp);
        auto w_dp_h   = Kokkos::create_mirror_view(w_dp);
        Kokkos::deep_copy(v_dp_h,   v_dp);
        Kokkos::deep_copy(vth_dp_h, vth_dp);
        Kokkos::deep_copy(dp_dp_h,  dp_dp);
        Kokkos::deep_copy(phi_dp_h, phi_dp);
        Kokkos::deep_copy(w_dp_h,   w_dp);

        auto v_JV   = ekat::scalarize(elems.m_state.m_v);
        auto vth_JV = ekat::scalarize(elems.m_state.m_vtheta_dp);
        auto dp_JV  = ekat::scalarize(elems.m_state.m_dp3d);
        auto phi_JV = ekat::scalarize(elems.m_state.m_phinh_i);
        auto w_JV   = ekat::scalarize(elems.m_state.m_w_i);
        auto v_JV_h   = Kokkos::create_mirror_view(v_JV);
        auto vth_JV_h = Kokkos::create_mirror_view(vth_JV);
        auto dp_JV_h  = Kokkos::create_mirror_view(dp_JV);
        auto phi_JV_h = Kokkos::create_mirror_view(phi_JV);
        auto w_JV_h   = Kokkos::create_mirror_view(w_JV);
        Kokkos::deep_copy(v_JV_h,   v_JV);
        Kokkos::deep_copy(vth_JV_h, vth_JV);
        Kokkos::deep_copy(dp_JV_h,  dp_JV);
        Kokkos::deep_copy(phi_JV_h, phi_JV);
        Kokkos::deep_copy(w_JV_h,   w_JV);

        for (int ie=0; ie<num_elems; ++ie) {
          for (int igp=0; igp<NP; ++igp) {
            for (int jgp=0; jgp<NP; ++jgp) {
              for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
                auto du_src = v_JV_h(ie,np1,0,igp,jgp,k), du_tgt = v_dp_h(ie,np1,0,igp,jgp,k).dx(0);
                CHECK_THAT (du_src, Catch::WithinRel(du_tgt,rtol) || Catch::WithinAbs(du_tgt,atol));

                auto dv_src = v_JV_h(ie,np1,1,igp,jgp,k), dv_tgt = v_dp_h(ie,np1,1,igp,jgp,k).dx(0);
                CHECK_THAT (dv_src, Catch::WithinRel(dv_tgt,rtol) || Catch::WithinAbs(dv_tgt,atol));

                auto dvth_src = vth_JV_h(ie,np1,igp,jgp,k), dvth_tgt = vth_dp_h(ie,np1,igp,jgp,k).dx(0);
                CHECK_THAT (dvth_src, Catch::WithinRel(dvth_tgt,rtol) || Catch::WithinAbs(dvth_tgt,atol));

                auto ddp_src = dp_JV_h(ie,np1,igp,jgp,k), ddp_tgt = dp_dp_h(ie,np1,igp,jgp,k).dx(0);
                CHECK_THAT (ddp_src, Catch::WithinRel(ddp_tgt,rtol) || Catch::WithinAbs(ddp_tgt,atol));

                auto dphi_src = phi_JV_h(ie,np1,igp,jgp,k), dphi_tgt = phi_dp_h(ie,np1,igp,jgp,k).dx(0);
                CHECK_THAT (dphi_src, Catch::WithinRel(dphi_tgt,rtol) || Catch::WithinAbs(dphi_tgt,atol));

                auto dw_src = w_JV_h(ie,np1,igp,jgp,k), dw_tgt = w_dp_h(ie,np1,igp,jgp,k).dx(0);
                CHECK_THAT (dw_src, Catch::WithinRel(dw_tgt,rtol) || Catch::WithinAbs(dw_tgt,atol));
              }
              int k = NUM_PHYSICAL_LEV;

              auto dphi_src = phi_JV_h(ie,np1,igp,jgp,k), dphi_tgt = phi_dp_h(ie,np1,igp,jgp,k).dx(0);
              CHECK_THAT (dphi_src, Catch::WithinRel(dphi_tgt,rtol) || Catch::WithinAbs(dphi_tgt,atol));

              auto dw_src = w_JV_h(ie,np1,igp,jgp,k), dw_tgt = w_dp_h(ie,np1,igp,jgp,k).dx(0);
              CHECK_THAT (dw_src, Catch::WithinRel(dw_tgt,rtol) || Catch::WithinAbs(dw_tgt,atol));
            }
          }
        }
      }
    }
  }

  cleanup_f90();
  c.finalize_singleton();
}

// Verify the JtV transpose identity: <a, J*b> == <J^T*a, b>.
TEST_CASE("caar_jtv_check") {
  constexpr int ne = 2;
  using DxFadType = DxFadTypeCaar;

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);

  auto& params = c.create<SimulationParams>();
  params.dp3d_thresh   = 0;
  params.vtheta_thresh = 0;
  params.params_set    = true;
  params.rsplit        = 3;

  auto& hvcoord = c.create<HybridVCoord>();
  auto& ref_FE  = c.create<ReferenceElement>();
  hvcoord.random_init(seed);

  auto hyai = Kokkos::create_mirror_view(hvcoord.hybrid_ai);
  auto hybi = Kokkos::create_mirror_view(hvcoord.hybrid_bi);
  auto hyam = Kokkos::create_mirror_view(hvcoord.hybrid_am);
  auto hybm = Kokkos::create_mirror_view(hvcoord.hybrid_bm);
  Kokkos::deep_copy(hyai, hvcoord.hybrid_ai);
  Kokkos::deep_copy(hybi, hvcoord.hybrid_bi);
  Kokkos::deep_copy(hyam, hvcoord.hybrid_am);
  Kokkos::deep_copy(hybm, hvcoord.hybrid_bm);

  HostViewManaged<Real[NUM_PHYSICAL_LEV]> hyam_r(""), hybm_r("");
  for (int i = 0; i < NUM_PHYSICAL_LEV; ++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyam_r(i) = ADValue(hyam(ilev)[ivec]);
    hybm_r(i) = ADValue(hybm(ilev)[ivec]);
  }

  std::vector<Real> dvv(NP*NP), mp(NP*NP);
  init_f90(ne, hyai.data(), hybi.data(), hyam_r.data(), hybm_r.data(),
           dvv.data(), mp.data(), hvcoord.ps0);
  ref_FE.init_mass(mp.data());
  ref_FE.init_deriv(dvv.data());

  const int  num_elems    = c.get<Connectivity>().get_num_local_elements();
  const Real max_pressure = 1000.0 + hvcoord.ps0;

  auto& elems = c.create<ElementsST<Real>>();
  elems.init(num_elems, false, true, PhysicalConstants::rearth0);
  auto& geo = elems.m_geometry;
  geo.randomize(seed);

  auto& elems_dx = c.create<ElementsST<DxFadType>>();
  elems_dx.init(num_elems, false, true, PhysicalConstants::rearth0);
  elems_dx.m_geometry = geo;
  Kokkos::deep_copy(elems_dx.m_derived.m_vn0,DxFadType(0));
  Kokkos::deep_copy(elems_dx.m_derived.m_vn0,DxFadType(0));

  auto& bm       = c.create<MpiBuffersManager>();
  auto& sphop    = c.create<SphereOperatorsST<Real>>();
  auto& sphop_dx = c.create<SphereOperatorsST<DxFadType>>();
  auto& tracers    = c.create<TracersST<Real>>();
  auto& tracers_dx = c.create<TracersST<DxFadType>>();

  auto& limiter    = c.create<LimiterFunctorST<Real>>(elems, hvcoord, params);
  auto& limiter_dx = c.create<LimiterFunctorST<DxFadType>>(elems_dx, hvcoord, params);
  limiter.m_verbose    = false;
  limiter_dx.m_verbose = false;

  sphop.setup(geo, ref_FE);
  sphop_dx.setup(geo, ref_FE);
  if (!bm.is_connectivity_set()) {
    bm.set_connectivity(c.get_ptr<Connectivity>());
  }

  auto& comm    = c.get<ekat::Comm>();
  auto mpi_comm = comm.mpi_comm();

  const int nm1 = 0, n0 = 1, np1 = 2;

  // Two adjoint vectors, created independently of the Context.
  ElementsStateST<Real> adj_a, adj_b;
  adj_a.init(num_elems);
  adj_b.init(num_elems);

  // Scalarized device views (share memory with the packed views above).
  auto sa_v   = ekat::scalarize(adj_a.m_v);
  auto sa_vth = ekat::scalarize(adj_a.m_vtheta_dp);
  auto sa_dp  = ekat::scalarize(adj_a.m_dp3d);
  auto sa_phi = ekat::scalarize(adj_a.m_phinh_i);
  auto sa_w   = ekat::scalarize(adj_a.m_w_i);

  auto sb_v   = ekat::scalarize(adj_b.m_v);
  auto sb_vth = ekat::scalarize(adj_b.m_vtheta_dp);
  auto sb_dp  = ekat::scalarize(adj_b.m_dp3d);
  auto sb_phi = ekat::scalarize(adj_b.m_phinh_i);
  auto sb_w   = ekat::scalarize(adj_b.m_w_i);

  using p4_mid_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
  using p5_mid_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>;
  using p4_int_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
  const p4_mid_t p4_mid({0,0,0,0}, {num_elems, NP, NP, NUM_PHYSICAL_LEV});
  const p5_mid_t p5_mid({0,0,0,0,0}, {num_elems, 2, NP, NP, NUM_PHYSICAL_LEV});
  const p4_int_t p4_int({0,0,0,0}, {num_elems, NP, NP, NUM_INTERFACE_LEV});

  const Real rtol = 1e-8;
  const Real atol = 1e-8;

  // NOTE: cannot use hydrostatic=true (requires a scan sum not supported here)
  for (const bool hydrostatic : {false}) {
    params.theta_hydrostatic_mode       = hydrostatic;
    limiter.m_theta_hydrostatic_mode    = hydrostatic;
    limiter_dx.m_theta_hydrostatic_mode = hydrostatic;
    if (comm.am_i_root())
      std::cout << " -> " << (hydrostatic ? "Hydrostatic\n" : "Non-Hydrostatic\n");

    for (const AdvectionForm adv_form : {AdvectionForm::Conservative,
                                         AdvectionForm::NonConservative}) {
      params.theta_adv_form = adv_form;
      if (comm.am_i_root())
        std::cout << "  -> " << (adv_form==AdvectionForm::Conservative
                                 ? "Conservative" : "Non-Conservative")
                  << " theta advection\n";

      for (const int pgrad : {1, 0}) {
        params.pgrad_correction = (pgrad != 0);
        if (comm.am_i_root())
          std::cout << "    -> pgrad_correction = " << pgrad << "\n";

        Real dt        = RPDF(1.0, 10.0)(engine);
        Real eta_ave_w = RPDF(0.1,  1.0)(engine);
        Real scale1    = RPDF(1.0,  2.0)(engine);
        Real scale2    = RPDF(1.0,  2.0)(engine);
        Real scale3    = RPDF(1.0,  2.0)(engine);
        MPI_Bcast(&dt,        1, MPI_DOUBLE, 0, mpi_comm);
        MPI_Bcast(&scale1,    1, MPI_DOUBLE, 0, mpi_comm);
        MPI_Bcast(&scale2,    1, MPI_DOUBLE, 0, mpi_comm);
        MPI_Bcast(&scale3,    1, MPI_DOUBLE, 0, mpi_comm);
        MPI_Bcast(&eta_ave_w, 1, MPI_DOUBLE, 0, mpi_comm);

        RKStageData data(nm1, n0, np1, 0, dt, eta_ave_w, scale1, scale2, scale3);

        // State at which J is evaluated
        elems_dx.m_state.randomize(seed, max_pressure, hvcoord.ps0,
                                   hvcoord.hybrid_ai0, geo.m_phis);

        CaarFunctorImplST<DxFadType> caar_dx(elems_dx, tracers_dx,
                                             ref_FE, hvcoord, sphop_dx, params);
        FunctorsBuffersManager fbm;
        fbm.request_size(caar_dx.requested_buffer_size());
        fbm.request_size(limiter_dx.requested_buffer_size());
        fbm.allocate();
        caar_dx.init_buffers(fbm);
        limiter_dx.init_buffers(fbm);
        caar_dx.init_boundary_exchanges(c.get_ptr<MpiBuffersManager>());

        // Randomize adj_state a and b (make them equal);
        adj_a.randomize(seed, max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, geo.m_phis);
        adj_b.import_values(adj_a,n0);
        adj_b.import_values(adj_a,nm1);

        // Init d/dx(n0) structures
        caar_dx.init_J(data);

        // Compute d/dx(np1)
        caar_dx.run_pre_exchange(data);

        // J*b   -> adj_b[np1]
        caar_dx.run_JV(data, adj_b);
        // J^T*a -> adj_a[np1]
        caar_dx.run_JtV(data, adj_a);

        // dot1 = <a[n0], (J*b)[np1]>
        // dot2 = <(J^T*a)[np1], b[n0]>
        // Note: we must sum over all state vars, since J and J^T scramble them differently
        Real2 V_dot, vth_dot, dp_dot, phi_dot, w_dot;
        Real2 gdot;

        Kokkos::parallel_reduce(p5_mid,
          KOKKOS_LAMBDA(int ie, int icmp, int ip, int jp, int k, Real2& acc) {
            acc.v[0] += sa_v(ie, n0,  icmp, ip, jp, k) * sb_v(ie, np1, icmp, ip, jp, k);
            acc.v[1] += sa_v(ie, np1, icmp,  ip, jp, k) * sb_v(ie, n0, icmp,  ip, jp, k);
          }, V_dot);
        Kokkos::parallel_reduce(p4_mid,
          KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
            acc.v[0] += sa_vth(ie, n0,  ip, jp, k) * sb_vth(ie, np1, ip, jp, k);
            acc.v[1] += sa_vth(ie, np1, ip, jp, k) * sb_vth(ie, n0,  ip, jp, k);
          }, vth_dot);
        Kokkos::parallel_reduce(p4_mid,
          KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
            acc.v[0] += sa_dp(ie, n0,  ip, jp, k) * sb_dp(ie, np1, ip, jp, k);
            acc.v[1] += sa_dp(ie, np1, ip, jp, k) * sb_dp(ie, n0,  ip, jp, k);
          }, dp_dot);
        Kokkos::parallel_reduce(p4_int,
          KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
            acc.v[0] += sa_phi(ie, n0,  ip, jp, k) * sb_phi(ie, np1, ip, jp, k);
            acc.v[1] += sa_phi(ie, np1, ip, jp, k) * sb_phi(ie, n0,  ip, jp, k);
          }, phi_dot);
        Kokkos::parallel_reduce(p4_int,
          KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
            acc.v[0] += sa_w(ie, n0,  ip, jp, k) * sb_w(ie, np1, ip, jp, k);
            acc.v[1] += sa_w(ie, np1, ip, jp, k) * sb_w(ie, n0,  ip, jp, k);
          }, w_dot);

        gdot += V_dot;
        gdot += vth_dot;
        gdot += dp_dot;
        gdot += phi_dot;
        gdot += w_dot;

        CHECK_THAT(gdot.v[0], Catch::WithinRel(gdot.v[1], rtol) || Catch::WithinAbs(gdot.v[1], atol));

        if (comm.am_i_root())
          std::cout << std::setprecision(15)
                    << "       <a, J*b> = " << gdot.v[0]
                    << ",  <J^T*a, b> = " << gdot.v[1] << "\n";
      }
    }
  }

  cleanup_f90();
  c.finalize_singleton();
}
