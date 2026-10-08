#include <catch2/catch.hpp>

#include "Types.hpp"
#include "Context.hpp"
#include "RemapFunctor.hpp"
#include "SimulationParams.hpp"
#include "Tracers.hpp"
#include "PhysicalConstants.hpp"
#include "HybridVCoord.hpp"
#include "Elements.hpp"
#include "mpi/Connectivity.hpp"
#include "RKStageData.hpp"
#include "PpmRemap.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_string_utils.hpp>
#include <ekat_comm.hpp>

#include <iomanip>
#include <random>

#include "FunctorsBuffersManager.hpp"
#include "VerticalRemapManager.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/ViewUtils.hpp"
#include "utilities/MathUtils.hpp"

// #include <cfenv>
// #include "ekat_fpe.hpp"

using namespace Homme;

// Test Fad dp calculation roughly matches a finite difference
TEST_CASE("remap", "remap_dp") {

  // The random numbers generator
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;
  using IPDF = std::uniform_int_distribution<int>;

  // Use stuff from Context, to increase similarity with actual runs
  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);
  auto& comm = c.get<ekat::Comm>();

  if (comm.am_i_root()) {
    std::cout << "Running remap_dp_check..." << std::endl;
  }

  // Init connectivity
  auto& conn = c.create<Connectivity>();
  conn.set_comm(comm);
  conn.set_num_elements(4);
  conn.finalize();
  const int num_elems = c.get<Connectivity>().get_num_local_elements();

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.params_set = true;
  params.qsize = std::max(int(QSIZE_D-1),0);

  // Create and init hvcoord
  auto& hvcoord = c.create<HybridVCoord>();
  hvcoord.random_init(seed);

  // Force ps0 and hybrid_ai0 to be something reasonable to avoid negative thickness
  // ps0 = 100.0, hybrid_ai0 = 1.0
  // hvcoord.ps0 = 100.0;
  // hvcoord.hybrid_ai0 = 1.0;
  const auto max_pressure = 1000.0 + hvcoord.ps0; // This ensures max_p > ps0

  // Create elements geometry
  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0);
  geo.randomize(seed);

  // Create elements with Real scalar type with original values since remap overwrites them
  // Can't use Context because it allows creating only one of each type
  ElementsST<Real> elems_orig;
  elems_orig.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_orig.m_geometry = geo; // Use same views for geometry

  // Create elements with Real scalar type with the base (unperturbed) values in the F.D. calculation
  // Can't use Context because it allows creating only one of each type
  ElementsST<Real> elems_base;
  elems_base.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_base.m_geometry = geo; // Use same views for geometry

  // Create elements with Real scalar type -- what is computed by remap
  auto& elems = c.create<ElementsST<Real>>();
  elems.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems.m_geometry = geo; // Use same views for geometry

  // Create elements with DpFadType scalar type -- what is computed by remap with Fad type
  auto& elems_dp = c.create<ElementsST<DpFadType>>();
  elems_dp.init(num_elems, false, true, PhysicalConstants::rearth0);
  elems_dp.m_geometry = geo; // Use same views for geometry

  // Create tracers
  TracersST<Real> tracers_orig, tracers_base;
  auto& tracers = c.create<TracersST<Real>>();
  auto& tracers_dp = c.create<TracersST<DpFadType>>();
  tracers_orig.init(num_elems,params.qsize);
  tracers_base.init(num_elems,params.qsize);
  tracers.init(num_elems,params.qsize);
  tracers_dp.init(num_elems,params.qsize);

  // Lambda to compute min of dp3d, to give meaningful initial value to derived.m_eta_dot_dpdn
  auto dp3d_min = [] (decltype(elems.m_state.m_dp3d) dp3d) -> Real {
    Real the_min = std::numeric_limits<Real>::max();
    for (int ie = 0; ie < dp3d.extent_int(0); ++ie) {
      // Because this constraint is difficult to satisfy for all of the tensors,
      // incrementally generate the view
      for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            ExecViewUnmanaged<Scalar[NUM_LEV]> pt_dp3d =
                Homme::subview(dp3d, ie, tl, igp, jgp);
            auto h_dp3d = Kokkos::create_mirror_view(pt_dp3d);
            Kokkos::deep_copy(h_dp3d,pt_dp3d);
            for (int ilev=0; ilev<NUM_LEV; ++ilev) {
              for (int iv=0; iv<VECTOR_SIZE; ++iv) {
                the_min = std::min(the_min,ADValue(h_dp3d(ilev)[iv]));
              }
            }
          }
        }
      }
    }
    return the_min;
  };

  // Tests don't seem to pass using PPM_LIMITED_EXTRAP
  auto remap_algs = {RemapAlg::PPM_MIRRORED/*, RemapAlg::PPM_LIMITED_EXTRAP*/};
  for (const bool hydrostatic : {true, false}) {
    std::cout << " -> " << (hydrostatic ? "hydrostatic" : "non-hydrostatic") << "\n";
    for (const int rsplit : {3,0}) {
      std::cout << "   -> rsplit = " << rsplit << "\n";
      for (auto alg : remap_algs) {
        std::cout << "     -> remap alg = " << remapAlg2str(alg) << "\n";
        
        // Set the parameters
        params.rsplit = rsplit;
        params.remap_alg = alg;
        params.theta_hydrostatic_mode = hydrostatic;

        // Generate timestep stage data
        const Real dt      = RPDF(1.0,100.0)(engine);
        const int nm1 = 0, n0 = 1, np1 = 2, np1_qdp = 0;

        // Randomize state/derived/tracers
        elems_orig.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,geo.m_phis);
        elems_orig.m_derived.randomize(seed,dp3d_min(elems_orig.m_state.m_dp3d));
        for (int tl=0; tl<NUM_TIME_LEVELS; ++tl) {
          elems_base.m_state.import_values(elems_orig.m_state,tl);
          elems.m_state.import_values(elems_orig.m_state,tl);
          elems_dp.m_state.import_values(elems_orig.m_state,tl);
        }
        elems_base.m_derived.import_values(elems_orig.m_derived);
        elems.m_derived.import_values(elems_orig.m_derived);
        elems_dp.m_derived.import_values(elems_orig.m_derived);

        // Set a nonzero derivative for an input
        auto dp3d_v_orig = ekat::scalarize(elems_orig.m_state.m_dp3d);
        auto dp3d_v_dp = ekat::scalarize(elems_dp.m_state.m_dp3d);
        Kokkos::parallel_for(Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0,0,0},{num_elems,NP,NP}),
                            [=](int ie, int ip, int jp){
                              for(int ilev=0; ilev<NUM_LEV; ++ilev){
                                DpFadType val = dp3d_v_orig(ie,np1,ip,jp,ilev);
                                // Mark ALL as active for differentiation to match the global perturbation in FD
                                val.fastAccessDx(0) = 1.0;
                                dp3d_v_dp(ie,np1,ip,jp,ilev) = val;
                              }
                            });

        tracers_orig.randomize(seed);
        tracers_base.import_values(tracers_orig, 0);
        tracers.import_values(tracers_orig, 0);
        tracers_dp.import_values(tracers_orig, 0);

        // Create the remap functor
        // Note: ALL the options must be set in params *before* creating the vrm.
        VerticalRemapManagerST<Real> vrm;
        VerticalRemapManagerST<DpFadType> vrm_dp;
        FunctorsBuffersManager fbm;
        fbm.request_size(vrm.requested_buffer_size());
        fbm.request_size(vrm_dp.requested_buffer_size());
        fbm.allocate();
        vrm.init_buffers(fbm);
        vrm_dp.init_buffers(fbm);

        // int mask = FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW;
        // ekat::enable_fpes(mask);

        // Run base (unperturbed) computation and save tracers
        vrm.run_remap(np1,np1_qdp,dt);
        auto qdp_base = ekat::scalarize(tracers.qdp);
        auto qdp_base_h = Kokkos::create_mirror(qdp_base);
        Kokkos::deep_copy(qdp_base_h, qdp_base);
        Real f_base = qdp_base_h(0,0,0,0,0,0);

        // Reset elems/tracers to original values
        for (int tl=0; tl<NUM_TIME_LEVELS; ++tl) {
          elems_base.m_state.import_values(elems_orig.m_state,tl);
        }
        elems_base.m_derived.import_values(elems_orig.m_derived);
        tracers_base.import_values(tracers_orig, 0);

        // Run AD computation
        vrm_dp.run_remap(np1,np1_qdp,dt);
        auto qdp_dp = ekat::scalarize(tracers_dp.qdp);
        auto qdp_dp_h = Kokkos::create_mirror_view(qdp_dp);
        Kokkos::deep_copy(qdp_dp_h, qdp_dp);
        Real f_ad = qdp_dp_h(0,0,0,0,0,0).val();
        Real df_ad = qdp_dp_h(0,0,0,0,0,0).fastAccessDx(0);

        // Check values agree
        if (comm.am_i_root()) {
          std::cout << "Val: " << f_base << ", AD: " << f_ad << std::endl;
        }
        CHECK(f_base == Approx(f_ad).margin(1e-15));

        std::vector<Real> dp_factor = {1e-4, 1e-6, 1e-8};
        for (auto factor : dp_factor) {

          // Reset state and tracers back to orig values
          for (int tl=0; tl<NUM_TIME_LEVELS; ++tl) {
            elems.m_state.import_values(elems_orig.m_state,tl);
          }
          elems.m_derived.import_values(elems_orig.m_derived);
          tracers.import_values(tracers_orig, 0);

          // Perturb all dp3d_v to ensure some effect
          auto dp3d_v = ekat::scalarize(elems.m_state.m_dp3d);
          auto dp3d_v_h = Kokkos::create_mirror_view(dp3d_v);
          for(int ie=0; ie<num_elems; ++ie)
            for(int ip=0; ip<NP; ++ip)
              for(int jp=0; jp<NP; ++jp)
                for(int ilev=0; ilev<NUM_LEV; ++ilev)
                  dp3d_v_h(ie,np1,ip,jp,ilev) = dp3d_v_orig(ie,np1,ip,jp,ilev) + factor;
          Kokkos::deep_copy(dp3d_v, dp3d_v_h);

          // Run remap with perturbed values
          vrm.run_remap(np1,np1_qdp,dt);
          auto qdp_pert = ekat::scalarize(tracers.qdp);
          auto qdp_pert_h = Kokkos::create_mirror_view(qdp_pert);
          Kokkos::deep_copy(qdp_pert_h, qdp_pert);
          Real f_pert = qdp_pert_h(0,0,0,0,0,0);
          Real df_fd = (f_pert - f_base)/factor;

          if (comm.am_i_root()) {
            std::cout.precision(10);
            std::cout << "factor: " << factor << ", val: " << f_base << " , pert: " << f_pert
                      << ", FD: " << df_fd << ", AD: " << df_ad << ", error: " << df_fd - df_ad << std::endl;
          }
          CHECK(df_fd == Approx(df_ad).margin(1e-5));

          // Check all tracer values
          const int tl=0;
          for (int ie=0; ie<num_elems; ++ie) {
            for (int it=0; it<params.qsize; ++it) {
              for (int igp=0; igp<NP; ++igp) {
                for (int jgp=0; jgp<NP; ++jgp) {
                  for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
                    Real ad = qdp_dp_h(ie,tl,it,igp,jgp,k).fastAccessDx(0);
                    Real fd = (qdp_pert_h(ie,tl,it,igp,jgp,k)-qdp_base_h(ie,tl,it,igp,jgp,k))/factor;
                    CHECK(fd == Approx(ad).margin(1e-5));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  c.finalize_singleton();
}

// Compute Dx/Dp two ways, and compare:
//  - Use ST=DpFadType to compute Dx_new/Dp from Dx_old/Dp
//  - Use product rule: Dx_new/Dp = Dx_new/Dx_old * Dx_old/Dp
TEST_CASE("remap_jv_testing") {
  using DxFadType = DxFadTypeRemap;

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
  auto& comm = c.get<ekat::Comm>();

  if (comm.am_i_root()) {
    std::cout << "Running remap_jv_testing..." << std::endl;
  }

  // Init connectivity
  auto& conn = c.create<Connectivity>();
  conn.set_comm(c.get<ekat::Comm>());
  conn.set_num_elements(4);
  conn.finalize();
  const int num_elems = c.get<Connectivity>().get_num_local_elements();

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.params_set = true;
  params.qsize = std::max(int(QSIZE_D-1),0);
  params.rsplit = 3;
  params.theta_hydrostatic_mode = false;

  // Create and init hvcoord and ref_elem
  auto& hvcoord = c.create<HybridVCoord>();
  hvcoord.random_init(seed);
  const auto max_pressure = 1000.0 + hvcoord.ps0; // This ensures max_p > ps0

  // Create elements geometry
  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0);
  geo.randomize(seed);

  // Create elements with DpFadType scalar type
  auto& elems_dp = c.create<ElementsST<DpFadType>>();
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dp.m_geometry = geo; // Use same views for geometry

  // Create elements with DxFadType scalar type
  auto& elems_dx = c.create<ElementsST<DxFadType>>();
  elems_dx.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems_dx.m_geometry = geo; // Use same views for geometry

  // Create tracers
  auto& tracers_dp = c.create<TracersST<DpFadType>>();
  auto& tracers_dx = c.create<TracersST<DxFadType>>();
  tracers_dp.init(num_elems, params.qsize);
  tracers_dx.init(num_elems, params.qsize);
  tracers_dp.randomize(seed, 1.0, 10.0);
  tracers_dx.import_values(tracers_dp, 0);

  // Lambda to compute min of dp3d, to give meaningful initial value to derived.m_eta_dot_dpdn
  auto dp3d_min = [] (decltype(elems_dp.m_state.m_dp3d) dp3d) -> Real {
    Real the_min = std::numeric_limits<Real>::max();
    for (int ie = 0; ie < dp3d.extent_int(0); ++ie) {
      // Because this constraint is difficult to satisfy for all of the tensors,
      // incrementally generate the view
      for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            auto pt_dp3d =
                Homme::subview(dp3d, ie, tl, igp, jgp);
            auto h_dp3d = Kokkos::create_mirror_view(pt_dp3d);
            Kokkos::deep_copy(h_dp3d,pt_dp3d);
            for (int ilev=0; ilev<NUM_LEV; ++ilev) {
              for (int iv=0; iv<VECTOR_SIZE; ++iv) {
                the_min = std::min(the_min,ADValue(h_dp3d(ilev)[iv]));
              }
            }
          }
        }
      }
    }
    return the_min;
  };

  // Use a simple remap algorithm
  using RemapAlgTypeDP = Remap::Ppm::PpmVertRemap<Remap::Ppm::PpmMirrored<DpFadType>>;
  using RemapAlgTypeDX = Remap::Ppm::PpmVertRemap<Remap::Ppm::PpmMirrored<DxFadType>>;

  auto rtol = 1e-10;
  auto atol = 1e-10;

  auto& catch_capture = Catch::getResultCapture();

  Real dt = RPDF(1.0,10.0)(engine);
  auto mpi_comm = Context::singleton().get<ekat::Comm>().mpi_comm();
  MPI_Bcast(&dt,1,MPI_DOUBLE,0,mpi_comm);

  const int  nm1 = 0;
  const int  n0  = 1;
  const int  np1 = 2;
  const int  np1_qdp = 0;

  // Randomize state, set derived stuff to 0
  elems_dp.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,geo.m_phis);
  elems_dp.m_derived.randomize(seed,dp3d_min(elems_dp.m_state.m_dp3d));

  // We initialize also dp derivs of elems_dp at slice tl.np1 with random values
  elems_dp.m_state.randomize_derivs(seed,np1);

  // Extract derivs into a non-fad type for the Dp = Dx*Dp_old test
  auto dxdp0 = elems_dp.m_state.take_deriv_snapshot(np1,0);

  // Init d/dx state
  for (int tl=0; tl<NUM_TIME_LEVELS; ++tl) {
    elems_dx.m_state.import_values(elems_dp.m_state,tl);
  }
  elems_dx.m_derived.import_values(elems_dp.m_derived);

  // Create the remap functors
  Remap::RemapFunctor<true, RemapAlgTypeDP, DpFadType> remap_dp(
      1, elems_dp, tracers_dp, hvcoord, -1);
  Remap::RemapFunctor<true, RemapAlgTypeDX, DxFadType> remap_dx(
      1, elems_dx, tracers_dx, hvcoord, -1);

  FunctorsBuffersManager fbm;
  fbm.request_size( remap_dp.requested_buffer_size() );
  fbm.request_size( remap_dx.requested_buffer_size() );
  fbm.allocate();
  remap_dp.init_buffers(fbm);
  remap_dx.init_buffers(fbm);

  // RUN Remap for ST=DpFadType
  remap_dp.run_remap(np1, np1_qdp, dt);

  // RUN remap's J*V functor for ST=DxFadType
  remap_dx.init_J(np1, elems_dx.m_state);
  remap_dx.run_remap(np1, np1_qdp, dt);
  auto x = dxdp0.clone(true);
  auto y = x.clone();
  remap_dx.run_JV(np1, elems_dx.m_state, x, y);

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

  auto v_JV   = ekat::scalarize(y.v);
  auto vth_JV = ekat::scalarize(y.vtheta_dp);
  auto dp_JV  = ekat::scalarize(y.dp3d);
  auto phi_JV = ekat::scalarize(y.phinh_i);
  auto w_JV   = ekat::scalarize(y.w_i);
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
          auto du_src = v_JV_h(ie,0,igp,jgp,k), du_tgt = v_dp_h(ie,np1,0,igp,jgp,k).dx(0);
          CHECK_THAT (du_src, Catch::WithinRel(du_tgt,rtol) || Catch::WithinAbs(du_tgt,atol));

          auto dv_src = v_JV_h(ie,1,igp,jgp,k), dv_tgt = v_dp_h(ie,np1,1,igp,jgp,k).dx(0);
          CHECK_THAT (dv_src, Catch::WithinRel(dv_tgt,rtol) || Catch::WithinAbs(dv_tgt,atol));

          auto dvth_src = vth_JV_h(ie,igp,jgp,k), dvth_tgt = vth_dp_h(ie,np1,igp,jgp,k).dx(0);
          CHECK_THAT (dvth_src, Catch::WithinRel(dvth_tgt,rtol) || Catch::WithinAbs(dvth_tgt,atol));

          auto ddp_src = dp_JV_h(ie,igp,jgp,k), ddp_tgt = dp_dp_h(ie,np1,igp,jgp,k).dx(0);
          CHECK_THAT (ddp_src, Catch::WithinRel(ddp_tgt,rtol) || Catch::WithinAbs(ddp_tgt,atol));

          auto dphi_src = phi_JV_h(ie,igp,jgp,k), dphi_tgt = phi_dp_h(ie,np1,igp,jgp,k).dx(0);
          CHECK_THAT (dphi_src, Catch::WithinRel(dphi_tgt,rtol) || Catch::WithinAbs(dphi_tgt,atol));

          auto dw_src = w_JV_h(ie,igp,jgp,k), dw_tgt = w_dp_h(ie,np1,igp,jgp,k).dx(0);
          CHECK_THAT (dw_src, Catch::WithinRel(dw_tgt,rtol) || Catch::WithinAbs(dw_tgt,atol));
        }
        int k = NUM_PHYSICAL_LEV;

        auto dphi_src = phi_JV_h(ie,igp,jgp,k), dphi_tgt = phi_dp_h(ie,np1,igp,jgp,k).dx(0);
        CHECK_THAT (dphi_src, Catch::WithinRel(dphi_tgt,rtol) || Catch::WithinAbs(dphi_tgt,atol));

        auto dw_src = w_JV_h(ie,igp,jgp,k), dw_tgt = w_dp_h(ie,np1,igp,jgp,k).dx(0);
        CHECK_THAT (dw_src, Catch::WithinRel(dw_tgt,rtol) || Catch::WithinAbs(dw_tgt,atol));
      }
    }
  }
  c.finalize_singleton();
}

// Verify the JtV transpose identity: <a, J*b> == <J^T*a, b>.
TEST_CASE("remap_jtv_testing") {
  using DxFadType = DxFadTypeRemap;

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  // Use stuff from Context, to increase similarity with actual runs
  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);
  auto& comm = c.get<ekat::Comm>();

  if (comm.am_i_root()) {
    std::cout << "Running remap_jtv_testing..." << std::endl;
  }

  // Init connectivity
  auto& conn = c.create<Connectivity>();
  conn.set_comm(c.get<ekat::Comm>());
  conn.set_num_elements(4);
  conn.finalize();
  const int num_elems = c.get<Connectivity>().get_num_local_elements();

  // Init parameters
  auto& params = c.create<SimulationParams>();
  params.params_set = true;
  params.qsize = std::max(int(QSIZE_D-1),0);
  params.rsplit = 3;
  params.theta_hydrostatic_mode = false;

  const int np1 = 2;
  const Real dt = RPDF(1.0,10.0)(engine);

  // Create and init hvcoord and ref_elem
  auto& hvcoord = c.create<HybridVCoord>();
  hvcoord.random_init(seed);
  const auto max_pressure = 1000.0 + hvcoord.ps0; // This ensures max_p > ps0

  // Create elements geometry
  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0);
  geo.randomize(seed);

  // Create elements
  ElementsST<DxFadType> elems;
  elems.init(num_elems,false,true,PhysicalConstants::rearth0);
  elems.m_geometry = geo; // Use same views for geometry

  // Random snapshots
  StateSnapshot xa(num_elems), xb(num_elems);
  xa.randomize(seed, max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, geo.m_phis);
  xb.randomize(seed+1, max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, geo.m_phis);
  auto ya = xa.clone();
  auto yb = xb.clone();

  // Setup remap algorithm
  using RemapAlgTypeDX = Remap::Ppm::PpmVertRemap<Remap::Ppm::PpmMirrored<DxFadType>>;
  TracersST<DxFadType> tracers;
  tracers.init(num_elems, params.qsize);
  Remap::RemapFunctor<true, RemapAlgTypeDX, DxFadType> remap(1, elems, tracers, hvcoord, -1);

  FunctorsBuffersManager fbm;
  fbm.request_size(remap.requested_buffer_size());
  fbm.allocate();
  remap.init_buffers(fbm);

  // Randomize element state
  elems.m_state.randomize(seed+2, max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, geo.m_phis);

  // Assemble Jacobian
  remap.init_J(np1, elems.m_state);
  remap.run_remap(np1, 0, dt);
  Kokkos::fence();

  // Compute ya = J*xa
  remap.run_JV(np1, elems.m_state, xa, ya);

  // compute yb = J^T*xb
  remap.run_JtV(np1, elems.m_state, xb, yb);
  Kokkos::fence();

  //
  // Check that xb^T*ya == yb^T*xa
  //

  // Scalarize views for reductions
  auto xa_v   = ekat::scalarize(xa.v);
  auto xa_vth = ekat::scalarize(xa.vtheta_dp);
  auto xa_dp3d= ekat::scalarize(xa.dp3d);
  auto xa_phi = ekat::scalarize(xa.phinh_i);
  auto xa_w   = ekat::scalarize(xa.w_i);

  auto xb_v   = ekat::scalarize(xb.v);
  auto xb_vth = ekat::scalarize(xb.vtheta_dp);
  auto xb_dp3d= ekat::scalarize(xb.dp3d);
  auto xb_phi = ekat::scalarize(xb.phinh_i);
  auto xb_w   = ekat::scalarize(xb.w_i);

  auto ya_v   = ekat::scalarize(ya.v);
  auto ya_vth = ekat::scalarize(ya.vtheta_dp);
  auto ya_dp3d= ekat::scalarize(ya.dp3d);
  auto ya_phi = ekat::scalarize(ya.phinh_i);
  auto ya_w   = ekat::scalarize(ya.w_i);

  auto yb_v   = ekat::scalarize(yb.v);
  auto yb_vth = ekat::scalarize(yb.vtheta_dp);
  auto yb_dp3d= ekat::scalarize(yb.dp3d);
  auto yb_phi = ekat::scalarize(yb.phinh_i);
  auto yb_w   = ekat::scalarize(yb.w_i);

  using p5_mid_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>;
  using p4_mid_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
  using p4_int_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
  const p5_mid_t p5_mid({0,0,0,0,0}, {num_elems, 2, NP, NP, NUM_PHYSICAL_LEV});
  const p4_mid_t p4_mid({0,0,0,0}, {num_elems, NP, NP, NUM_PHYSICAL_LEV});
  const p4_int_t p4_int({0,0,0,0}, {num_elems, NP, NP, NUM_INTERFACE_LEV});

  Real2 V_dot, vth_dot, dp_dot, phi_dot, w_dot;

  Kokkos::parallel_reduce(p5_mid,
    KOKKOS_LAMBDA(int ie, int icmp, int ip, int jp, int k, Real2& acc) {
      acc.v[0] += xa_v(ie, icmp, ip, jp, k) * yb_v(ie, icmp, ip, jp, k);
      acc.v[1] += ya_v(ie, icmp, ip, jp, k) * xb_v(ie, icmp, ip, jp, k);
    }, V_dot);

  Kokkos::parallel_reduce(p4_mid,
    KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
      acc.v[0] += xa_vth(ie, ip, jp, k) * yb_vth(ie, ip, jp, k);
      acc.v[1] += ya_vth(ie, ip, jp, k) * xb_vth(ie, ip, jp, k);
    }, vth_dot);

  Kokkos::parallel_reduce(p4_mid,
    KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
      acc.v[0] += xa_dp3d(ie, ip, jp, k) * yb_dp3d(ie, ip, jp, k);
      acc.v[1] += ya_dp3d(ie, ip, jp, k) * xb_dp3d(ie, ip, jp, k);
    }, dp_dot);

  Kokkos::parallel_reduce(p4_int,
    KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
      acc.v[0] += xa_phi(ie, ip, jp, k) * yb_phi(ie, ip, jp, k);
      acc.v[1] += ya_phi(ie, ip, jp, k) * xb_phi(ie, ip, jp, k);
    }, phi_dot);

  Kokkos::parallel_reduce(p4_int,
    KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
      acc.v[0] += xa_w(ie, ip, jp, k) * yb_w(ie, ip, jp, k);
      acc.v[1] += ya_w(ie, ip, jp, k) * xb_w(ie, ip, jp, k);
    }, w_dot);

  Real2 gdot = V_dot; gdot += vth_dot; gdot += dp_dot; gdot += phi_dot; gdot += w_dot;

  const Real rtol = 1e-8;
  const Real atol = 1e-8;
  CHECK_THAT(gdot.v[0], Catch::WithinRel(gdot.v[1], rtol) || Catch::WithinAbs(gdot.v[1], atol));

  c.finalize_singleton();
}
