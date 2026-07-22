#include <catch2/catch.hpp>

#include "DirkFunctorImpl.hpp"

#include <random>

#include "Types.hpp"
#include "Context.hpp"
#include "mpi/Connectivity.hpp"
#include "SimulationParams.hpp"
#include "Elements.hpp"
#include "PhysicalConstants.hpp"

#include "utilities/TestUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/ViewUtils.hpp"

#include <ekat_string_utils.hpp>
#include <ekat_test_utils.hpp>

#include <iomanip>

using namespace Homme;

// A constants provider that perturbs Rgas by a factor (1+perturb).
// kappa = Rgas/cp is also perturbed as a result.
template<typename ST>
struct PerturbedConstants : public PhysicalConstantsProvider {
  using PC = PhysicalConstantsProvider;

  PerturbedConstants() = default;

  ST perturb = 0;

  // Only redefine the ones we want to perturb (the rest is inherited)
  KOKKOS_FORCEINLINE_FUNCTION ST Rgas  () const { return (1+perturb)*PC::Rgas(); }
  KOKKOS_FORCEINLINE_FUNCTION ST kappa () const { return Rgas() / cp();    }
};

TEST_CASE ("dirk_dp_testing")
{
  // Setup random number generator
  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");

  using rpdf = std::uniform_real_distribution<Real>;
  using ipdf = std::uniform_int_distribution<int>;
  using rngAlg = std::mt19937_64;

  rngAlg engine(seed);

  // Create data structures
  Context::finalize_singleton();
  Context::singleton().create<ekat::Comm>(MPI_COMM_WORLD);

  const auto ALL = Kokkos::ALL;
  const int ntl = NUM_TIME_LEVELS;
  const int nlev = NUM_PHYSICAL_LEV;
  const int np = NP;

  auto& ts = ekat::TestSession::get();

  int num_elems = ipdf(5,50)(engine);
  if (ts.params.count("ne")>0) {
    num_elems = std::stoi(ts.params["ne"]);
  }
  int nm1 = 0;
  int n0  = 1;
  int np1 = 2;

  HybridVCoord hvcoord;
  hvcoord.random_init(seed);
  const auto max_pressure = 1000 + hvcoord.ps0;

  ElementsST<Real> elems_h0, elems_h, elems_ref;
  ElementsST<DpFadType> elems_dp;
  elems_h0.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_h.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_ref.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);

  auto& geometry = elems_h0.m_geometry = elems_h.m_geometry = elems_dp.m_geometry = elems_ref.m_geometry;
  Context::singleton().create_ref(elems_h0.m_geometry);
  geometry.randomize(seed);

  elems_ref.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,geometry.m_phis);

  // Create Perturbed constants and Dirk instances
  using PerturbedConstantsReal = PerturbedConstants<Real>;
  using PerturbedConstantsFad  = PerturbedConstants<DpFadType>;
  using DirkReal = DirkFunctorImplST<Real,PerturbedConstantsReal>;
  using DirkFad  = DirkFunctorImplST<DpFadType,PerturbedConstantsFad>;

  // EquationOfState<PerturbedConstantsReal> eos_real;
  // EquationOfState<PerturbedConstantsFad>  eos_fad;

  DirkReal dirk_real(num_elems);
  DirkFad  dirk_dp(num_elems);
  dirk_real.m_verbose = false;
  dirk_dp.m_verbose = false;
  dirk_dp.m_eos.m_constants.perturb.fastAccessDx(0) = 1;

  FunctorsBuffersManager fbm;
  fbm.request_size(dirk_real.requested_buffer_size());
  fbm.request_size(dirk_dp.requested_buffer_size());
  fbm.allocate();
  dirk_real.init_buffers(fbm);
  dirk_dp.init_buffers(fbm);

  // Run for different configs and check convergence as perturb->0
  const std::vector<Real> h_vals = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};
  Real dt2 = rpdf(0.1,0.9)(engine);
  const auto rtol = 1e-4;
  const auto atol = 1e-6;

  SECTION ("initial_guess") {
    // Run dirk_dp
    elems_dp.m_state.import_values(elems_ref.m_state,np1);
    dirk_dp.run_initial_guess(np1,elems_dp,hvcoord);
    auto dphi = ekat::scalarize(elems_dp.m_derived.m_divdp_proj);

    // Run dirk_real with no perturbation
    dirk_real.m_eos.m_constants.perturb = 0;
    elems_h0.m_state.import_values(elems_ref.m_state,np1);
    dirk_real.run_initial_guess(np1,elems_h0,hvcoord);
    auto phi0 = ekat::scalarize(elems_h0.m_derived.m_divdp_proj);

    Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> policy({0,0,0,0},{num_elems,np,np,nlev});
    // Run perturbed dirk_real
    std::vector<Real> eh;
    for (auto h : h_vals) {
      dirk_real.m_eos.m_constants.perturb = h;
      elems_h.m_state.import_values(elems_ref.m_state,np1);
      dirk_real.run_initial_guess(np1,elems_h,hvcoord);

      // Compute error on initial guess deriv
      auto phih = ekat::scalarize(elems_h.m_derived.m_divdp_proj);

      auto linf = KOKKOS_LAMBDA(int ie, int ip, int jp, int ilev, Real& accum) {
        Real dphi_fd = (phih(ie,ip,jp,ilev) - phi0(ie,ip,jp,ilev)) / h;
        Real dphi_ex = dphi(ie,ip,jp,ilev).fastAccessDx(0);
        Real lcl_err = Kokkos::abs(dphi_fd-dphi_ex);
        if (lcl_err > accum) accum = lcl_err;
      };
      Kokkos::parallel_reduce(policy,linf,Kokkos::Max<Real>(eh.emplace_back(0)));
    }
    std::cout << std::setprecision(16) << "      h = [" << ekat::join(h_vals,",") << "]\n";
    std::cout << std::setprecision(16) << "      |dv/dp - FD(dvdp)|_inf = [" << ekat::join(eh,",") << "]\n";
    const Real min_err = *std::min_element(eh.begin(),eh.end());
    REQUIRE ((eh[0]<atol or min_err<eh[0]*rtol));
  }

  SECTION ("run") {
    const bool bfb_solver = false;
    Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> policy({0,0,0,0},{num_elems,np,np,nlev+1});
    for (Real alphadt_nm1 : {0.0, 0.3}) {
      nm1 = alphadt_nm1 == 0.0 ? -1 : 0;
      printf("-> alphadt_nm1: %f\n", alphadt_nm1);
      for (Real alphadt_n0 : {0.0, 0.7}) {
        printf("  -> alphadt_n0: %f\n", alphadt_n0);

        // Run dirk_dp
        elems_dp.m_state.import_values(elems_ref.m_state,0);
        elems_dp.m_state.import_values(elems_ref.m_state,1);
        elems_dp.m_state.import_values(elems_ref.m_state,2);
        dirk_dp.run(nm1,alphadt_nm1,n0,alphadt_n0,np1,dt2,elems_dp,hvcoord,bfb_solver);
        auto dphi = ekat::scalarize(elems_dp.m_state.m_phinh_i);

        // Run dirk_real with no perturbation
        dirk_real.m_eos.m_constants.perturb = 0;
        elems_h0.m_state.import_values(elems_ref.m_state,0);
        elems_h0.m_state.import_values(elems_ref.m_state,1);
        elems_h0.m_state.import_values(elems_ref.m_state,2);
        dirk_real.run(nm1,alphadt_nm1,n0,alphadt_n0,np1,dt2,elems_h0,hvcoord,bfb_solver);
        auto phi0 = ekat::scalarize(elems_h0.m_state.m_phinh_i);

        // Run perturbed dirk_real
        std::vector<Real> eh;
        for (auto h : h_vals) {
          dirk_real.m_eos.m_constants.perturb = h;
          elems_h.m_state.import_values(elems_ref.m_state,0);
          elems_h.m_state.import_values(elems_ref.m_state,1);
          elems_h.m_state.import_values(elems_ref.m_state,2);
          dirk_real.run(nm1,alphadt_nm1,n0,alphadt_n0,np1,dt2,elems_h,hvcoord,bfb_solver);
          auto phih = ekat::scalarize(elems_h.m_state.m_phinh_i);

          // Compute error on phi(np1)
          auto linf = KOKKOS_LAMBDA(int ie, int ip, int jp, int ilev, Real& accum) {
            Real dphi_fd = (phih(ie,np1,ip,jp,ilev) - phi0(ie,np1,ip,jp,ilev)) / h;
            Real dphi_ex = dphi(ie,np1,ip,jp,ilev).fastAccessDx(0);
            Real lcl_err = Kokkos::abs(dphi_fd-dphi_ex);
            if (lcl_err > accum) accum = lcl_err;
          };
          Kokkos::parallel_reduce(policy,linf,Kokkos::Max<Real>(eh.emplace_back(0)));
        }
        std::cout << std::setprecision(16) << "      h = [" << ekat::join(h_vals,",") << "]\n";
        std::cout << std::setprecision(16) << "      |dv/dp - FD(dvdp)|_inf = [" << ekat::join(eh,",") << "]\n";
        const Real min_err = *std::min_element(eh.begin(),eh.end());
        REQUIRE ((eh[0]<atol or min_err<eh[0]*rtol));
      }
    }
  }

  Context::finalize_singleton();
}

TEST_CASE ("dirk_jv_testing") {
  // Verify the chain rule: dX_np1/dp = J * (dX_n0/dp)
  //  - LHS: computed via DpFadType by running DIRK with FAD arithmetic over p
  //  - RHS: computed via init_J + run + run_JV (DxFadTypeDirk)
  // This checks that J = d(state_np1) / d(state_n0) is correctly assembled.

  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed == 0 ? " (catch rng seed was 0)\n" : "\n");

  using ipdf = std::uniform_int_distribution<int>;
  using rpdf = std::uniform_real_distribution<Real>;
  using rngAlg = std::mt19937_64;

  rngAlg engine(seed);

  Context::finalize_singleton();
  Context::singleton().create<ekat::Comm>(MPI_COMM_WORLD);

  const int nlev = NUM_PHYSICAL_LEV;
  const int np   = NP;
  const auto rtol = 1e-4;
  const auto atol = 1e-6;
  const int num_elems = ipdf(5,20)(engine);
  int nm1 = 0;
  int n0  = 1;
  int np1 = 2;

  HybridVCoord hvcoord;
  hvcoord.random_init(seed);
  const auto max_pressure = 1000 + hvcoord.ps0;

  using DxFadType = DxFadTypeDirk;
  using DirkDp = DirkFunctorImplST<DpFadType>;
  using DirkDx = DirkFunctorImplST<DxFadType>;

  // elems_dp: DpFadType, used to compute dX_np1/dp directly via FAD DIRK run
  // elems_dx: DxFadTypeDirk, used to compute J via init_J + run
  ElementsST<DpFadType> elems_dp;
  ElementsST<DxFadType> elems_dx;
  elems_dp.init(num_elems, false, true, PhysicalConstants::rearth0, -1, true);
  elems_dx.init(num_elems, false, true, PhysicalConstants::rearth0, -1, true);

  auto& geometry = elems_dp.m_geometry = elems_dx.m_geometry;
  Context::singleton().create_ref(geometry);
  geometry.randomize(seed);

  // Randomize state values (real part) for all time levels
  elems_dp.m_state.randomize(seed, max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, geometry.m_phis);

  // Randomize FAD derivatives at n0: these represent dX_n0/dp for a random scalar p
  elems_dp.m_state.randomize_derivs(seed, n0);

  // Initialize DxFadType state with the same real values as DpFadType state
  for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl)
    elems_dx.m_state.import_values(elems_dp.m_state, tl);

  // Create DIRK functors
  DirkDp dirk_dp(num_elems);
  DirkDx dirk_dx(num_elems);
  dirk_dp.m_verbose = false;
  dirk_dx.m_verbose = false;

  FunctorsBuffersManager fbm;
  fbm.request_size(dirk_dp.requested_buffer_size());
  fbm.request_size(dirk_dx.requested_buffer_size());
  fbm.allocate();
  dirk_dp.init_buffers(fbm);
  dirk_dx.init_buffers(fbm);

  const Real dt2 = rpdf(0.1, 0.9)(engine);
  const bool bfb_solver = false;

  for (Real alphadt_nm1 : {0.0, 0.3}) {
    printf("-> alphadt_nm1: %f\n", alphadt_nm1);
    for (Real alphadt_n0 : {0.0, 0.7}) {
      printf("  -> alphadt_n0: %f\n", alphadt_n0);

      // Step 1: run DpFadType DIRK to compute dX_np1/dp
      dirk_dp.run(nm1, alphadt_nm1, n0, alphadt_n0, np1, dt2,
                  elems_dp, hvcoord, bfb_solver);
      Kokkos::fence();

      // Step 2: run DxFadTypeDirk DIRK:
      //   init_J  - set d(X_n0[k])/d(X_n0[k]) = 1 per state variable per level k per column
      //   run     - Newton solve with FAD arithmetic, propagating J through the solve
      //   run_JV  - apply product rule: (J at np1) * (V at n0) -> result at np1
      dirk_dx.init_J(n0, elems_dx.m_state);
      dirk_dx.run(nm1, alphadt_nm1, n0, alphadt_n0, np1, dt2,
                  elems_dx, hvcoord, bfb_solver);
      Kokkos::fence();
      auto x = elems_dp.m_state.take_snapshot(n0);
      auto y = x.clone(); 
      dirk_dx.run_JV(np1, elems_dx.m_state, x, y);
      Kokkos::fence();

      // Compare: elems.m_state at np1 (= J*V) vs. elems_dp.m_state.dx(0) at np1 (= dX_np1/dp)
      // DIRK only updates w_i and phinh_i; check both interface-level fields.
      auto dphi_dp = ekat::scalarize(elems_dp.m_state.m_phinh_i);
      auto dw_dp   = ekat::scalarize(elems_dp.m_state.m_w_i);
      auto dphi_jv = ekat::scalarize(y.phinh_i);
      auto dw_jv   = ekat::scalarize(y.w_i);

      const auto dphi_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dphi_dp);
      const auto dw_dp_h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dw_dp);
      const auto dphi_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dphi_jv);
      const auto dw_jv_h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dw_jv);

      for (int ie = 0; ie < num_elems; ++ie)
        for (int ip = 0; ip < np; ++ip)
          for (int jp = 0; jp < np; ++jp)
            for (int k = 0; k <= nlev; ++k) {
              auto dphi_src = dphi_jv_h(ie, ip, jp, k);
              auto dphi_tgt = dphi_dp_h(ie, np1, ip, jp, k).dx(0);
              CHECK_THAT(dphi_src, Catch::WithinRel(dphi_tgt, rtol) || Catch::WithinAbs(dphi_tgt, atol));

              auto dw_src = dw_jv_h(ie, ip, jp, k);
              auto dw_tgt = dw_dp_h(ie, np1, ip, jp, k).dx(0);
              CHECK_THAT(dw_src, Catch::WithinRel(dw_tgt, rtol) || Catch::WithinAbs(dw_tgt, atol));
            }
    }
  }

  Context::finalize_singleton();
}

TEST_CASE ("dirk_jtv_testing") {
  // Verify the transpose identity: <a[n0], J*b[np1]> == <J^T*a[np1], b[n0]>
  // where J = d(state_np1) / d(state_n0) is assembled via DxFadTypeDirk.
  // J*b   is computed via run_JV  and stored at np1 in adj_b.
  // J^T*a is computed via run_JtV and stored at np1 in adj_a.

  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  unsigned int seed = catchRngSeed == 0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed == 0 ? " (catch rng seed was 0)\n" : "\n");

  using ipdf = std::uniform_int_distribution<int>;
  using rpdf = std::uniform_real_distribution<Real>;
  using rngAlg = std::mt19937_64;

  rngAlg engine(seed);

  Context::finalize_singleton();
  Context::singleton().create<ekat::Comm>(MPI_COMM_WORLD);

  const Real rtol = 1e-8;
  const Real atol = 1e-8;
  const int num_elems = ipdf(5, 20)(engine);
  int nm1 = 0;
  int n0  = 1;
  int np1 = 2;

  HybridVCoord hvcoord;
  hvcoord.random_init(seed++);
  const auto max_pressure = 1000 + hvcoord.ps0;

  using DxFadType = DxFadTypeDirk;
  using DirkDx = DirkFunctorImplST<DxFadType>;

  ElementsST<DxFadType> elems_dx;
  elems_dx.init(num_elems, false, true, PhysicalConstants::rearth0, -1, true);

  auto& geometry = elems_dx.m_geometry;
  Context::singleton().create_ref(geometry);
  geometry.randomize(seed++);

  // Two adjoint vectors (Real), both initialized to the same random state at n0.
  // run_JV / run_JtV only read n0 and write np1, so n0 remains unchanged throughout.
  StateSnapshot xa(num_elems), xb(num_elems);
  xa.randomize(seed++, max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, geometry.m_phis);
  xb.randomize(seed++, max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, geometry.m_phis);

  auto ya = xa.clone();
  auto yb = xb.clone();

  DirkDx dirk_dx(num_elems);
  dirk_dx.m_verbose = false;

  FunctorsBuffersManager fbm;
  fbm.request_size(dirk_dx.requested_buffer_size());
  fbm.allocate();
  dirk_dx.init_buffers(fbm);

  const Real dt2 = rpdf(0.1, 0.9)(engine);
  const bool bfb_solver = false;

  using p4_mid_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
  using p5_mid_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<5>>;
  using p4_int_t = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>;
  const p4_mid_t p4_mid({0,0,0,0}, {num_elems, NP, NP, NUM_PHYSICAL_LEV});
  const p5_mid_t p5_mid({0,0,0,0,0}, {num_elems, 2, NP, NP, NUM_PHYSICAL_LEV});
  const p4_int_t p4_int({0,0,0,0}, {num_elems, NP, NP, NUM_INTERFACE_LEV});

  auto xa_v   = ekat::scalarize(xa.v);
  auto xa_vth = ekat::scalarize(xa.vtheta_dp);
  auto xa_dp  = ekat::scalarize(xa.dp3d);
  auto xa_phi = ekat::scalarize(xa.phinh_i);
  auto xa_w   = ekat::scalarize(xa.w_i);

  auto ya_v   = ekat::scalarize(ya.v);
  auto ya_vth = ekat::scalarize(ya.vtheta_dp);
  auto ya_dp  = ekat::scalarize(ya.dp3d);
  auto ya_phi = ekat::scalarize(ya.phinh_i);
  auto ya_w   = ekat::scalarize(ya.w_i);

  auto xb_v   = ekat::scalarize(xb.v);
  auto xb_vth = ekat::scalarize(xb.vtheta_dp);
  auto xb_dp  = ekat::scalarize(xb.dp3d);
  auto xb_phi = ekat::scalarize(xb.phinh_i);
  auto xb_w   = ekat::scalarize(xb.w_i);

  auto yb_v   = ekat::scalarize(yb.v);
  auto yb_vth = ekat::scalarize(yb.vtheta_dp);
  auto yb_dp  = ekat::scalarize(yb.dp3d);
  auto yb_phi = ekat::scalarize(yb.phinh_i);
  auto yb_w   = ekat::scalarize(yb.w_i);

  for (Real alphadt_nm1 : {0.0, 0.3}) {
    printf("-> alphadt_nm1: %f\n", alphadt_nm1);
    for (Real alphadt_n0 : {0.0, 0.7}) {
      printf("  -> alphadt_n0: %f\n", alphadt_n0);

      // Randomize the DxFad state for this run
      elems_dx.m_state.randomize(seed, max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0, geometry.m_phis);

      // Initialize Jacobian d/dx(np1) and run DIRK with FAD arithmetic
      dirk_dx.init_J(np1, elems_dx.m_state);
      dirk_dx.run(nm1, alphadt_nm1, n0, alphadt_n0, np1, dt2, elems_dx, hvcoord, bfb_solver);
      Kokkos::fence();

      // J*b  -> adj_b[np1]
      dirk_dx.run_JV (np1, elems_dx.m_state, xa, ya);
      // J^T*a -> adj_a[np1]
      dirk_dx.run_JtV(np1, elems_dx.m_state, xb, yb);
      Kokkos::fence();

      // (x_b,y_a) = (x_b,J*x_a) = (J^T*x_b,x_a) = (y_b,x_a)
      Real2 V_dot, vth_dot, dp_dot, phi_dot, w_dot;
      Real2 gdot;

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
          acc.v[0] += xa_dp(ie, ip, jp, k) * yb_dp(ie, ip, jp, k);
          acc.v[1] += ya_dp(ie, ip, jp, k) * xb_dp(ie, ip, jp, k);
        }, dp_dot);
      Kokkos::parallel_reduce(p4_int,
        KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
          acc.v[0] += xa_phi(ie, ip, jp, k) * yb_phi(ie, ip, jp, k);
          acc.v[1] += ya_phi(ie, ip, jp, k) * xb_phi(ie, ip, jp, k);
        }, phi_dot);
      Kokkos::parallel_reduce(p4_int,
        KOKKOS_LAMBDA(int ie, int ip, int jp, int k, Real2& acc) {
          acc.v[0] += xa_w(ie, ip, jp, k) * xb_w(ie, ip, jp, k);
          acc.v[1] += ya_w(ie, ip, jp, k) * yb_w(ie, ip, jp, k);
        }, w_dot);

      gdot += V_dot;
      gdot += vth_dot;
      gdot += dp_dot;
      gdot += phi_dot;
      gdot += w_dot;

      CHECK_THAT(gdot.v[0], Catch::WithinRel(gdot.v[1], rtol) || Catch::WithinAbs(gdot.v[1], atol));

      std::cout << std::setprecision(15)
                << "       <a, J*b> = " << gdot.v[0]
                << ",  <J^T*a, b> = " << gdot.v[1] << "\n";
    }
  }

  Context::finalize_singleton();
}
