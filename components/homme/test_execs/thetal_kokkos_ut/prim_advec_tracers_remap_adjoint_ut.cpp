#include <catch2/catch.hpp>

#include "Types.hpp"
#include "thetal_ut_session.hpp"
#include "thetal_ut_utils.hpp"
#include "Context.hpp"
#include "ElementsDerivedState.hpp"
#include "ElementsGeometry.hpp"
#include "EulerStepFunctor.hpp"
#include "EulerStepFunctorImpl.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HybridVCoord.hpp"
#include "PhysicalConstants.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "TimeLevel.hpp"
#include "Tracers.hpp"
#include "prim_advec_tracers_remap_adj.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_comm.hpp>
#include <ekat_pack_kokkos.hpp>

#include <cmath>
#include <iomanip>
#include <random>
#include <vector>

#ifdef HOMMEXX_ENABLE_FAD_TYPES

namespace Homme
{

// Adjoint test for the whole tracers step of prim_step (prim_advec_tracers_remap, RK2):
// precompute_divdp, 3 chained euler_steps (rhs_multiplier=0,1,2, so the qlim chaining
// and the biharmonic term are exercised too), and qdp_time_avg. Same dot-product pattern as
// euler_step_adjoint_ut.cpp:
//   <J*du0, lambdaN> == <du0, J^T*lambdaN>
// with the tangent J*du0 obtained from a Fad-typed (DpFadType) replica of
// prim_advec_tracers_remap. The inputs are qdp, and the derived vn0 and dp (the only
// quantities the tracers step takes from the dynamics); all the other derived fields the
// fwd uses are either computed from them (divdp, divdp_proj) or not consumed (eta_dot_dpdn,
// omega_p are only DSS'd).
TEST_CASE("prim_advec_tracers_remap_adjoint")
{
  constexpr int ne = 2;
  constexpr int qsize = 2;
  constexpr int nlevs = NUM_PHYSICAL_LEV;

  std::random_device rd;
  using rngAlg = std::mt19937_64;
  using RPDF = std::uniform_real_distribution<Real>;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);

  auto& c = Context::singleton();
  auto& comm = c.create<ekat::Comm>(MPI_COMM_WORLD);

  auto& params = c.create<SimulationParams>();
  params.params_set = true;
  params.qsize = qsize;
  params.transport_alg = 0;
  params.limiter_option = 9;
  params.nu_p = 0;
  // Set below, by looping over a few values (huge ones too: on this tiny grid, with ne=2 and
  // dp0 = ps0*delta(eta) with ps0=1 in the random hvcoord, it takes a huge viscosity for the
  // biharmonic term of the last euler_step, the only one with that term, to matter).
  params.nu_q = 0;
  params.hypervis_scaling = 0.0;
  params.scale_factor = PhysicalConstants::rearth0;
  params.laplacian_rigid_factor = PhysicalConstants::rrearth0;
  params.dt_tracer_factor = 2;

  // Create the f90 mesh/connectivity, and init hvcoord and ref_FE.
  // The session also takes care of cleaning up (f90 and Context) at the end of the scope.
  ThetalUnitTestSession session(ne,seed);
  auto& hvcoord = session.hvcoord;
  auto& ref_FE  = session.ref_FE;
  (void) hvcoord;

  const int num_elems = session.num_elems();

  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  session.init_geometry(geo);

  auto& bmm = c.create<MpiBuffersManagerMap>();
  auto conn_ptr = c.get_ptr<Connectivity>();
  bmm[MPI_EXCHANGE]->set_connectivity(conn_ptr);
  bmm[MPI_EXCHANGE_MIN_MAX]->set_connectivity(conn_ptr);

  // ---- Real-typed stack: the adjoint half. ----
  auto& tracers_r = c.create<TracersST<Real>>();
  tracers_r.init(num_elems,qsize);
  auto& derived_r = c.create<ElementsDerivedStateST<Real>>();
  derived_r.init(num_elems);
  auto& sphops_r = c.create<SphereOperatorsST<Real>>();
  sphops_r.setup(geo,ref_FE);

  EulerStepFunctorST<Real> euler_r(num_elems);
  euler_r.setup();
  euler_r.reset(params);
  FunctorsBuffersManager fbm_r;
  fbm_r.request_size(euler_r.requested_buffer_size());
  fbm_r.allocate();
  euler_r.init_buffers(fbm_r);
  euler_r.init_boundary_exchanges();

  // ---- DpFadType-typed stack: the fwd tangent half. ----
  auto& tracers_dp = c.create<TracersST<DpFadType>>();
  tracers_dp.init(num_elems,qsize);
  auto& derived_dp = c.create<ElementsDerivedStateST<DpFadType>>();
  derived_dp.init(num_elems);
  auto& sphops_dp = c.create<SphereOperatorsST<DpFadType>>();
  sphops_dp.setup(geo,ref_FE);

  EulerStepFunctorST<DpFadType> euler_dp(num_elems);
  euler_dp.setup();
  euler_dp.reset(params);
  FunctorsBuffersManager fbm_dp;
  fbm_dp.request_size(euler_dp.requested_buffer_size());
  fbm_dp.allocate();
  euler_dp.init_buffers(fbm_dp);
  euler_dp.init_boundary_exchanges();

  // Time levels: start at nstep=2 so that (with dt_tracer_factor=2) the tracers use
  // time levels (n0_qdp,np1_qdp)=(1,0)
  auto& tl = c.create<TimeLevel>();
  tl.nm1 = 0;
  tl.n0  = 1;
  tl.np1 = 2;
  tl.nstep = 2;
  tl.nstep0 = 0;
  tl.tevolve = 0;
  TimeLevel tl_dp = tl;
  tl_dp.update_tracers_levels(params.dt_tracer_factor);
  const int n0_qdp  = tl_dp.n0_qdp;
  const int np1_qdp = tl_dp.np1_qdp;

  // Total tracers dt (what is passed to prim_advec_tracers_remap)
  const Real dt = 1.0;

  // Base point. Same ranges as in euler_step_adjoint_ut.cpp: dp safely positive, vn0 with
  // real magnitude so that the limiter's clip has something to clip, and qdp strictly
  // positive, to stay away from the 0-floor and mass-relaxation branches of the limiter
  // (for which euler_step_adj knowingly gives an approximate gradient).
  RPDF dp_pdf(80.0,120.0), qdp_pdf(300.0,400.0), vn0_pdf(-40.0,40.0), tiny_pdf(-1e-3,1e-3), unit_pdf(-1.0,1.0);
  genRandArray(tracers_r.qdp,               engine, qdp_pdf);
  // dp is a DSS'd (continuous) field in real runs, so use one that only depends on the level
  // (random per level), rather than a dp that jumps randomly from node to node. This also
  // makes the DSS'd inputs of the 2nd and 3rd euler steps tie with their neighbors' at shared
  // nodes (as they do in real runs), which the min/max reduction adjoint has to handle.
  {
    auto dp_h = Kokkos::create_mirror_view(derived_r.m_dp);
    auto dp_s = ekat::scalarize(dp_h);
    for (int lev=0; lev<nlevs; ++lev) {
      const Real dp_lev = dp_pdf(engine);
      for (int ie=0; ie<num_elems; ++ie)
        for (int i=0; i<NP; ++i)
          for (int j=0; j<NP; ++j)
            dp_s(ie,i,j,lev) = dp_lev;
    }
    Kokkos::deep_copy(derived_r.m_dp,dp_h);
  }
  genRandArray(derived_r.m_vn0,             engine, vn0_pdf);
  genRandArray(derived_r.m_divdp,           engine, tiny_pdf); // recomputed from vn0 by the fwd
  genRandArray(derived_r.m_divdp_proj,      engine, tiny_pdf); // ditto
  genRandArray(derived_r.m_eta_dot_dpdn,    engine, tiny_pdf);
  genRandArray(derived_r.m_omega_p,         engine, tiny_pdf);
  genRandArray(derived_r.m_dpdiss_biharmonic, engine, tiny_pdf);
  genRandArray(derived_r.m_dpdiss_ave,      engine, dp_pdf);

  // Random tangents (du0) on the inputs: qdp, vn0 and dp
  auto dqdp = Kokkos::create_mirror(tracers_r.qdp);
  genRandArray(dqdp, engine, unit_pdf);
  auto dvn0_h = Kokkos::create_mirror(derived_r.m_vn0);
  genRandArray(dvn0_h, engine, unit_pdf);
  auto ddp_h = Kokkos::create_mirror(derived_r.m_dp);
  genRandArray(ddp_h, engine, unit_pdf);
  TracersVn0AdjView dvn0("dvn0",num_elems);
  Kokkos::deep_copy(dvn0,dvn0_h);
  TracersDpAdjView ddp("ddp",num_elems);
  Kokkos::deep_copy(ddp,ddp_h);

  // A random seed lambdaN on qdp after the step
  TracersQdpAdjView qdp_seed("qdp_seed",num_elems);
  {
    auto h = Kokkos::create_mirror(qdp_seed);
    genRandArray(h, engine, unit_pdf);
    Kokkos::deep_copy(qdp_seed,h);
  }

  // ==== Forward-tangent half (ST=DpFadType): exact duN = J*du0. ====
  // Runs the fwd with the given nu_q, and returns <J*du0, lambdaN>. The base point is
  // reset each time (the fwd overwrites qdp, and DSS's/recomputes some of the derived fields).
  auto duN_qdp = Kokkos::create_mirror(tracers_r.qdp);
  // mq/mv/md in {0,1} switch on the tangent on qdp/vn0/dp: running with only one of them
  // on isolates the part of the chain that carries it.
  auto zq = Kokkos::create_mirror(tracers_r.qdp);
  Kokkos::deep_copy(zq,Real(0));
  auto zv = Kokkos::create_mirror(derived_r.m_vn0);
  Kokkos::deep_copy(zv,Real(0));
  auto zd = Kokkos::create_mirror(derived_r.m_dp);
  Kokkos::deep_copy(zd,Real(0));
  auto run_forward = [&](const Real nu_q, const int mq, const int mv, const int md) {
    params.nu_q = nu_q;
    euler_dp.reset(params);

    // Base point's .val() first (zeroes any pre-existing tangent), *then* the tangents
    ut::set_val(tracers_dp.qdp,                 tracers_r.qdp);
    ut::set_val(derived_dp.m_dp,                derived_r.m_dp);
    ut::set_val(derived_dp.m_vn0,               derived_r.m_vn0);
    ut::set_val(derived_dp.m_divdp,             derived_r.m_divdp);
    ut::set_val(derived_dp.m_divdp_proj,        derived_r.m_divdp_proj);
    ut::set_val(derived_dp.m_eta_dot_dpdn,      derived_r.m_eta_dot_dpdn);
    ut::set_val(derived_dp.m_omega_p,           derived_r.m_omega_p);
    ut::set_val(derived_dp.m_dpdiss_biharmonic, derived_r.m_dpdiss_biharmonic);
    ut::set_val(derived_dp.m_dpdiss_ave,        derived_r.m_dpdiss_ave);

    ut::set_deriv0(tracers_dp.qdp,  mq ? dqdp : zq);
    ut::set_deriv0(derived_dp.m_vn0, mv ? dvn0_h : zv);
    ut::set_deriv0(derived_dp.m_dp,  md ? ddp_h : zd);

    // See prim_advec_tracers_remap_RK2 in prim_advec_tracers_remap.cpp
    euler_dp.precompute_divdp();
    euler_dp.euler_step(np1_qdp,n0_qdp, dt/2.0,0.0,DSSOption::DIV_VDP_AVE);
    euler_dp.euler_step(np1_qdp,np1_qdp,dt/2.0,1.0,DSSOption::ETA);
    euler_dp.euler_step(np1_qdp,np1_qdp,dt/2.0,2.0,DSSOption::OMEGA);
    euler_dp.qdp_time_avg(n0_qdp,np1_qdp);

    ut::get_deriv0(duN_qdp, tracers_dp.qdp);
    return ut::dot_qdp(qdp_seed,duN_qdp,np1_qdp,qsize);
  };

  // ==== Adjoint half (ST=Real): J^T*lambdaN, at the SAME base point. ====
  // Runs the adjoint with the given nu_q, and returns <du0, J^T*lambdaN>
  auto run_adjoint = [&](const Real nu_q, const int mq, const int mv, const int md) {
    params.nu_q = nu_q; // prim_advec_tracers_remap_adj resets the euler functor from params

    TracersQdpAdjView adj_qdp("adj_qdp",num_elems);
    Kokkos::deep_copy(adj_qdp,qdp_seed);
    TracersVn0AdjView adj_vn0("adj_vn0",num_elems);
    TracersDpAdjView  adj_dp("adj_dp",num_elems);

    prim_advec_tracers_remap_adj(euler_r,dt,adj_qdp,adj_vn0,adj_dp);

    return (mq ? ut::dot_qdp(adj_qdp,dqdp,n0_qdp,qsize) : 0.0)
         + (mv ? dot(ekat::scalarize(adj_vn0),ekat::scalarize(dvn0),nlevs) : 0.0)
         + (md ? dot(ekat::scalarize(adj_dp),ekat::scalarize(ddp),nlevs) : 0.0);
  };

  // <du0, J^T*lambdaN> vs <J*du0, lambdaN>, for several values of nu_q. Each is checked with
  // all the tangents on at once, and with each input's alone (to tell which input's path
  // through the adjoint is off, if one is).
  // With nu_q=0 the biharmonic term of the last euler step vanishes, so that case checks all
  // the rest of the chain (qlim chaining, qdp_time_avg, precompute_divdp, the dependence on
  // vn0 and dp). The larger values add the adjoint of the biharmonic term.
  constexpr auto tol = std::numeric_limits<double>::epsilon()*1e5;
  const Real nu_q_list[] = {0.0, 1e16, 1e18, 1e20};
  const int masks[4][3] = {{1,1,1},{1,0,0},{0,1,0},{0,0,1}};
  const char* mnames[4] = {"all","qdp","vn0","dp"};
  double lhs_first = 0, lhs_last = 0;
  for (const Real nu_q : nu_q_list) {
    double scale = 0; // magnitude of the full inner product
    for (int m=0; m<4; ++m) {
      const int mq = masks[m][0], mv = masks[m][1], md = masks[m][2];
      const double rhs = run_forward(nu_q,mq,mv,md);
      const double lhs = run_adjoint(nu_q,mq,mv,md);
      if (m==0) scale = std::abs(rhs);
      if (comm.am_i_root())
        std::cout << std::setprecision(15)
                  << "   nu_q=" << nu_q << " [" << mnames[m] << "]:  <du0,J^T*lambdaN> = " << lhs
                  << ",  <J*du0,lambdaN> = " << rhs
                  << ",  rel err = " << std::abs(lhs-rhs)/std::abs(rhs) << "\n";
      INFO ("nu_q = " << nu_q << ", tangent on " << mnames[m]);
      // The single-input products can be orders of magnitude smaller than the full one
      // (e.g., the dp tangent barely moves qdp), so roundoff in the full-size terms they are
      // computed from can dominate them. Allow for it with a margin relative to the full product.
      CHECK_THAT (lhs, Catch::Matchers::WithinRel(rhs,tol) ||
                       Catch::Matchers::WithinAbs(rhs,1e-2*tol*scale));

      if (m==0) {
        if (nu_q==nu_q_list[0]) lhs_first = lhs;
        lhs_last = lhs;
      }
    }
  }

  // Sanity check on the test itself: make sure that the biharmonic term is not negligible
  // for the largest nu_q, by checking that the adjoint changes when we turn it off.
  CHECK (std::abs(lhs_last-lhs_first) > 1e-9*std::abs(lhs_last));
}

} // namespace Homme

#endif // HOMMEXX_ENABLE_FAD_TYPES
