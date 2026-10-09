#include "ComposeTransport.hpp"
#include "compose_test.hpp"

#include "Types.hpp"
#include "thetal_ut_session.hpp"
#include "Context.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"
#include "FunctorsBuffersManager.hpp"
#include "SimulationParams.hpp"
#include "Elements.hpp"
#include "Tracers.hpp"
#include "TimeLevel.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "PhysicalConstants.hpp"
#include "ReferenceElement.hpp"
#include "SphereOperators.hpp"
#include "ElementOps.hpp"
#include "profiling.hpp"
#include "ErrorDefs.hpp"
#include "VerticalRemapManager.hpp"
#include "PhysicalConstants.hpp"

#include "utilities/TestUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/ViewUtils.hpp"

#include <ekat_comm.hpp>
#include <ekat_string_utils.hpp>
#include <ekat_test_utils.hpp>

#include <catch2/catch.hpp>
#include <random>

using namespace Homme;

extern "C" {
  // Compose-specific f90 init. Must be called after init_f90
  void init_compose_f90(int ne, int qsize, int hv_q, int limiter_option, bool cdr_check,
                        bool is_sphere, bool nearest_point, int halo, int traj_nsubstep);
  void init_geometry_f90();
  // Must be called before the f90 cleanup
  void finalize_compose_f90();
  void run_compose_standalone_test_f90(int* nmax, Real* eval, int* nerr);
  void run_trajectory_f90(Real t0, Real t1, bool independent_time_steps, Real* dep,
                          Real* dprecon);
  void run_sl_vertical_remap_bfb_f90(Real* diagnostic);
} // extern "C"

using CA4d = Kokkos::View<Real****, Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA5d = Kokkos::View<Real*****, Kokkos::LayoutRight, Kokkos::HostSpace>;

template <typename V>
decltype(Kokkos::create_mirror_view(V())) cmvdc (const V& v) {
  const auto h = Kokkos::create_mirror_view(v);
  deep_copy(h, v);
  return h;
}

template <typename View> static
Real* pack2real (const View& v) { return &(*v.data())[0]; }
template <typename View> static
ScalarValue* pack2scalar (const View& v) { return &(*v.data())[0]; }

class Random {
  using rngalg = std::mt19937_64;
  using rpdf = std::uniform_real_distribution<Real>;
  using ipdf = std::uniform_int_distribution<int>;
  std::random_device rd;
  unsigned int seed;
  rngalg engine;
public:
  Random (unsigned int seed_ = Catch::rngSeed()) : seed(seed_ == 0 ? rd() : seed_), engine(seed) {}
  unsigned int gen_seed () { return seed; }
  Real urrng (const Real lo = 0, const Real hi = 1) { return rpdf(lo, hi)(engine); }
  int  uirng (const int lo, const int hi) { return ipdf(lo, hi)(engine); }
};

template <typename V>
void fill (Random& r, const V& a, const Real scale = 1,
           typename std::enable_if<V::rank == 3>::type* = 0) {
  const auto am = cmvdc(a);
  for (int i = 0; i < a.extent_int(0); ++i)
    for (int j = 0; j < a.extent_int(1); ++j)
      for (int k = 0; k < a.extent_int(2); ++k)
        for (int s = 0; s < VECTOR_SIZE; ++s)
          am(i,j,k)[s] = scale*r.urrng(-1,1); 
  deep_copy(a, am);
}

template <typename V>
void fill (Random& r, const V& a,
           typename std::enable_if<V::rank == 4>::type* = 0) {
  const auto am = cmvdc(a);
  for (int i = 0; i < a.extent_int(0); ++i)
    for (int j = 0; j < a.extent_int(1); ++j)
      for (int k = 0; k < a.extent_int(2); ++k)
        for (int l = 0; l < a.extent_int(3); ++l)
          for (int s = 0; s < VECTOR_SIZE; ++s)
            am(i,j,k,l)[s] = r.urrng(-1,1); 
  deep_copy(a, am);
}

struct Session {
  int ne, hv_q, nmax, halo, traj_nsubstep;
  bool cdr_check, is_sphere, run_only_advection_test, nearest_point;
  HybridVCoord h;
  Random r;
  std::shared_ptr<Elements> e;
  int nelemd, qsize, nlev, np;
  FunctorsBuffersManager fbm;
  // Takes care of hvcoord, f90 mesh and cleanup (of f90 and Context)
  std::unique_ptr<CubeSphereTestSession> f90_session;

  //Session () : r(269041989) {}

  void init () {
    auto& c = Context::singleton();
    c.create<ekat::Comm>(MPI_COMM_WORLD);

    const auto seed = r.gen_seed();
    printf("seed %u\n", seed);

    nlev = NUM_PHYSICAL_LEV;
    assert(nlev > 0);
    np = NP;
    assert(np == 4);

    assert(QSIZE_D >= 4);
    parse_test_options ();
    assert(is_sphere); // planar isn't available in Hxx yet

    auto& p = c.create<SimulationParams>();
    p.transport_alg = 12;
    p.qsize = qsize;
    p.limiter_option = 9;
    p.hypervis_scaling = 0;
    p.remap_alg = RemapAlg::PPM_LIMITED_EXTRAP;
    p.qsplit = 1;
    p.rsplit = 1;
    p.dt_tracer_factor = 1;
    p.dt_remap_factor = 1;
    p.params_set = true;
    p.theta_hydrostatic_mode = true;
    p.scale_factor = is_sphere ? PhysicalConstants::rearth0 : 1;
    p.laplacian_rigid_factor = is_sphere ? 1/p.scale_factor : 0;

    // Create hvcoord and ref_FE, and init f90 (including its mesh/connectivity)
    f90_session = std::make_unique<CubeSphereTestSession>(ne, seed);
    init_compose_f90(ne, qsize, hv_q, p.limiter_option, cdr_check,
                     is_sphere, nearest_point, halo, traj_nsubstep);
    h = f90_session->hvcoord;
    auto& ref_FE = f90_session->ref_FE;

    nelemd = f90_session->num_elems();
    auto& bmm = c.create<MpiBuffersManagerMap>();
    bmm.set_connectivity(c.get_ptr<Connectivity>());
    e = c.get_ptr<Elements>();
    c.create<TimeLevel>();

    init_geometry_f90();    
    auto& geo = c.get<ElementsGeometry>();

    auto& sphop = c.create<SphereOperators>();
    sphop.setup(geo, ref_FE);
    
    auto& ct = c.create<ComposeTransport>();
    ct.reset(p);
    fbm.request_size(ct.requested_buffer_size());
    fbm.allocate();
    ct.init_buffers(fbm);
    ct.init_boundary_exchanges();

    c.create<VerticalRemapManagerST<ScalarValue>>();
  }

  void cleanup () {
    finalize_compose_f90();
    // This calls cleanup_f90 and finalizes the Context
    f90_session = nullptr;
  }

  static Session& singleton () {
    if ( ! s_session) {
      s_session = std::make_shared<Session>();
      s_session->init();
    }
    return *s_session;
  }

  // Call only in last line of last TEST_CASE.
  static void delete_singleton () {
    if (s_session) s_session->cleanup();
    s_session = nullptr;
  }

  const ekat::Comm& get_comm () const { return Context::singleton().get<ekat::Comm>(); }

private:
  static std::shared_ptr<Session> s_session;

  // compose_ut hommexx -ne NE -qsize QSIZE -hvq HV_Q -cdrcheck
  void parse_test_options () {
    ne = 2;
    qsize = QSIZE_D;
    hv_q = 1;
    cdr_check = false;
    is_sphere = true;
    run_only_advection_test = false;
    nmax = -1;
    halo = 2;
    traj_nsubstep = 0;
    nearest_point = true;

    auto& ts = ekat::TestSession::get();
    if (ts.params.count("ne")==1) {
      ne = std::stoi(ts.params.at("ne"));
      EKAT_REQUIRE_MSG(ne >= 2,"ne must be >= 2\n");
      ts.params.erase("ne");
    }
    if (ts.params.count("qsize")==1) {
      qsize = std::stoi(ts.params.at("qsize"));
      EKAT_REQUIRE_MSG(qsize>=1 and qsize <= QSIZE_D, "qsize must be >= 1 and <= QSIZE_D\n");
      ts.params.erase("qsize");
    }
    if (ts.params.count("hvq")==1) {
      hv_q = std::stoi(ts.params.at("hvq"));
      ts.params.erase("hvq");
    }
    if (ts.flags.count("cdrcheck")==1) {
      cdr_check = true;
      ts.flags.erase("cdrcheck");
    }
    if (ts.flags.count("planar")==1) {
      is_sphere = false;
      ts.flags.erase("planar");
    }
    if (ts.flags.count("convergence")==1) {
      run_only_advection_test = true;
      ts.flags.erase("convergence");
    }
    if (ts.params.count("nmax")==1) {
      nmax = std::stoi(ts.params.at("nmax"));
        EKAT_REQUIRE_MSG(nmax>=1, "nmax must be >= 1\n");
      ts.params.erase("nmax");
    }
    if (ts.params.count("halo")==1) {
      halo = std::stoi(ts.params.at("halo"));
      EKAT_REQUIRE_MSG(halo>=1, "halo must be >= 1");
      ts.params.erase("halo");
    }
    if (ts.params.count("trajnsubstep")==1) {
      traj_nsubstep = std::stoi(ts.params.at("trajnsubstep"));
      EKAT_REQUIRE_MSG(traj_nsubstep>=0, "traj_nsubstep must be >= 0");
      ts.params.erase("trajnsubstep");
    }
    if (ts.flags.count("nonearest")==1) {
      nearest_point = false;
      ts.flags.erase("nonearest");
    }

    auto print_key = [](const auto& it) {
      return it.first;
    };
    EKAT_REQUIRE_MSG(ts.flags.size()==0,
        "Error! Unrecognized flags: " + ekat::join(ts.flags,print_key,",") + "\n");
    EKAT_REQUIRE_MSG(ts.params.size()==0,
        "Error! Unrecognized flags: " + ekat::join(ts.params,print_key,",") + "\n");
    EKAT_REQUIRE_MSG(ts.vec_params.size()==0,
        "Error! Unrecognized flags: " + ekat::join(ts.vec_params,print_key,",") + "\n");

    ne = std::max(2, ne);
    qsize = std::max(1, std::min(QSIZE_D, qsize));
    hv_q = std::max(0, std::min(qsize, hv_q));

    if (get_comm().am_i_root()) {
      const int bfb =
#ifdef HOMMEXX_BFB_TESTING
        1;
#else
        0;
#endif
      printf("compose_ut> sphere %d bfb %d ne %d qsize %d hv_q %d cdr_check %d "
             "halo %d traj_nsubstep %d nearest %d\n",
             int(is_sphere), bfb, ne, qsize, hv_q, cdr_check ? 1 : 0, halo,
             traj_nsubstep, int(nearest_point));
    }
  }
};

std::shared_ptr<Session> Session::s_session;

static bool almost_equal (const Real& a, const Real& b,
                          const Real tol = 0) {
  const auto re = std::abs(a-b)/(1 + std::abs(a));
  const bool good = re <= tol;
  if ( ! good)
    printf("equal: a,b = %23.16e %23.16e re = %23.16e tol %9.2e\n",
           a, b, re, tol);
  return good;
}
static bool equal (const Real& a, const Real& b,
                   // Used only if not defined HOMMEXX_BFB_TESTING.
                   const Real tol = 0) {
#ifdef HOMMEXX_BFB_TESTING
  if (a != b)
    printf("equal: a,b = %23.16e %23.16e re = %23.16e\n",
           a, b, std::abs((a-b)/a));
  return a == b;
#else
  return almost_equal(a, b, tol);
#endif
}
#ifdef HOMMEXX_ENABLE_FWD_SENS
static bool almost_equal (const ScalarValue& a, const ScalarValue& b,
                   // Used only if not defined HOMMEXX_BFB_TESTING.
                   const Real tol = 0) {
  return almost_equal(ADValue(a),ADValue(b),tol);
}

static bool equal (const ScalarValue& a, const ScalarValue& b,
                   // Used only if not defined HOMMEXX_BFB_TESTING.
                   const Real tol = 0) {
  return equal(ADValue(a),ADValue(b),tol);
}
#endif

typedef ExecViewUnmanaged<ScalarValue*[NP][NP][NUM_LEV*VECTOR_SIZE]> RNlev;
typedef ExecViewUnmanaged<ScalarValue**[NP][NP][NUM_LEV*VECTOR_SIZE]> RsNlev;
typedef ExecViewUnmanaged<ScalarValue***[NP][NP][NUM_LEV*VECTOR_SIZE]> RssNlev;
typedef HostView<ScalarValue*[NP][NP][NUM_LEV*VECTOR_SIZE]> RNlevH;
typedef HostView<ScalarValue**[NP][NP][NUM_LEV*VECTOR_SIZE]> RsNlevH;
typedef HostView<ScalarValue***[NP][NP][NUM_LEV*VECTOR_SIZE]> RssNlevH;

void run_sl_vertical_remap_bfb_cpp (const Session& s, ComposeTransport& ct,
                                    Real& diagnostic) {
  using Kokkos::create_mirror_view;
  using Kokkos::deep_copy;

  static const Real c1 = 0.491827, c2 = 0.2432109, c3 = 0.1234567, c4 = 0.0832;
  
  const auto elements = Context::singleton().get<Elements>();
  const auto& state = elements.m_state;
  const auto& derived = elements.m_derived;
  const auto& tracers = Context::singleton().get<Tracers>();
  const auto& params = Context::singleton().get<SimulationParams>();

  TimeLevel tl;
  tl.np1 = 0;
  tl.nstep = 42;
  tl.update_tracers_levels(params.qsplit);

  const auto qdp_h = create_mirror_view(tracers.qdp);
  const auto dp3d_h = create_mirror_view(state.m_dp3d);
  const auto dp_h = create_mirror_view(derived.m_divdp);
  RssNlevH qdp(pack2scalar(qdp_h), s.nelemd, qdp_h.extent_int(1), qdp_h.extent_int(2));
  RsNlevH dp3d(pack2scalar(dp3d_h), s.nelemd, dp3d_h.extent_int(1));
  RNlevH dp(pack2scalar(dp_h), s.nelemd);
  for (int ie = 0; ie < s.nelemd; ++ie)
    for (int iq = 0; iq < tracers.num_tracers(); ++iq)
      for (int k = 0; k < s.nlev; ++k)
        for (int i = 0; i < s.np; ++i)
          for (int j = 0; j < s.np; ++j) {
            const int ie1 = ie+1, iq1 = iq+1, i1 = i+1, j1 = j+1, k1 = k+1, krev = s.nlev - k;
            if (iq == 0) {
              dp3d(ie,tl.np1,i,j,k) = k1 + c1*i1 + c2*j1*i1 + c3*(j1*j1 + j1*k1);
              dp(ie,i,j,k) = krev + c1*i1 + c2*j1*i1 + c3*(j1*j1 + j1*krev);
            }
            qdp(ie,tl.np1_qdp,iq,i,j,k) = ie1 + c1*i1 + c2*j1*i1 + c3*(j1*j1 + j1*k1) + c4*iq1;
          }
  deep_copy(tracers.qdp, qdp_h);
  deep_copy(state.m_dp3d, dp3d_h);
  deep_copy(derived.m_divdp, dp_h);

  ct.remap_q(tl);

  const auto q_h = cmvdc(tracers.Q);
  RsNlevH q(pack2scalar(q_h), s.nelemd, q_h.extent_int(1));
  
  diagnostic = 0;
  for (int ie = 0; ie < s.nelemd; ++ie)
    for (int iq = 0; iq < tracers.num_tracers(); ++iq)
      for (int k = 0; k < s.nlev; ++k)
        for (int j = 0; j < s.np; ++j)
          for (int i = 0; i < s.np; ++i)
            diagnostic += ADValue(q(ie,iq,i,j,k));
}

TEST_CASE ("compose_transport_testing") {
  static constexpr Real tol = std::numeric_limits<Real>::epsilon();

  auto& s = Session::singleton(); try {
  do { // breakable

  if (s.run_only_advection_test) {
    int nmax = s.nmax, nerr;
    std::vector<Real> eval_f((s.nlev+1)*s.qsize);
    run_compose_standalone_test_f90(&nmax, eval_f.data(), &nerr);
    REQUIRE(nerr == 0);
    break;
  }

  // unit tests
  REQUIRE(compose::test::slmm_unittest() == 0);
  REQUIRE(compose::test::cedr_unittest() == 0);
  REQUIRE(compose::test::interpolate_unittest() == 0);
  REQUIRE(compose::test::cedr_unittest(s.get_comm().mpi_comm()) == 0);

  auto& ct = Context::singleton().get<ComposeTransport>();
  const auto fails = ct.run_unit_tests();
  for (const auto& e : fails) printf("%s %d\n", e.first.c_str(), e.second);
  REQUIRE(fails.empty());

  // trajectory BFB
  if (s.traj_nsubstep == 0) {
    for (const bool independent_time_steps : {false, true}) {
      printf("independent_time_steps %d\n", independent_time_steps);
      const Real twelve_days = 3600 * 24 * 12;
      const Real t0 = 0.13*twelve_days;
      const Real t1 = independent_time_steps ? t0 + 1800 : 0.22*twelve_days;
      CA5d depf("depf", s.nelemd, s.nlev, s.np, s.np, 3);
      CA4d dpreconf("dpreconf", s.nelemd, s.nlev, s.np, s.np);
      run_trajectory_f90(t0, t1, independent_time_steps, depf.data(),
                         dpreconf.data());
      const auto depc = ct.test_trajectory(t0, t1, independent_time_steps);
      REQUIRE(depc.extent_int(0) == s.nelemd);
      REQUIRE(depc.extent_int(2) == s.np);
      REQUIRE(depc.extent_int(4) == 3);
      if (independent_time_steps) {
        const auto dpreconc = cmvdc(RNlev(pack2scalar(s.e->m_derived.m_divdp), s.nelemd));
        for (int ie = 0; ie < s.nelemd; ++ie)
          for (int lev = 0; lev < s.nlev; ++lev)
            for (int i = 0; i < s.np; ++i)
              for (int j = 0; j < s.np; ++j)
                REQUIRE(equal(dpreconf(ie,lev,i,j), dpreconc(ie,i,j,lev), 100*tol));
      }
      for (int ie = 0; ie < s.nelemd; ++ie)
        for (int lev = 0; lev < s.nlev; ++lev)
          for (int i = 0; i < s.np; ++i)
            for (int j = 0; j < s.np; ++j)
              for (int d = 0; d < 3; ++d)
                REQUIRE(equal(depf(ie,lev,i,j,d), depc(ie,lev,i,j,d), 100*tol));
    }
  }

  { // q vertical remap
    Real diagnostic_f90, diagnostic_cpp;
    run_sl_vertical_remap_bfb_f90(&diagnostic_f90);
    run_sl_vertical_remap_bfb_cpp(s, ct, diagnostic_cpp);
    REQUIRE(equal(diagnostic_f90, diagnostic_cpp, 1e2*tol));
  }

  { // 2D SL BFB
    int nmax = s.nmax, nerr;
    std::vector<Real> eval_f((s.nlev+1)*s.qsize), eval_c(eval_f.size());
    run_compose_standalone_test_f90(&nmax, eval_f.data(), &nerr);
    REQUIRE(nerr == 0);
    for (const bool bfb : {false, true}) {
#if ! defined HOMMEXX_BFB_TESTING
      if (bfb) continue;
#endif
      ct.test_2d(bfb, nmax, eval_c);
      if (s.get_comm().am_i_root()) {
        const Real f = bfb ? 0 : 1;
        const int n = s.nlev*s.qsize;
        // When not a BFB build, still expect l2 error to be the same to several digits.
        for (int i = 0; i < n; ++i) REQUIRE(almost_equal(eval_f[i], eval_c[i], f*1e5*tol));
        // Mass conservation error should be within a factor of 10 of each
        // other.
        for (int i = n; i < n + s.qsize; ++i) REQUIRE(almost_equal(eval_f[i], eval_c[i], f*10));
        // And mass conservation itself should be small.
        for (int i = n; i < n + s.qsize; ++i) REQUIRE(std::abs(eval_f[i]) <= 20*tol);
        for (int i = n; i < n + s.qsize; ++i) REQUIRE(std::abs(eval_c[i]) <= 20*tol);
        //todo add an l2 ceiling for some select tracers as a function of ne
      }
    }
  }

  } while (false); // do
  } catch (...) {}
  Session::delete_singleton();
}
