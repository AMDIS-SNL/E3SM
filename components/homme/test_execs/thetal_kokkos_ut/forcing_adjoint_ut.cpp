/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

// Adjoint of ForcingFunctor::states_forcing (the momentum/vtheta/phi half of
// ForcingFunctor's forward pass): deposits dJ/d(fm) directly (fm is applied
// to the state verbatim), plus dJ/d(fvtheta) and dJ/d(fphi) (intermediate
// quantities produced from ft by tracers_forcing, whose own adjoint is not
// implemented yet).
//
// states_forcing is an exactly affine map (no bias term) of
// (state(np1), forcing) -> state(np1), so two independent checks are used:
//  - "dot-product consistency": the algebraic adjoint identity
//    <lambda_out, J*[dx;df]> == <lambda_in, dx> + <dJ_dforcing, df>,
//    computed *exactly* (up to floating-point round-off) by applying the
//    forward operator directly to a random tangent (dx,df) -- valid because
//    the map is linear, so this needs no finite perturbation step (unlike
//    the check below).
//  - "finite-difference check": a literal central-difference check of
//    dJ/d(applied_forcing) around a random, nonzero base point. Since the
//    map is affine, this should match the adjoint's prediction far more
//    tightly than a typical (nonlinear) FD check -- limited by round-off,
//    not by truncation error.

#include <catch2/catch.hpp>

#include <random>

#include "Context.hpp"
#include "ElementsForcing.hpp"
#include "ElementsGeometry.hpp"
#include "ElementsState.hpp"
#include "ForcingFunctor.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HybridVCoord.hpp"
#include "PhysicalConstants.hpp"
#include "SimulationParams.hpp"
#include "StateSnapshot.hpp"
#include "Tracers.hpp"
#include "Types.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_comm.hpp>
#include <ekat_pack_kokkos.hpp>

using namespace Homme;

namespace {

// Sum of dot products over the (v, vtheta_dp, w_i, phinh_i) fields that
// states_forcing/states_forcing_adj actually touch. dp3d/ps_v are ignored:
// states_forcing never reads or writes them.
double dot_state (const StateSnapshot& a, const StateSnapshot& b) {
  return dot(ekat::scalarize(a.v),        ekat::scalarize(b.v))
       + dot(ekat::scalarize(a.vtheta_dp),ekat::scalarize(b.vtheta_dp))
       + dot(ekat::scalarize(a.w_i),      ekat::scalarize(b.w_i))
       + dot(ekat::scalarize(a.phinh_i),  ekat::scalarize(b.phinh_i));
}

// Sum of dot products over the forcing fields states_forcing reads (m_fm,
// m_fvtheta, m_fphi). m_ft is ignored: states_forcing never reads it (it's
// tracers_forcing's input).
double dot_forcing (const ElementsForcing& a, const ElementsForcing& b) {
  return dot(ekat::scalarize(a.m_fm),     ekat::scalarize(b.m_fm))
       + dot(ekat::scalarize(a.m_fvtheta),ekat::scalarize(b.m_fvtheta))
       + dot(ekat::scalarize(a.m_fphi),   ekat::scalarize(b.m_fphi));
}

// y += alpha*x, via a host round-trip. Fine for the small problem sizes used
// in this unit test; avoids needing a device-side axpy kernel.
template<typename ViewT>
void axpy (const ViewT& y, const Real alpha, const ViewT& x) {
  auto y_h = Kokkos::create_mirror_view(y);
  auto x_h = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(y_h,y);
  Kokkos::deep_copy(x_h,x);
  for (size_t i=0; i<y_h.span(); ++i) {
    y_h.data()[i] += alpha*x_h.data()[i];
  }
  Kokkos::deep_copy(y,y_h);
}

void forcing_deep_copy (ElementsForcing& dst, const ElementsForcing& src) {
  Kokkos::deep_copy(dst.m_fm,src.m_fm);
  Kokkos::deep_copy(dst.m_fvtheta,src.m_fvtheta);
  Kokkos::deep_copy(dst.m_ft,src.m_ft);
  Kokkos::deep_copy(dst.m_fphi,src.m_fphi);
}

void forcing_axpy (ElementsForcing& y, const Real alpha, const ElementsForcing& x) {
  axpy(y.m_fm,alpha,x.m_fm);
  axpy(y.m_fvtheta,alpha,x.m_fvtheta);
  axpy(y.m_fphi,alpha,x.m_fphi);
}

template<typename rngAlg, typename PDF>
void randomize_state_snapshot (StateSnapshot& s, rngAlg& engine, PDF&& pdf) {
  genRandArray(s.v,        engine, pdf);
  genRandArray(s.vtheta_dp,engine, pdf);
  genRandArray(s.w_i,      engine, pdf);
  genRandArray(s.phinh_i,  engine, pdf);
}

} // anonymous namespace

TEST_CASE("forcing_adjoint_states", "forcing_adjoint") {
  using rngAlg = std::mt19937_64;
  using ipdf = std::uniform_int_distribution<int>;
  using dpdf = std::uniform_real_distribution<double>;

  std::random_device rd;
  constexpr int num_elems = 3;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);

  // Init everything through singleton, which is what happens in normal runs.
  // Note: unlike most of the other thetal_kokkos_ut tests, this one needs no
  // F90 side at all: states_forcing is a purely local, per-column
  // computation (no MPI/connectivity), and ElementsGeometry/ElementsState's
  // own init()/randomize() are pure C++.
  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);
  auto& p = c.create<SimulationParams>();
  p.dt_remap_factor = 1;
  p.theta_hydrostatic_mode = false;

  auto& hv = c.create<HybridVCoord>();
  hv.random_init(seed);

  auto& geo = c.create<ElementsGeometry>();
  geo.init(num_elems,true, /* alloc_gradphis = */ true,
           PhysicalConstants::rearth0);
  geo.randomize(seed);

  auto& state = c.create<ElementsState>();
  state.init(num_elems);

  auto& forcing = c.create<ElementsForcing>();
  forcing.init(num_elems);

  auto& tracers = c.create<Tracers>();
  p.qsize = 1;
  tracers.init(num_elems,p.qsize);

  const Real dt = dpdf(0.1,10.0)(engine);
  const int np1 = ipdf(0,2)(engine);

  FunctorsBuffersManager fbm;
  ForcingFunctor ff;
  fbm.request_size(ff.requested_buffer_size());
  fbm.allocate();
  ff.init_buffers(fbm);

  SECTION ("dot-product consistency") {
    // states_forcing has no bias term, so applying it directly to a random
    // "tangent" (dx,df) computes the *exact* directional derivative
    // dy = J_state*dx + J_forcing*df -- an exact result, not an
    // approximation of one via a finite perturbation.
    state.randomize(seed,hv);
    forcing.randomize(seed+1);

    StateSnapshot dx(num_elems,false);
    state.take_snapshot(dx,np1,false);

    ElementsForcing df;
    df.init(num_elems);
    forcing_deep_copy(df,forcing);

    ff.states_forcing(dt,np1);

    StateSnapshot dy(num_elems,false);
    state.take_snapshot(dy,np1,false);

    StateSnapshot lambda_out(num_elems,false);
    randomize_state_snapshot(lambda_out,engine,dpdf(-1,1));

    const double lhs = dot_state(lambda_out,dy);

    ElementsForcing dJ_dforcing;
    dJ_dforcing.init(num_elems);
    dJ_dforcing.zero();

    StateSnapshot lambda = lambda_out.clone(true);
    ff.states_forcing_adj(dt,np1,lambda,dJ_dforcing);

    const double rhs = dot_state(lambda,dx) + dot_forcing(dJ_dforcing,df);

    REQUIRE_THAT (lhs, Catch::Matchers::WithinRel(rhs,1e-10) || Catch::Matchers::WithinAbs(rhs,1e-10));
  }

  SECTION ("finite-difference check") {
    const Real eps = 1e-6;

    state.randomize(seed,hv);
    forcing.randomize(seed+1);

    StateSnapshot x0(num_elems,false);
    state.take_snapshot(x0,np1,false);
    ElementsForcing f0;
    f0.init(num_elems);
    forcing_deep_copy(f0,forcing);

    StateSnapshot dx(num_elems,false);
    randomize_state_snapshot(dx,engine,dpdf(-1,1));

    ElementsForcing df;
    df.init(num_elems);
    df.randomize(seed+2);

    StateSnapshot lambda_out(num_elems,false);
    randomize_state_snapshot(lambda_out,engine,dpdf(-1,1));

    auto eval_J = [&](const Real alpha) -> double {
      StateSnapshot x = x0.clone(true);
      StateSnapshot dxs = dx.clone(true);
      dxs.scale(alpha);
      x.add(dxs);
      state.import_snapshot(x,np1,false);

      ElementsForcing f;
      f.init(num_elems);
      forcing_deep_copy(f,f0);
      forcing_axpy(f,alpha,df);
      forcing_deep_copy(forcing,f);

      ff.states_forcing(dt,np1);

      StateSnapshot y(num_elems,false);
      state.take_snapshot(y,np1,false);
      return dot_state(lambda_out,y);
    };

    const double Jp = eval_J(eps);
    const double Jm = eval_J(-eps);
    const double dJ_fd = (Jp - Jm) / (2*eps);

    ElementsForcing dJ_dforcing;
    dJ_dforcing.init(num_elems);
    dJ_dforcing.zero();

    StateSnapshot lambda = lambda_out.clone(true);
    ff.states_forcing_adj(dt,np1,lambda,dJ_dforcing);

    const double dJ_adj = dot_state(lambda,dx) + dot_forcing(dJ_dforcing,df);

    REQUIRE_THAT (dJ_fd, Catch::Matchers::WithinRel(dJ_adj,1e-6) || Catch::Matchers::WithinAbs(dJ_adj,1e-8));
  }

  c.finalize_singleton();
}
