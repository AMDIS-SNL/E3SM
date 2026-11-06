#include "vertical_remap_adj.hpp"

#include "Context.hpp"
#include "Elements.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HommexxEnums.hpp"
#include "HybridVCoord.hpp"
#include "PpmRemap.hpp"
#include "RemapFunctor.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "Tracers.hpp"

#include "profiling.hpp"

#include <ekat_pack_kokkos.hpp>

#include <memory>
#include <type_traits>

namespace Homme
{

namespace {

using DxST = DxFadTypeRemap;

// Type-erased Dx-Fad remap functor (its type depends on the remap algorithm),
// along with its scratch buffers.
struct RemapAdjBase {
  virtual ~RemapAdjBase () = default;

  // Assuming the Dx-Fad state(np1)/qdp(np1_qdp) hold the values of the fwd remap
  // inputs, seed them, run the fwd remap on them, and compute y = J^T x.
  virtual void apply (const int np1, const int np1_qdp, const Real dt,
                      const StateSnapshot& x, const TracersQdpAdjView& x_qdp,
                      StateSnapshot& y, const TracersQdpAdjView& y_qdp) = 0;
};

template<typename RemapAlgType>
struct RemapAdjImpl : public RemapAdjBase {
  using functor_t = Remap::RemapFunctor<true,RemapAlgType,DxST>;

  RemapAdjImpl (const int qsize, const ElementsST<DxST>& elems,
                const TracersST<DxST>& tracers, const HybridVCoord& hvcoord)
   : m_state(elems.m_state)
   , m_tracers(tracers)
   , m_remap(qsize,elems,tracers,hvcoord)
  {
    m_fbm.request_size(m_remap.requested_buffer_size());
    m_fbm.allocate();
    m_remap.init_buffers(m_fbm);
  }

  void apply (const int np1, const int np1_qdp, const Real dt,
              const StateSnapshot& x, const TracersQdpAdjView& x_qdp,
              StateSnapshot& y, const TracersQdpAdjView& y_qdp) override
  {
    m_remap.init_J(np1,m_state);
    m_remap.init_J_qdp(np1_qdp,m_tracers);
    Kokkos::fence();
    m_remap.run_remap(np1,np1_qdp,dt);
    Kokkos::fence();
    m_remap.run_JtV(np1,np1_qdp,m_state,m_tracers,x,x_qdp,y,y_qdp);
  }

  ElementsStateST<DxST>   m_state;
  TracersST<DxST>         m_tracers;
  functor_t               m_remap;
  FunctorsBuffersManager  m_fbm;
};

// What vertical_remap_adj keeps across calls. The Dx-Fad remap functor reads
// some params at construction (e.g., theta_hydrostatic_mode, via the state
// provider), so it is rebuilt if any of them changed.
struct VerticalRemapAdjScratch {
  void setup (const SimulationParams& params, const ElementsST<DxST>& elems_dx,
              const HybridVCoord& hvcoord)
  {
    const int nelem = elems_dx.num_elems();
    const bool same = m_impl and
                      m_nelem==nelem and
                      m_qsize==params.qsize and
                      m_hydrostatic==params.theta_hydrostatic_mode and
                      m_alg==params.remap_alg and
                      m_state_ptr==elems_dx.m_state.m_v.data();
    if (same) return;

    m_impl.reset();
    m_nelem = nelem;
    m_qsize = params.qsize;
    m_hydrostatic = params.theta_hydrostatic_mode;
    m_alg = params.remap_alg;
    m_state_ptr = elems_dx.m_state.m_v.data();

    m_tracers = TracersST<DxST>();
    m_tracers.init(nelem,m_qsize);
    m_y = StateSnapshot(nelem);
    m_y_qdp = TracersQdpAdjView("vertical_remap_adj y_qdp",nelem);

    using namespace Remap::Ppm;
    if (m_alg==RemapAlg::PPM_MIRRORED) {
      m_impl = std::make_shared<RemapAdjImpl<PpmVertRemap<PpmMirrored<DxST>>>>(
          m_qsize,elems_dx,m_tracers,hvcoord);
    } else if (m_alg==RemapAlg::PPM_LIMITED_EXTRAP) {
      m_impl = std::make_shared<RemapAdjImpl<PpmVertRemap<PpmLimitedExtrap<DxST>>>>(
          m_qsize,elems_dx,m_tracers,hvcoord);
    } else {
      EKAT_ERROR_MSG("[vertical_remap_adj] Error! Unknown remap algorithm.\n");
    }
  }

  std::shared_ptr<RemapAdjBase> m_impl;
  TracersST<DxST>   m_tracers;
  StateSnapshot     m_y = StateSnapshot(0);
  TracersQdpAdjView m_y_qdp;

  int       m_nelem = -1;
  int       m_qsize = -1;
  bool      m_hydrostatic = false;
  RemapAlg  m_alg = RemapAlg::PPM_MIRRORED;
  const void* m_state_ptr = nullptr;
};

// The Dx-Fad Elements: the one in the Context if there (e.g., the one prim_advance_adj
// uses for DIRK, which has the same scalar type), or a new one, put in the Context.
ElementsST<DxST>& get_dx_elements (const ElementsST<Real>& elems,
                                   const SimulationParams& params)
{
  auto& c = Context::singleton();
  const int nelem = elems.num_elems();
  if (not c.has<ElementsST<DxST>>()) {
    auto& e = c.create<ElementsST<DxST>>();
    e.init(nelem,true,params.scale_factor,params.laplacian_rigid_factor);
    e.m_geometry = elems.m_geometry;
  }
  auto& e = c.get<ElementsST<DxST>>();
  EKAT_REQUIRE_MSG (e.inited() and e.num_elems()==nelem,
      "[vertical_remap_adj] Error! The Dx-Fad Elements in the Context are not inited, "
      "or have the wrong number of elements.\n");
  // The remap of w_i uses phis/gradphis (surface b.c.), so make sure they're the real ones
  EKAT_REQUIRE_MSG (e.m_geometry.m_phis.data()==elems.m_geometry.m_phis.data() and
                    e.m_geometry.m_gradphis.data()==elems.m_geometry.m_gradphis.data(),
      "[vertical_remap_adj] Error! The Dx-Fad Elements in the Context must share the "
      "geometry of the Real Elements.\n");
  return e;
}

} // anonymous namespace

void vertical_remap_adj (const Real dt,
                         StateSnapshot& adj_state,
                         TracersQdpAdjView& adj_qdp)
{
  GPTLstart("vertical_remap_adj");

  auto& c = Context::singleton();
  const auto& params  = c.get<SimulationParams>();
  const auto& tl      = c.get<TimeLevel>();
  const auto& hvcoord = c.get<HybridVCoord>();

  EKAT_REQUIRE_MSG ((std::is_same_v<ScalarValue,Real>),
      "[vertical_remap_adj] Error! The fwd state must have Real scalar type.\n");
  EKAT_REQUIRE_MSG (params.rsplit>0,
      "[vertical_remap_adj] Error! Only vertically Lagrangian dynamics (rsplit>0) is supported.\n");
  EKAT_REQUIRE_MSG (params.qsize>=0 and params.qsize<=QSIZE_D,
      "[vertical_remap_adj] Error! Invalid number of tracers.\n");

  const auto& elems   = c.get<ElementsST<Real>>();
  const auto& tracers = c.get<TracersST<Real>>();
  const int nelem = elems.num_elems();
  EKAT_REQUIRE_MSG (adj_state.num_elems==nelem and adj_qdp.extent_int(0)==nelem,
      "[vertical_remap_adj] Error! Adjoint views have the wrong number of elements.\n");
  EKAT_REQUIRE_MSG (adj_state.ps_v.size()==0 or adj_state.ps_v.extent_int(0)==nelem,
      "[vertical_remap_adj] Error! adj_state.ps_v has the wrong number of elements.\n");
  EKAT_REQUIRE_MSG (tracers.num_tracers()>=params.qsize,
      "[vertical_remap_adj] Error! The Tracers in the Context have fewer than qsize tracers.\n");

  const int np1     = tl.np1;
  const int np1_qdp = tl.np1_qdp;

  auto& elems_dx = get_dx_elements(elems,params);
  auto& scratch  = c.create_if_not_there<VerticalRemapAdjScratch>();
  scratch.setup(params,elems_dx,hvcoord);

  // The fwd inputs (values only; derivatives are seeded by apply)
  elems_dx.m_state.import_values(elems.m_state,np1);
  scratch.m_tracers.import_values(tracers,np1_qdp);
  Kokkos::fence();

  // y = J^T * adj
  scratch.m_impl->apply(np1,np1_qdp,dt,adj_state,adj_qdp,scratch.m_y,scratch.m_y_qdp);

  // Output
  const auto& y = scratch.m_y;
  Kokkos::deep_copy(adj_state.v,        y.v);
  Kokkos::deep_copy(adj_state.vtheta_dp,y.vtheta_dp);
  Kokkos::deep_copy(adj_state.dp3d,     y.dp3d);
  Kokkos::deep_copy(adj_state.w_i,      y.w_i);
  Kokkos::deep_copy(adj_state.phinh_i,  y.phinh_i);
  if (adj_state.ps_v.size()>0) {
    Kokkos::deep_copy(adj_state.ps_v,Real(0));
  }
  Kokkos::deep_copy(adj_qdp,scratch.m_y_qdp);
  Kokkos::fence();

  GPTLstop("vertical_remap_adj");
}

void update_q_adj (const int np1_qdp, const int np1,
                   const TracersQdpAdjView& adj_Q,
                   TracersQdpAdjView& adj_qdp,
                   RemapPsAdjView& adj_ps_v)
{
  GPTLstart("update_q_adj");

  auto& c = Context::singleton();
  const auto& params  = c.get<SimulationParams>();
  const auto& hvcoord = c.get<HybridVCoord>();
  const auto& elems   = c.get<ElementsST<Real>>();
  const auto& tracers = c.get<TracersST<Real>>();

  EKAT_REQUIRE_MSG ((std::is_same_v<ScalarValue,Real>),
      "[update_q_adj] Error! The fwd state must have Real scalar type.\n");

  const int nelem = elems.num_elems();
  const int qsize = params.qsize;
  EKAT_REQUIRE_MSG (adj_Q.extent_int(0)==nelem and adj_qdp.extent_int(0)==nelem and
                    adj_ps_v.extent_int(0)==nelem,
      "[update_q_adj] Error! Adjoint views have the wrong number of elements.\n");

  const Real ps0 = hvcoord.ps0;
  auto hyai = ekat::scalarize(hvcoord.hybrid_ai_delta);
  auto hybi = ekat::scalarize(hvcoord.hybrid_bi_delta);
  auto ps_v = elems.m_state.m_ps_v;
  auto qdp  = ekat::scalarize(tracers.qdp);
  auto a_Q  = ekat::scalarize(adj_Q);
  auto a_q  = ekat::scalarize(adj_qdp);
  auto a_ps = adj_ps_v;

  // Q = qdp/dp, with dp = hyai*ps0 + hybi*ps:
  //   dQ/dqdp = 1/dp,  dQ/dps = -qdp*hybi/dp^2
  using md3_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>;
  using md5_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>>;
  if (qsize>0) {
    Kokkos::parallel_for(md5_t({0,0,0,0,0},{nelem,qsize,NP,NP,NUM_PHYSICAL_LEV}),
                         KOKKOS_LAMBDA (const int ie, const int iq, const int i, const int j, const int k) {
      const Real dp = hyai(k)*ps0 + hybi(k)*ps_v(ie,np1,i,j);
      a_q(ie,iq,i,j,k) += a_Q(ie,iq,i,j,k)/dp;
    });
  }
  Kokkos::parallel_for(md3_t({0,0,0},{nelem,NP,NP}),
                       KOKKOS_LAMBDA (const int ie, const int i, const int j) {
    Real s = 0;
    for (int iq=0; iq<qsize; ++iq) {
      for (int k=0; k<NUM_PHYSICAL_LEV; ++k) {
        const Real dp = hyai(k)*ps0 + hybi(k)*ps_v(ie,np1,i,j);
        s -= a_Q(ie,iq,i,j,k)*qdp(ie,np1_qdp,iq,i,j,k)*hybi(k)/(dp*dp);
      }
    }
    a_ps(ie,i,j) += s;
  });
  Kokkos::fence();

  GPTLstop("update_q_adj");
}

} // namespace Homme
