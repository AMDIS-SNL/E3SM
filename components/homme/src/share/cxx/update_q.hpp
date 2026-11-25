/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_UPDATE_Q_HPP
#define HOMMEXX_UPDATE_Q_HPP

#include "ElementsState.hpp"
#include "HybridVCoord.hpp"
#include "Tracers.hpp"
#include "Types.hpp"

namespace Homme
{

// Q = qdp(np1_qdp)/dp, with dp = hyai_delta*ps0 + hybi_delta*ps_v(np1) the reference
// layer thickness (which, right after vertical_remap, is dp3d(np1)), for the first qsize
// tracers. Templated on the scalar type, so that tests can differentiate the actual fwd
// code (e.g., with ST=DpFadType). update_q (in prim_driver.cpp) is the ST=ScalarValue
// version on the objects in the Context.
template<typename ST>
void update_q_impl (const ElementsStateST<ST>& state, const TracersST<ST>& tracers,
                    const HybridVCoord& hvcoord, const int np1_qdp, const int np1,
                    const int qsize)
{
  using PT = PackType<ST>;

  auto hyai_delta = hvcoord.hybrid_ai_delta;
  auto hybi_delta = hvcoord.hybrid_bi_delta;
  const Real ps0 = hvcoord.ps0;

  auto ps_v = state.m_ps_v;
  auto qdp  = tracers.qdp;
  auto Q    = tracers.Q;

  const int num_elems = state.num_elems();
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,num_elems*qsize*NP*NP*NUM_LEV),
                       KOKKOS_LAMBDA(const int idx) {
    const int ie    =  idx / (qsize*NP*NP*NUM_LEV);
    const int iq    = (idx / (NP*NP*NUM_LEV)) % qsize;
    const int igp   = (idx / (NP*NUM_LEV)) % NP;
    const int jgp   = (idx / NUM_LEV) % NP;
    const int level =  idx % NUM_LEV;

    const PT dp = hyai_delta(level)*ps0 + hybi_delta(level)*ps_v(ie,np1,igp,jgp);
    Q(ie,iq,igp,jgp,level) = qdp(ie,np1_qdp,iq,igp,jgp,level)/dp;
  });
}

// Q := qdp/dp on the objects in the Context (see update_q_impl)
void update_q (const int np1_qdp, const int np1);

} // namespace Homme

#endif // HOMMEXX_UPDATE_Q_HPP
