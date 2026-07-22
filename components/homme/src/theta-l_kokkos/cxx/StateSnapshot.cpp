#include "StateSnapshot.hpp"

#include "ElementsState.hpp"

namespace Homme {

StateSnapshot::
StateSnapshot(int nelem, bool alloc_ps)
 : num_elems(nelem)
 , v ("v",nelem)
 , vtheta_dp ("vtheta_dp",nelem)
 , dp3d ("dp3d",nelem)
 , w_i ("w",nelem)
 , phinh_i ("phinh",nelem)
{
  if (alloc_ps)
    ps_v = decltype(ps_v)("ps",nelem);
}

void StateSnapshot::deep_copy (const StateSnapshot& src)
{
  EKAT_REQUIRE_MSG ((ps_v.data()!=nullptr)==(src.ps_v.data()!=nullptr),
      "Error! Src and tgt StateSnapshot must agree on whether to use ps_v.\n");
  Kokkos::deep_copy(v,src.v);
  Kokkos::deep_copy(vtheta_dp,src.vtheta_dp);
  Kokkos::deep_copy(dp3d,src.dp3d);
  Kokkos::deep_copy(w_i,src.w_i);
  Kokkos::deep_copy(phinh_i,src.phinh_i);

  if (ps_v.data()!=nullptr)
    Kokkos::deep_copy(ps_v,src.ps_v);
}

StateSnapshot StateSnapshot::clone (const bool deep_copy) const
{
  auto copy = StateSnapshot(num_elems,ps_v.data()!=nullptr);
  if (deep_copy)
    copy.deep_copy(*this);
  return copy;
}

void StateSnapshot::randomize (const int seed, const Real p_max, const Real p0, const Real hyai0)
{
  constexpr auto A = Kokkos::ALL;

  // Recycle ElementsStateST implementation
  ElementsStateST<Real> st;
  st.init(num_elems);
  st.randomize(seed,p_max,p0,hyai0);

  Kokkos::deep_copy(v,Kokkos::subview(st.m_v,A,0,A,A,A,A));
  Kokkos::deep_copy(vtheta_dp,Kokkos::subview(st.m_vtheta_dp,A,0,A,A,A));
  Kokkos::deep_copy(w_i,Kokkos::subview(st.m_w_i,A,0,A,A,A));
  Kokkos::deep_copy(dp3d,Kokkos::subview(st.m_dp3d,A,0,A,A,A));
  Kokkos::deep_copy(phinh_i,Kokkos::subview(st.m_phinh_i,A,0,A,A,A));
  if (ps_v.data()!=nullptr)
    Kokkos::deep_copy(ps_v,Kokkos::subview(st.m_ps_v,A,0,A,A));
}

void StateSnapshot::
randomize (const int seed, const Real p_max, const Real p0, const Real hyai0,
           const ExecViewUnmanaged<const Real*[NP][NP]>& phis)
{
  constexpr auto A = Kokkos::ALL;

  // Recycle ElementsStateST implementation
  ElementsStateST<Real> st;
  st.init(num_elems);
  st.randomize(seed,p_max,p0,hyai0,phis);

  Kokkos::deep_copy(v,Kokkos::subview(st.m_v,A,0,A,A,A,A));
  Kokkos::deep_copy(vtheta_dp,Kokkos::subview(st.m_vtheta_dp,A,0,A,A,A));
  Kokkos::deep_copy(w_i,Kokkos::subview(st.m_w_i,A,0,A,A,A));
  Kokkos::deep_copy(dp3d,Kokkos::subview(st.m_dp3d,A,0,A,A,A));
  Kokkos::deep_copy(phinh_i,Kokkos::subview(st.m_phinh_i,A,0,A,A,A));
  if (ps_v.data()!=nullptr)
    Kokkos::deep_copy(ps_v,Kokkos::subview(st.m_ps_v,A,0,A,A));
}

} // namespace Homme
