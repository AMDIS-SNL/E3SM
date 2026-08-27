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

void StateSnapshot::zero ()
{
  Kokkos::deep_copy(v,0);
  Kokkos::deep_copy(vtheta_dp,0);
  Kokkos::deep_copy(dp3d,0);
  Kokkos::deep_copy(w_i,0);
  Kokkos::deep_copy(phinh_i,0);
  if (ps_v.data()!=nullptr) {
    Kokkos::deep_copy(ps_v,0);
  }
}

void StateSnapshot::add (const StateSnapshot& x)
{
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV});
  auto p4_int = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV_P});

  auto lhs_v               = v        ;
  auto lhs_vtheta_dp       = vtheta_dp;
  auto lhs_dp3d            = dp3d     ;
  auto lhs_w_i             = w_i      ;
  auto lhs_phinh_i         = phinh_i  ;
  auto lhs_ps_v            = ps_v     ;

  auto rhs_v               = x.v        ;
  auto rhs_vtheta_dp       = x.vtheta_dp;
  auto rhs_dp3d            = x.dp3d     ;
  auto rhs_w_i             = x.w_i      ;
  auto rhs_phinh_i         = x.phinh_i  ;
  auto rhs_ps_v            = x.ps_v     ;

  auto add_mid = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    lhs_v(ie,0,ipt,jpt,k) += rhs_v(ie,0,ipt,jpt,k);
    lhs_v(ie,1,ipt,jpt,k) += rhs_v(ie,1,ipt,jpt,k);
    lhs_vtheta_dp(ie,ipt,jpt,k) += rhs_vtheta_dp(ie,ipt,jpt,k);
    lhs_dp3d     (ie,ipt,jpt,k) += rhs_dp3d     (ie,ipt,jpt,k);
    if (lhs_ps_v.data()!=nullptr and k==0)
      lhs_ps_v(ie,ipt,jpt) += rhs_ps_v(ie,ipt,jpt);
  };
  auto add_int = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    lhs_w_i      (ie,ipt,jpt,k) += rhs_w_i    (ie,ipt,jpt,k);
    lhs_phinh_i  (ie,ipt,jpt,k) += rhs_phinh_i(ie,ipt,jpt,k);
  };

  Kokkos::parallel_for(p4_mid,add_mid);
  Kokkos::parallel_for(p4_int,add_int);
}

void StateSnapshot::add (const StateSnapshot& x, const Real alpha, const Real beta)
{
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV});
  auto p4_int = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV_P});

  auto lhs_v               = v        ;
  auto lhs_vtheta_dp       = vtheta_dp;
  auto lhs_dp3d            = dp3d     ;
  auto lhs_w_i             = w_i      ;
  auto lhs_phinh_i         = phinh_i  ;
  auto lhs_ps_v            = ps_v     ;

  auto rhs_v               = x.v        ;
  auto rhs_vtheta_dp       = x.vtheta_dp;
  auto rhs_dp3d            = x.dp3d     ;
  auto rhs_w_i             = x.w_i      ;
  auto rhs_phinh_i         = x.phinh_i  ;
  auto rhs_ps_v            = x.ps_v     ;

  auto add_mid = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    lhs_v(ie,0,ipt,jpt,k) *= beta;
    lhs_v(ie,1,ipt,jpt,k) *= beta;
    lhs_vtheta_dp(ie,ipt,jpt,k) *= beta;
    lhs_dp3d     (ie,ipt,jpt,k) *= beta;
    if (lhs_ps_v.data()!=nullptr and k==0)
      lhs_ps_v(ie,ipt,jpt) *= beta;

    lhs_v(ie,0,ipt,jpt,k) += alpha*rhs_v(ie,0,ipt,jpt,k);
    lhs_v(ie,1,ipt,jpt,k) += alpha*rhs_v(ie,1,ipt,jpt,k);
    lhs_vtheta_dp(ie,ipt,jpt,k) += alpha*rhs_vtheta_dp(ie,ipt,jpt,k);
    lhs_dp3d     (ie,ipt,jpt,k) += alpha*rhs_dp3d     (ie,ipt,jpt,k);
    if (lhs_ps_v.data()!=nullptr and k==0)
      lhs_ps_v(ie,ipt,jpt) += alpha*rhs_ps_v(ie,ipt,jpt);
  };
  auto add_int = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    lhs_w_i      (ie,ipt,jpt,k) *= beta;
    lhs_phinh_i  (ie,ipt,jpt,k) *= beta;

    lhs_w_i      (ie,ipt,jpt,k) += alpha*rhs_w_i    (ie,ipt,jpt,k);
    lhs_phinh_i  (ie,ipt,jpt,k) += alpha*rhs_phinh_i(ie,ipt,jpt,k);
  };

  Kokkos::parallel_for(p4_mid,add_mid);
  Kokkos::parallel_for(p4_int,add_int);
}

void StateSnapshot::add_weighted (const StateSnapshot& x,
                                  const ExecViewManaged<Real*[NP][NP]>& weight,
                                  const Real scale)
{
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV});
  auto p4_int = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV_P});

  auto lhs_v               = v        ;
  auto lhs_vtheta_dp       = vtheta_dp;
  auto lhs_dp3d            = dp3d     ;
  auto lhs_w_i             = w_i      ;
  auto lhs_phinh_i         = phinh_i  ;
  auto lhs_ps_v            = ps_v     ;

  auto rhs_v               = x.v        ;
  auto rhs_vtheta_dp       = x.vtheta_dp;
  auto rhs_dp3d            = x.dp3d     ;
  auto rhs_w_i             = x.w_i      ;
  auto rhs_phinh_i         = x.phinh_i  ;
  auto rhs_ps_v            = x.ps_v     ;

  auto w = weight;
  auto add_mid = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    Real s = scale * w(ie,ipt,jpt);
    lhs_v(ie,0,ipt,jpt,k) += s * rhs_v(ie,0,ipt,jpt,k);
    lhs_v(ie,1,ipt,jpt,k) += s * rhs_v(ie,1,ipt,jpt,k);
    lhs_vtheta_dp(ie,ipt,jpt,k) += s * rhs_vtheta_dp(ie,ipt,jpt,k);
    lhs_dp3d     (ie,ipt,jpt,k) += s * rhs_dp3d     (ie,ipt,jpt,k);
    if (lhs_ps_v.data()!=nullptr and k==0)
      lhs_ps_v(ie,ipt,jpt) += s * rhs_ps_v(ie,ipt,jpt);
  };
  auto add_int = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    Real s = scale * w(ie,ipt,jpt);
    lhs_w_i      (ie,ipt,jpt,k) += s * rhs_w_i    (ie,ipt,jpt,k);
    lhs_phinh_i  (ie,ipt,jpt,k) += s * rhs_phinh_i(ie,ipt,jpt,k);
  };

  Kokkos::parallel_for(p4_mid,add_mid);
  Kokkos::parallel_for(p4_int,add_int);
}

void StateSnapshot::add_weighted (const StateSnapshot& x,
                                  const ExecViewManaged<Real*[NP][NP]>& weight,
                                  const Real scale,
                                  const Real beta)
{
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV});
  auto p4_int = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV_P});

  auto lhs_v               = v        ;
  auto lhs_vtheta_dp       = vtheta_dp;
  auto lhs_dp3d            = dp3d     ;
  auto lhs_w_i             = w_i      ;
  auto lhs_phinh_i         = phinh_i  ;
  auto lhs_ps_v            = ps_v     ;

  auto rhs_v               = x.v        ;
  auto rhs_vtheta_dp       = x.vtheta_dp;
  auto rhs_dp3d            = x.dp3d     ;
  auto rhs_w_i             = x.w_i      ;
  auto rhs_phinh_i         = x.phinh_i  ;
  auto rhs_ps_v            = x.ps_v     ;

  auto w = weight;
  auto add_mid = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    lhs_v(ie,0,ipt,jpt,k) *= beta;
    lhs_v(ie,1,ipt,jpt,k) *= beta;
    lhs_vtheta_dp(ie,ipt,jpt,k) *= beta;
    lhs_dp3d     (ie,ipt,jpt,k) *= beta;
    if (lhs_ps_v.data()!=nullptr and k==0)
      lhs_ps_v(ie,ipt,jpt) *= beta;

    Real s = scale * w(ie,ipt,jpt);
    lhs_v(ie,0,ipt,jpt,k) += s * rhs_v(ie,0,ipt,jpt,k);
    lhs_v(ie,1,ipt,jpt,k) += s * rhs_v(ie,1,ipt,jpt,k);
    lhs_vtheta_dp(ie,ipt,jpt,k) += s * rhs_vtheta_dp(ie,ipt,jpt,k);
    lhs_dp3d     (ie,ipt,jpt,k) += s * rhs_dp3d     (ie,ipt,jpt,k);
    if (lhs_ps_v.data()!=nullptr and k==0)
      lhs_ps_v(ie,ipt,jpt) += s * rhs_ps_v(ie,ipt,jpt);
  };
  auto add_int = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    lhs_w_i      (ie,ipt,jpt,k) *= beta;
    lhs_phinh_i  (ie,ipt,jpt,k) *= beta;

    Real s = scale * w(ie,ipt,jpt);
    lhs_w_i      (ie,ipt,jpt,k) += s * rhs_w_i    (ie,ipt,jpt,k);
    lhs_phinh_i  (ie,ipt,jpt,k) += s * rhs_phinh_i(ie,ipt,jpt,k);
  };

  Kokkos::parallel_for(p4_mid,add_mid);
  Kokkos::parallel_for(p4_int,add_int);
}

void StateSnapshot::scale (const Real alpha)
{
  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  auto p4_mid = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV});
  auto p4_int = md_range_t({0,0,0,0},{num_elems,NP,NP,NUM_LEV_P});

  auto lhs_v               = v        ;
  auto lhs_vtheta_dp       = vtheta_dp;
  auto lhs_dp3d            = dp3d     ;
  auto lhs_w_i             = w_i      ;
  auto lhs_phinh_i         = phinh_i  ;
  auto lhs_ps_v            = ps_v     ;

  auto scale_mid = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    lhs_v(ie,0,ipt,jpt,k) *= alpha;
    lhs_v(ie,1,ipt,jpt,k) *= alpha;
    lhs_vtheta_dp(ie,ipt,jpt,k) *= alpha;
    lhs_dp3d     (ie,ipt,jpt,k) *= alpha;
    if (lhs_ps_v.data()!=nullptr and k==0)
      lhs_ps_v(ie,ipt,jpt) *= alpha;
  };
  auto scale_int = KOKKOS_LAMBDA (const int ie, const int ipt, const int jpt, const int k) {
    lhs_w_i    (ie,ipt,jpt,k) *= alpha;
    lhs_phinh_i(ie,ipt,jpt,k) *= alpha;
  };

  Kokkos::parallel_for(p4_mid,scale_mid);
  Kokkos::parallel_for(p4_int,scale_int);
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
