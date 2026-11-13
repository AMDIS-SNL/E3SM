#include "DirkAdjoint.hpp"

#include "PhysicalConstants.hpp"
#include "profiling.hpp"

#include <ekat_assert.hpp>
#include <ekat_pack_kokkos.hpp>

#include <utility>

namespace Homme
{

// See DirkAdjoint.hpp for the math. Everything is done column by column, with plain serial loops over
// the levels: the column ops are O(nlev), and there is no horizontal coupling in DIRK.
// The formulas mirror DirkFunctorImpl (compute_pnh_and_exner, pnh_and_exner_from_eos, calc_gwphis and
// calc_jacobian), including the order of the operations. Keep them in sync.

namespace {

constexpr int NL = NUM_PHYSICAL_LEV;    // midpoints
constexpr int NI = NUM_INTERFACE_LEV;   // interfaces (NL+1)

// One column of a state, in plain arrays
struct ColState {
  Real v1[NL], v2[NL], vth[NL], dp[NL];
  Real w[NI], phi[NI];
};

// Scalar views of a snapshot
template<typename SS>
struct SView {
  using V5 = decltype(ekat::scalarize(std::declval<SS&>().v));
  using V4 = decltype(ekat::scalarize(std::declval<SS&>().vtheta_dp));

  explicit SView (SS& s)
   : v   (ekat::scalarize(s.v))
   , vth (ekat::scalarize(s.vtheta_dp))
   , dp  (ekat::scalarize(s.dp3d))
   , w   (ekat::scalarize(s.w_i))
   , phi (ekat::scalarize(s.phinh_i))
  {}

  V5 v;
  V4 vth, dp, w, phi;
};

template<typename SV>
KOKKOS_INLINE_FUNCTION
void load_col (const SV& s, const int ie, const int ip, const int jp, ColState& c)
{
  for (int k=0; k<NL; ++k) {
    c.v1 [k] = s.v  (ie,0,ip,jp,k);
    c.v2 [k] = s.v  (ie,1,ip,jp,k);
    c.vth[k] = s.vth(ie,  ip,jp,k);
    c.dp [k] = s.dp (ie,  ip,jp,k);
  }
  for (int k=0; k<NI; ++k) {
    c.w  [k] = s.w  (ie,ip,jp,k);
    c.phi[k] = s.phi(ie,ip,jp,k);
  }
}

template<typename SV>
KOKKOS_INLINE_FUNCTION
void store_col (const SV& s, const int ie, const int ip, const int jp,
                const Real* v1, const Real* v2, const Real* vth, const Real* dp,
                const Real* w, const Real* phi)
{
  for (int k=0; k<NL; ++k) {
    s.v  (ie,0,ip,jp,k) = v1 [k];
    s.v  (ie,1,ip,jp,k) = v2 [k];
    s.vth(ie,  ip,jp,k) = vth[k];
    s.dp (ie,  ip,jp,k) = dp [k];
  }
  for (int k=0; k<NI; ++k) {
    s.w  (ie,ip,jp,k) = w  [k];
    s.phi(ie,ip,jp,k) = phi[k];
  }
}

// pnh at midpoints (EOS), and D = dpnh_dp at interfaces 0..NL-1 (one-sided at the top)
KOKKOS_INLINE_FUNCTION
void eval_pnh_D (const Real* vth, const Real* dp, const Real* dphi, const Real ptop,
                 Real* pnh, Real* D)
{
  constexpr Real Rgas  = PhysicalConstants::Rgas;
  constexpr Real kappa = PhysicalConstants::kappa;
  constexpr Real p0    = PhysicalConstants::p0;

  for (int m=0; m<NL; ++m) {
    const Real exner = (-Rgas)*vth[m] / dphi[m];
    Real p = exner/p0;
    p = Kokkos::pow(p,1.0/(1.0-kappa));
    p *= p0;
    pnh[m] = p;
  }

  D[0] = 2*(pnh[0] - ptop)/dp[0];
  for (int k=1; k<NL; ++k) {
    D[k] = (pnh[k] - pnh[k-1]) / ((dp[k-1] + dp[k])/2);
  }
}

// Vertical velocity induced by the surface topography, at interfaces 1..NL-1 (gwh[0]=0)
template<typename Hybi>
KOKKOS_INLINE_FUNCTION
void eval_gwh (const ColState& c, const Real gp1, const Real gp2, const Hybi& hybi, Real* gwh)
{
  gwh[0] = 0;
  for (int k=1; k<NL; ++k) {
    const Real den = c.dp[k-1] + c.dp[k];
    const Real v1i = (c.dp[k]*c.v1[k] + c.dp[k-1]*c.v1[k-1]) / den;
    const Real v2i = (c.dp[k]*c.v2[k] + c.dp[k-1]*c.v2[k-1]) / den;
    gwh[k] = (v1i*gp1 + v2i*gp2)*hybi(k);
  }
}

// Vector-Jacobian product of D(vth,dp,dphi): given nu[k] (dJ/dD[k]), computes
// dJ/dphi_diff[m], dJ/dvth[m], dJ/ddp[m], for all midpoints m (overwriting the outputs).
// D[0] = 2*(pnh[0]-ptop)/dp[0],  D[k] = 2*(pnh[k]-pnh[k-1])/(dp[k-1]+dp[k]),
// pnh[m] = p0*(-Rgas*vth[m]/(p0*dphi[m]))^(1/(1-kappa)), so
//   dpnh/dvth = pnh/((1-kappa)*vth),   dpnh/ddphi = -pnh/((1-kappa)*dphi)
KOKKOS_INLINE_FUNCTION
void vjp_D (const Real* vth, const Real* dp, const Real* dphi, const Real* pnh, const Real* D,
            const Real* nu, Real* l_dphi, Real* l_vth, Real* l_dp)
{
  constexpr Real kappa = PhysicalConstants::kappa;
  constexpr Real inv_1mk = 1.0/(1.0-kappa);

  for (int m=0; m<NL; ++m) {
    // c_k = dD[k]/dpnh[k] = 2/dp[0] (k=0), 2/(dp[k-1]+dp[k]) (k>0), and dD[k+1]/dpnh[k] = -c_{k+1}
    const Real cm = m==0 ? 2/dp[0] : 2/(dp[m-1]+dp[m]);
    Real q = nu[m]*cm;
    Real cn = 0;
    if (m<NL-1) {
      cn = 2/(dp[m]+dp[m+1]);
      q -= nu[m+1]*cn;
    }

    l_dphi[m] = q * (-pnh[m]*inv_1mk/dphi[m]);
    l_vth [m] = q * ( pnh[m]*inv_1mk/vth [m]);

    // dD[k]/ddp[k-1] = dD[k]/ddp[k] = -D[k]/(dp[k-1]+dp[k]) = -D[k]*c_k/2, and dD[0]/ddp[0] = -D[0]/dp[0]
    const Real self = m==0 ? -D[0]/dp[0] : -D[m]*cm/2;
    Real acc = nu[m]*self;
    if (m<NL-1) {
      acc += nu[m+1]*(-D[m+1]*cn/2);
    }
    l_dp[m] = acc;
  }
}

// Vector-Jacobian product of gwh(v,dp): given lg[k] (dJ/dgwh[k], k=1..NL-1), ACCUMULATES into
// l_v1, l_v2, l_dp.
template<typename Hybi>
KOKKOS_INLINE_FUNCTION
void vjp_gwh (const ColState& c, const Real gp1, const Real gp2, const Hybi& hybi, const Real* lg,
              Real* l_v1, Real* l_v2, Real* l_dp)
{
  for (int k=1; k<NL; ++k) {
    const Real den = c.dp[k-1] + c.dp[k];
    const Real a_k   = c.dp[k]  /den;
    const Real a_km1 = c.dp[k-1]/den;
    const Real v1i = a_k*c.v1[k] + a_km1*c.v1[k-1];
    const Real v2i = a_k*c.v2[k] + a_km1*c.v2[k-1];
    const Real gk = lg[k]*hybi(k);

    l_v1[k]   += gk*gp1*a_k;
    l_v1[k-1] += gk*gp1*a_km1;
    l_v2[k]   += gk*gp2*a_k;
    l_v2[k-1] += gk*gp2*a_km1;
    l_dp[k]   += gk*(gp1*(c.v1[k]   - v1i) + gp2*(c.v2[k]   - v2i))/den;
    l_dp[k-1] += gk*(gp1*(c.v1[k-1] - v1i) + gp2*(c.v2[k-1] - v2i))/den;
  }
}

template<typename SVin, typename SVout, typename Hybi>
KOKKOS_INLINE_FUNCTION
void column_adjoint (const int ie, const int ip, const int jp,
                     const Real alphadt_nm1, const Real alphadt_n0, const Real dt2,
                     const Real ptop, const Real gp1, const Real gp2, const Hybi& hybi,
                     const SVin& s_nm1, const SVin& s_n0, const SVin& s_pre,
                     const SVin& s_post, const SVin& s_mu,
                     const SVout& o_pre, const SVout& o_n0, const SVout& o_nm1,
                     const bool do_o_n0, const bool do_o_nm1)
{
  constexpr Real g     = PhysicalConstants::g;
  constexpr Real kappa = PhysicalConstants::kappa;

  // Read the seed first, so that o_pre can alias s_mu
  ColState mu;
  load_col(s_mu,ie,ip,jp,mu);

  ColState pre;
  load_col(s_pre,ie,ip,jp,pre);

  // The converged solution
  Real ws[NI];
  for (int k=0; k<NL; ++k) {
    ws[k] = s_post.w(ie,ip,jp,k);
  }

  // ---- Rebuild the inputs of the Newton solve: w0, phi0 ---- //
  Real w0[NL], phi0[NI];
  for (int k=0; k<NL; ++k) {
    w0[k] = pre.w[k];
  }
  for (int k=0; k<NI; ++k) {
    phi0[k] = pre.phi[k];
  }

  // Contributions of the n0 and nm1 states (index 0 is nm1, 1 is n0)
  ColState nt[2];
  Real dphi_nt[2][NL], pnh_nt[2][NL], D_nt[2][NL];
  const bool use_nt[2] = { alphadt_nm1!=0, alphadt_n0!=0 };
  const Real dt3[2]    = { alphadt_nm1,    alphadt_n0    };
  for (int t=0; t<2; ++t) {
    if (not use_nt[t]) {
      continue;
    }
    if (t==0) {
      load_col(s_nm1,ie,ip,jp,nt[t]);
    } else {
      load_col(s_n0,ie,ip,jp,nt[t]);
    }
    for (int m=0; m<NL; ++m) {
      dphi_nt[t][m] = nt[t].phi[m+1] - nt[t].phi[m];
    }
    eval_pnh_D(nt[t].vth,nt[t].dp,dphi_nt[t],ptop,pnh_nt[t],D_nt[t]);
    Real gwh[NL];
    eval_gwh(nt[t],gp1,gp2,hybi,gwh);
    for (int k=0; k<NL; ++k) {
      w0[k]   += dt3[t]*g*(D_nt[t][k] - 1);
      phi0[k] += dt3[t]*g*nt[t].w[k] - dt3[t]*gwh[k];
    }
  }

  Real gwh_pre[NL];
  eval_gwh(pre,gp1,gp2,hybi,gwh_pre);
  for (int k=0; k<NL; ++k) {
    phi0[k] -= dt2*gwh_pre[k];
  }

  Real dphi0[NL];
  for (int k=0; k<NL; ++k) {
    dphi0[k] = phi0[k+1] - phi0[k];
  }

  // ---- The state at the converged solution ---- //
  Real dphi[NL];
  for (int k=0; k<NL-1; ++k) {
    dphi[k] = dphi0[k] + dt2*g*(ws[k+1] - ws[k]);
  }
  dphi[NL-1] = dphi0[NL-1] - dt2*g*ws[NL-1];

  Real pnh[NL], D[NL];
  eval_pnh_D(pre.vth,pre.dp,dphi,ptop,pnh,D);

  // The Jacobian dr/dw: the tridiagonal matrix of DirkFunctorImpl::calc_jacobian
  Real dl[NL], d[NL], du[NL];
  const Real a = (dt2*g)*(dt2*g) / (1-kappa);
  {
    const Real b = a/pre.dp[0];
    dl[0] = 0;
    du[0] = 2*b*(pnh[0]/dphi[0]);
    d [0] = 1 - du[0];
  }
  for (int k=1; k<NL-1; ++k) {
    const Real b = 2*a/(pre.dp[k-1] + pre.dp[k]);
    dl[k] = b*(pnh[k-1]/dphi[k-1]);
    du[k] = b*(pnh[k  ]/dphi[k  ]);
    d [k] = 1 - dl[k] - du[k];
  }
  {
    const int k = NL-1;
    const Real b = 2*a/(pre.dp[k-1] + pre.dp[k]);
    dl[k] = b*(pnh[k-1]/dphi[k-1]);
    du[k] = 0;
    d [k] = 1 - dl[k] - b*(pnh[k]/dphi[k]);
  }

  // ---- Adjoint ---- //
  // phi_out = phi0 + dt2*g*w* (levels 0..NL-1), so the seed on w* is
  Real omega[NL];
  for (int k=0; k<NL; ++k) {
    omega[k] = mu.w[k] + dt2*g*mu.phi[k];
  }

  // Solve J^T*lam = omega. J^T is tridiagonal with sub-diagonal du[k-1], diagonal d[k] and
  // super-diagonal dl[k+1]. J is strictly diagonally dominant by rows (and an M-matrix), so there
  // is no need to pivot.
  Real lam[NL];
  {
    Real cp[NL], dpv[NL];
    cp [0] = dl[1]/d[0];
    dpv[0] = omega[0]/d[0];
    for (int k=1; k<NL; ++k) {
      const Real low   = du[k-1];
      const Real denom = d[k] - low*cp[k-1];
      cp [k] = (k<NL-1 ? dl[k+1] : Real(0))/denom;
      dpv[k] = (omega[k] - low*dpv[k-1])/denom;
    }
    lam[NL-1] = dpv[NL-1];
    for (int k=NL-2; k>=0; --k) {
      lam[k] = dpv[k] - cp[k]*lam[k+1];
    }
  }

  // r(w;p) = w - (w0 + g*dt2*(D - 1)), so dr/dw0 = -I, and dJ/dp = -(dr/dp)^T lam:
  //   dJ/dw0 = lam,   dJ/dp = g*dt2 * (dD/dp)^T lam   (for p=vth, dp, dphi0)
  Real nu[NL];
  for (int k=0; k<NL; ++k) {
    nu[k] = g*dt2*lam[k];
  }
  Real l_dphi0[NL], l_vth[NL], l_dp[NL];
  vjp_D(pre.vth,pre.dp,dphi,pnh,D,nu,l_dphi0,l_vth,l_dp);

  // dJ/dphi0: the direct one (phi_out = phi0 + ...), and the one via dphi0 = diff(phi0)
  Real l_phi0[NI];
  for (int k=0; k<NL; ++k) {
    l_phi0[k] = mu.phi[k];
  }
  l_phi0[NL] = 0;
  for (int k=0; k<NI; ++k) {
    l_phi0[k] += (k>0 ? l_dphi0[k-1] : Real(0)) - (k<NL ? l_dphi0[k] : Real(0));
  }

  // ---- The adjoint of the np1 state before the stage ---- //
  {
    Real o_v1[NL], o_v2[NL], o_vth[NL], o_dp[NL], o_w[NI], o_phi[NI];
    for (int k=0; k<NL; ++k) {
      // v, vtheta_dp, dp3d: identity block, plus the dependency of the solve on vth, dp (and v, dp via gwh, below)
      o_v1 [k] = mu.v1[k];
      o_v2 [k] = mu.v2[k];
      o_vth[k] = mu.vth[k] + l_vth[k];
      o_dp [k] = mu.dp [k] + l_dp [k];
      o_w  [k] = lam[k];     // w0 = w(np1)
    }
    o_w[NL] = mu.w[NL];      // The surface w is not touched by the stage
    for (int k=0; k<NI; ++k) {
      o_phi[k] = l_phi0[k];  // (the output surface phi is a constant (phis): nothing from mu.phi[NL])
    }

    // phi0 -= dt2*gwh(v,dp)
    Real lg[NL];
    lg[0] = 0;
    for (int k=1; k<NL; ++k) {
      lg[k] = -dt2*l_phi0[k];
    }
    vjp_gwh(pre,gp1,gp2,hybi,lg,o_v1,o_v2,o_dp);

    store_col(o_pre,ie,ip,jp,o_v1,o_v2,o_vth,o_dp,o_w,o_phi);
  }

  // ---- The adjoint of the n0 and nm1 states ---- //
  for (int t=0; t<2; ++t) {
    const bool do_out = t==0 ? do_o_nm1 : do_o_n0;
    if (not use_nt[t] or not do_out) {
      continue;
    }

    // w0 += dt3*g*(D(nt)-1)
    Real nu_t[NL];
    for (int k=0; k<NL; ++k) {
      nu_t[k] = dt3[t]*g*lam[k];
    }
    Real l_dphi_t[NL], l_vth_t[NL], l_dp_t[NL];
    vjp_D(nt[t].vth,nt[t].dp,dphi_nt[t],pnh_nt[t],D_nt[t],nu_t,l_dphi_t,l_vth_t,l_dp_t);

    Real o_v1[NL], o_v2[NL], o_vth[NL], o_dp[NL], o_w[NI], o_phi[NI];
    for (int k=0; k<NL; ++k) {
      o_v1 [k] = 0;
      o_v2 [k] = 0;
      o_vth[k] = l_vth_t[k];
      o_dp [k] = l_dp_t [k];
      // phi0 += dt3*g*w(nt) - dt3*gwh(nt)
      o_w  [k] = dt3[t]*g*l_phi0[k];
    }
    o_w[NL] = 0;
    // dphi(nt) = diff(phi(nt))
    for (int k=0; k<NI; ++k) {
      o_phi[k] = (k>0 ? l_dphi_t[k-1] : Real(0)) - (k<NL ? l_dphi_t[k] : Real(0));
    }

    Real lg[NL];
    lg[0] = 0;
    for (int k=1; k<NL; ++k) {
      lg[k] = -dt3[t]*l_phi0[k];
    }
    vjp_gwh(nt[t],gp1,gp2,hybi,lg,o_v1,o_v2,o_dp);

    store_col(t==0 ? o_nm1 : o_n0,ie,ip,jp,o_v1,o_v2,o_vth,o_dp,o_w,o_phi);
  }
}

} // anonymous namespace

void dirk_stage_adjoint_ift (const Real alphadt_nm1, const Real alphadt_n0, const Real dt2,
                             const ElementsGeometry& geo, const HybridVCoord& hvcoord,
                             const StateSnapshot* s_nm1, const StateSnapshot* s_n0,
                             const StateSnapshot& s_pre, const StateSnapshot& s_post,
                             const StateSnapshot& mu,
                             StateSnapshot& out_pre,
                             StateSnapshot* out_n0, StateSnapshot* out_nm1)
{
  EKAT_REQUIRE_MSG (alphadt_nm1==0 or s_nm1!=nullptr,
      "[dirk_stage_adjoint_ift] Error! The nm1 state is needed if alphadt_nm1!=0.\n");
  EKAT_REQUIRE_MSG (alphadt_n0==0 or s_n0!=nullptr,
      "[dirk_stage_adjoint_ift] Error! The n0 state is needed if alphadt_n0!=0.\n");
  const int nelem = s_pre.num_elems;
  EKAT_REQUIRE_MSG (s_post.num_elems==nelem and mu.num_elems==nelem and out_pre.num_elems==nelem and
                    geo.num_elems()==nelem,
      "[dirk_stage_adjoint_ift] Error! Inconsistent number of elements.\n");

  GPTLstart("dirk_stage_adjoint_ift");

  // If a state/output is not needed, pass in another valid one: it won't be read/written
  const StateSnapshot& snm1 = s_nm1 ? *s_nm1 : s_pre;
  const StateSnapshot& sn0  = s_n0  ? *s_n0  : s_pre;
  StateSnapshot& onm1 = out_nm1 ? *out_nm1 : out_pre;
  StateSnapshot& on0  = out_n0  ? *out_n0  : out_pre;
  const bool do_o_nm1 = out_nm1!=nullptr;
  const bool do_o_n0  = out_n0!=nullptr;

  using SVin  = SView<const StateSnapshot>;
  using SVout = SView<StateSnapshot>;
  const SVin v_nm1(snm1), v_n0(sn0), v_pre(s_pre), v_post(s_post), v_mu(mu);
  const SVout vo_pre(out_pre), vo_n0(on0), vo_nm1(onm1);

  const auto gradphis = geo.m_gradphis;
  const auto hybi     = hvcoord.hybrid_bi;
  const Real ptop     = hvcoord.hybrid_ai0*hvcoord.ps0;

  using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>;
  const md_range_t policy({0,0,0},{nelem,NP,NP});
  Kokkos::parallel_for("dirk_stage_adjoint_ift", policy,
                       KOKKOS_LAMBDA (const int ie, const int ip, const int jp) {
    column_adjoint(ie,ip,jp,alphadt_nm1,alphadt_n0,dt2,
                   ptop,gradphis(ie,0,ip,jp),gradphis(ie,1,ip,jp),hybi,
                   v_nm1,v_n0,v_pre,v_post,v_mu,
                   vo_pre,vo_n0,vo_nm1,do_o_n0,do_o_nm1);
  });
  Kokkos::fence();

  GPTLstop("dirk_stage_adjoint_ift");
}

} // namespace Homme
