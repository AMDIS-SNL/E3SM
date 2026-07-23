#ifndef HOMMEXX_STATE_SNAPSHOT_HPP
#define HOMMEXX_STATE_SNAPSHOT_HPP

#include "utilities/SubviewUtils.hpp"
#include "HybridVCoord.hpp"
#include "Types.hpp"

#include <random>

namespace Homme {

template<typename ST>
struct StateSnapshot {
  using PT = PackType<ST>;

  StateSnapshot (int nelem, bool alloc_ps = false);
  StateSnapshot (const StateSnapshot&) = default;

  StateSnapshot& operator= (const StateSnapshot&) = default;

  void deep_copy (const StateSnapshot& src);

  StateSnapshot clone (const bool deep_copy = false) const;

  void randomize (const int seed, const Real p_max, const Real p0, const Real hyai0);
  void randomize (const int seed, const Real p_max, const Real p0, const Real hyai0,
                  const ExecViewUnmanaged<const Real*[NP][NP]>& phis);

  int num_elems;

  ExecViewManaged<PT * [2][NP][NP][NUM_LEV  ]> v;          // Horizontal velocity
  ExecViewManaged<PT *    [NP][NP][NUM_LEV  ]> vtheta_dp;  // Virtual potential temperature (mass)
  ExecViewManaged<PT *    [NP][NP][NUM_LEV  ]> dp3d;       // Delta p on levels
  ExecViewManaged<PT *    [NP][NP][NUM_LEV_P]> w_i;        // Vertical velocity at interfaces
  ExecViewManaged<PT *    [NP][NP][NUM_LEV_P]> phinh_i;    // Geopotential used by NH model at interfaces
  ExecViewManaged<ST *    [NP][NP]           > ps_v;       // Surface pressure
};

// ================================ IMPLEMENTATION ================================ //
template<typename ST>
StateSnapshot<ST>::
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

template<typename ST>
void StateSnapshot<ST>::deep_copy (const StateSnapshot<ST>& src)
{
  EKAT_REQUIRE_MSG ((ps_v.data()!=nullptr)==(src.ps_v.data()!=nullptr),
      "Error! Src and tgt StateSnapshot<ST> must agree on whether to use ps_v.\n");
  Kokkos::deep_copy(v,src.v);
  Kokkos::deep_copy(vtheta_dp,src.vtheta_dp);
  Kokkos::deep_copy(dp3d,src.dp3d);
  Kokkos::deep_copy(w_i,src.w_i);
  Kokkos::deep_copy(phinh_i,src.phinh_i);

  if (ps_v.data()!=nullptr)
    Kokkos::deep_copy(ps_v,src.ps_v);
}

template<typename ST>
StateSnapshot<ST> StateSnapshot<ST>::clone (const bool deep_copy) const
{
  auto copy = StateSnapshot<ST>(num_elems,ps_v.data()!=nullptr);
  if (deep_copy)
    copy.deep_copy(*this);
  return copy;
}

template<typename ST>
void StateSnapshot<ST>::randomize (const int seed, const Real max_pressure, const Real ps0, const Real hyai0)
{
  // Check elements were inited
  assert (num_elems>0);

  // Check data makes sense
  assert (max_pressure>ps0);
  assert (ps0>0);
  assert (hyai0>=0);

  // Arbitrary minimum value to generate
  constexpr const Real min_value = 0.015625;

  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<Real> random_dist(min_value, 1.0 / min_value);
  std::uniform_real_distribution<Real> pdf_vtheta_dp(100.0, 1000.0);

  genRandArray(v,         engine, random_dist);
  genRandArray(w_i,       engine, random_dist);
  genRandArray(vtheta_dp, engine, pdf_vtheta_dp);
  // Note: to avoid errors in the equation of state, we need phi to be increasing.
  //       Rather than using a constraint (which may call the function many times,
  //       we simply ask that there are no duplicates, then we sort it later.
  auto sort_and_chek = [](const auto& vh)->bool {
    auto* start = vh.data();
    auto* end   = vh.data() + vh.size();
    std::sort(start,end);
    std::reverse(start,end);
    auto it = std::unique(start,end);
    return it==end;
  };
  for (int ie=0; ie<num_elems; ++ie) {
    for (int igp=0; igp<NP; ++igp) {
      for (int jgp=0; jgp<NP; ++ jgp) {
        auto col = ekat::scalarize(Homme::subview(phinh_i,ie,igp,jgp));
        genRandArray(col,engine,random_dist,sort_and_chek);
      }
    }
  }

  // This ensures the pressure in a single column is monotonically increasing
  // and has fixed upper and lower values
  const auto make_pressure_partition = [=](
      ExecViewUnmanaged<PT[NUM_LEV]> pt_dp) {

    auto h_pt_dp = Kokkos::create_mirror_view(pt_dp);
    Kokkos::deep_copy(h_pt_dp,pt_dp);
    ST* data     = reinterpret_cast<ST*>(h_pt_dp.data());
    ST* data_end = data + NUM_PHYSICAL_LEV;

    ST p[NUM_INTERFACE_LEV];
    ST* p_start = &p[0];
    ST* p_end   = p_start+NUM_INTERFACE_LEV;

    for (int i=0; i<NUM_PHYSICAL_LEV; ++i) {
      p[i+1] = data[i];
    }
    p[0] = ps0*hyai0;
    p[NUM_INTERFACE_LEV-1] = max_pressure;

    // Put in monotonic order
    std::sort(p_start, p_end);

    // Check for no repetitions
    if (std::unique(p_start,p_end)!=p_end) {
      return false;
    }

    // Compute dp from p (we assume p(last interface)=max_pressure)
    for (int i=0; i<NUM_PHYSICAL_LEV; ++i) {
      data[i] = p[i+1]-p[i];
    }

    // Check that dp>=dp_min
    const Real min_dp = std::numeric_limits<Real>::epsilon()*1000;
    for (auto it=data; it!=data_end; ++it) {
      if (*it < min_dp) {
        return false;
      }
    }

    // Fill remainder of last vector pack with quiet nan's
    ST* alloc_end = data+NUM_LEV*VECTOR_SIZE;
    for (auto it=data_end; it!=alloc_end; ++it) {
      *it = std::numeric_limits<Real>::quiet_NaN();
    }

    Kokkos::deep_copy(pt_dp,h_pt_dp);

    return true;
  };

  std::uniform_real_distribution<Real> pressure_pdf(min_value, max_pressure);

  for (int ie = 0; ie < num_elems; ++ie) {
    // Because this constraint is difficult to satisfy for all of the tensors,
    // incrementally generate the view
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        ExecViewUnmanaged<PT[NUM_LEV]> pt_dp3d =
            Homme::subview(dp3d, ie, igp, jgp);
        do {
          genRandArray(pt_dp3d, engine, pressure_pdf);
        } while (make_pressure_partition(pt_dp3d)==false);
      }
    }
  }

  // Generate ps_v so that it is equal to sum(dp3d).
  HybridVCoord hvcoord;
  hvcoord.ps0 = ps0;
  hvcoord.hybrid_ai0 = hyai0;
  hvcoord.m_inited = true;
  auto dp = dp3d;
  auto ps = ps_v;
  auto policy = get_default_team_policy<ExecSpace>(num_elems);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team) {
    KernelVariables kv(team);
    hvcoord.compute_ps_ref_from_dp(kv,Homme::subview(dp,kv.ie),
                                      Homme::subview(ps,kv.ie));
  });
  Kokkos::fence();
}

template<typename ST>
void StateSnapshot<ST>::
randomize (const int seed, const Real p_max, const Real p0, const Real hyai0,
           const ExecViewUnmanaged<const Real*[NP][NP]>& phis)
{
  randomize(seed,p_max,p0,hyai0);

  // Re-do phinh so it satisfies phinh_i(bottom)=phis

  // Sanity check
  assert(phis.extent_int(0)==num_elems);

  std::mt19937_64 engine(seed);

  // Note: to avoid errors in the equation of state, we need phi to be increasing.
  //       Rather than using a constraint (which may call the function many times,
  //       we simply ask that there are no duplicates, then we sort it later.
  auto sort_and_chek = [](const auto& vh)->bool {
    auto* start = vh.data();
    auto* end   = vh.data() + NUM_PHYSICAL_LEV;
    std::sort(start,end);
    std::reverse(start,end);
    auto it = std::unique(start,end);
    return it==end;
  };

  auto h_phis = Kokkos::create_mirror_view(phis);
  Kokkos::deep_copy(h_phis,phis);
  auto phinh_h = Kokkos::create_mirror_view(phinh_i);
  Real phi_top = 1e5;
  std::uniform_real_distribution<Real> pdf_dphi (0.7,1.3);
  for (int ie=0; ie<num_elems; ++ie) {
    for (int igp=0; igp<NP; ++igp) {
      for (int jgp=0; jgp<NP; ++ jgp) {
        const Real phis_ij = h_phis(ie,igp,jgp);
        Real dphi_mean = (phi_top - phis_ij) / NUM_PHYSICAL_LEV;
        // Get column
        auto phi_col = ekat::scalarize(Homme::subview(phinh_h,ie,igp,jgp));
        phi_col(NUM_PHYSICAL_LEV) = phis_ij;
        for (int k=NUM_PHYSICAL_LEV; k>0; --k) {
          phi_col(k-1) = phi_col(k) + dphi_mean*pdf_dphi(engine);
        }
      }
    }
  }
  Kokkos::deep_copy(phinh_i,phinh_h);
}

} // namespace Homme

#endif  // HOMMEXX_STATE_SNAPSHOT_HPP
