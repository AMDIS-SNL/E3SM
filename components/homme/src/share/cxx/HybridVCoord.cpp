/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "HybridVCoord.hpp"
#include "ColumnOps.hpp"
#include "ErrorDefs.hpp"
#include "PhysicalConstants.hpp"
#include "utilities/TestUtils.hpp"
#include "utilities/ViewUtils.hpp"

#include <random>
#include <vector>

namespace Homme
{

void HybridVCoord::init(const Real ps0_in,
                   CRCPtr hybrid_am_ptr,
                   CRCPtr hybrid_ai_ptr,
                   CRCPtr hybrid_bm_ptr,
                   CRCPtr hybrid_bi_ptr)
{
  // Sanity checks
  assert(hybrid_am_ptr!=nullptr);
  assert(hybrid_ai_ptr!=nullptr);
  assert(hybrid_bm_ptr!=nullptr);
  assert(hybrid_bi_ptr!=nullptr);

  ps0 = ps0_in;

  // Wrap pointers in unmanaged views
  HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]>  hyam_h(hybrid_am_ptr);
  HostViewUnmanaged<const Real[NUM_PHYSICAL_LEV]>  hybm_h(hybrid_bm_ptr);
  HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> hyai_h(hybrid_ai_ptr);
  HostViewUnmanaged<const Real[NUM_INTERFACE_LEV]> hybi_h(hybrid_bi_ptr);

  // Create member views
  hybrid_ai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Hybrid coordinates; coefficient A_interfaces");
  hybrid_bi = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Hybrid coordinates; coefficient B_interfaces");
  hybrid_ai_packed = ExecViewManaged<RPack[NUM_LEV_P]>("Hybrid coordinates; coefficient A_interfaces");
  hybrid_bi_packed = ExecViewManaged<RPack[NUM_LEV_P]>("Hybrid coordinates; coefficient B_interfaces");
  hybrid_am = ExecViewManaged<RPack[NUM_LEV]>("Hybrid coordinates; coefficient A_midpoints");
  hybrid_bm = ExecViewManaged<RPack[NUM_LEV]>("Hybrid coordinates; coefficient B_midpoints");

  // Init interface views
  Kokkos::deep_copy(hybrid_ai, hyai_h);
  Kokkos::deep_copy(hybrid_bi, hybi_h);

  auto hyai_p_h = Kokkos::create_mirror_view(hybrid_ai_packed);
  auto hybi_p_h = Kokkos::create_mirror_view(hybrid_bi_packed);
  for (int i=0; i<NUM_INTERFACE_LEV; ++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyai_p_h(ilev)[ivec] = hyai_h(i);
    hybi_p_h(ilev)[ivec] = hybi_h(i);
  }
  Kokkos::deep_copy(hybrid_ai_packed,hyai_p_h);
  Kokkos::deep_copy(hybrid_bi_packed,hybi_p_h);

  // Init midpoints views
  auto hybrid_am_h = Kokkos::create_mirror_view(ekat::scalarize(hybrid_am));
  auto hybrid_bm_h = Kokkos::create_mirror_view(ekat::scalarize(hybrid_bm));

  for (int i=0; i<NUM_PHYSICAL_LEV; ++i) {
    hybrid_am_h(i) = hyam_h(i);
    hybrid_bm_h(i) = hybm_h(i);
  }
  Kokkos::deep_copy(hybrid_am,hybrid_am_h);
  Kokkos::deep_copy(hybrid_bm,hybrid_bm_h);

  // i don't think this saves us much now
  {
    // Only hybrid_ai(0) is needed.
    hybrid_ai0 = hybrid_ai_ptr[0];
  }

  // Compute delta_A and delta_B
  compute_deltas();
  // Compute Eta=A+B
  compute_eta();

  m_inited = true;
}

void HybridVCoord::random_init(int seed)
{
  // Create views
  hybrid_ai_packed = ExecViewManaged<RPack[NUM_LEV_P]>("Hybrid coordinates; coefficient A_interfaces");
  hybrid_bi_packed = ExecViewManaged<RPack[NUM_LEV_P]>("Hybrid coordinates; coefficient B_interfaces");

  hybrid_ai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Hybrid coordinates; coefficient A_interfaces");
  hybrid_bi = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Hybrid coordinates; coefficient B_interfaces");

  hybrid_am = ExecViewManaged<RPack[NUM_LEV]>("Hybrid a_interface coefs");
  hybrid_bm = ExecViewManaged<RPack[NUM_LEV]>("Hybrid b_interface coefs");

  std::mt19937_64 engine(seed);
  ps0 = 1.0;

  using urd = std::uniform_real_distribution<Real>;

  // eta=hybrid_a+hybrid_b increasing ensures p is monotonic (sufficient, not necessary cond).
  // Also, hybrid_a and hybrid_b must be between 0 and 1.
  // So easy way: generate uniform eta=a+b in [eta_min,1], with eta_min>0.
  // That means eta(i) = i*delta_eta, with delta_eta = (1-eta_min)/NUM_PHYSICAL_LEV.
  // Then pick a = c*b, with c<1 (say c=0.1). Then eta = b+c*b=(c+1)b,
  // so b(i) = i*delta_eta/(c+1) and a(i) = c*i*delta_eta/(c+1).
  // To add randomness, perturb sligthly: a(i)=a(i)+da and b(i)=b(i)+db,
  // without violating eta(i+1)>eta(i). For instance, let da+db<delta_eta/2,
  // which is true if da<delta_eta/4 and db<delta_eta/4.

  // This somewhat odd expression ensures that deta (below) is smaller
  // than eta_min, so that eta_min + eps*deta is always positive
  constexpr Real eta_min = (1.0/NUM_PHYSICAL_LEV)+0.001;

  // The proportionality constant between a and b.
  Real c;
  genRandArray(&c,1,engine,urd(0.0001,0.25));

  // The eta uniform grid spacing
  Real deta = (1-eta_min)/NUM_PHYSICAL_LEV;
  Real perturbation = deta/5;

  // Interface coordinates
  auto hyai_h = Kokkos::create_mirror_view(hybrid_ai);
  auto hybi_h = Kokkos::create_mirror_view(hybrid_bi);
  for (int i=0; i<NUM_INTERFACE_LEV; ++i) {
    // Compute eta
    Real eta_i = eta_min+deta*i;

    // Create b and a
    hybi_h(i) = eta_i/(c+1);
    hyai_h(i) = c*hybi_h(i);

    // Add some randomness to a and b
    Real dab[2];
    Real perturbation_adjusted = std::min(std::min(perturbation,eta_i),std::min(perturbation,1-eta_i));;

    auto perturbation_dist = urd(-perturbation_adjusted,perturbation_adjusted);
    genRandArray(dab,2,engine,perturbation_dist);
    hyai_h(i) += dab[0];
    hybi_h(i) += dab[1];

    // Enforce 0<a,b<1
    if (hyai_h(i)<0) {
      hyai_h(i) = 0;
    }
    if (hybi_h(i)<0) {
      hybi_h(i) = 0;
    }
    if (hyai_h(i)>1) {
      hyai_h(i) = 1;
    }
    if (hybi_h(i)>1) {
      hybi_h(i) = 1;
    }
  }

  // Enforce b(0)=0, a(0)=eta_min, and b(end)=1, a(end)=0;
  hyai_h(0) = eta_min;
  hybi_h(0) = 0;
  hyai_h(NUM_PHYSICAL_LEV) = 0;
  hybi_h(NUM_PHYSICAL_LEV) = 1;

  Kokkos::deep_copy(hybrid_ai, hyai_h);
  Kokkos::deep_copy(hybrid_bi, hybi_h);
  hybrid_ai0 = hyai_h(0);
  
  // Packed hybrid coords at interfaces
  auto hyai_packed_h = Kokkos::create_mirror_view(hybrid_ai_packed);
  auto hybi_packed_h = Kokkos::create_mirror_view(hybrid_bi_packed);
  for (int i=0; i<NUM_INTERFACE_LEV; ++i) {
    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyai_packed_h(ilev)[ivec] = hyai_h(i);
    hybi_packed_h(ilev)[ivec] = hybi_h(i);
  }
  Kokkos::deep_copy(hybrid_ai_packed,hyai_packed_h);
  Kokkos::deep_copy(hybrid_bi_packed,hybi_packed_h);
  
  // Midpoints coordinates
  auto hyam_h = Kokkos::create_mirror_view(hybrid_am);
  auto hybm_h = Kokkos::create_mirror_view(hybrid_bm);
  for (int i=0; i<NUM_PHYSICAL_LEV; ++i) {
    // Safety check: a+b should be monotone here
    Real prev = ADValue(hyai_h(i) + hybi_h(i));   
    Real next = ADValue(hyai_h(i+1) + hybi_h(i+1));

    EKAT_REQUIRE_MSG(next>prev,"Error! hybrid_a+hybrid_b is not increasing.\n");

    int ilev = i / VECTOR_SIZE;
    int ivec = i % VECTOR_SIZE;
    hyam_h(ilev)[ivec] = (hyai_h(i) + hyai_h(i+1))/2.0;
    hybm_h(ilev)[ivec] = (hybi_h(i) + hybi_h(i+1))/2.0;
  }

  Kokkos::deep_copy(hybrid_am, hyam_h);
  Kokkos::deep_copy(hybrid_bm, hybm_h);

  // Derived quantities
  compute_deltas();
  compute_eta();

  m_inited = true;
}

void HybridVCoord::smooth_init (const Real ps0_in)
{
  constexpr int n  = NUM_INTERFACE_LEV;
  constexpr int nh = (n+1)/2;  // Number of interfaces with b>0 (the lower half)
  constexpr int nh0 = n-nh;    // First interface with b>0
  static_assert (nh>1, "Error! Too few levels for smooth_init.\n");

  constexpr Real eta_top = 1e-4;
  constexpr Real b_min = 0.02;

  std::vector<Real> ai(n), bi(n,0.0), am(NUM_PHYSICAL_LEV), bm(NUM_PHYSICAL_LEV);

  // b: 0 in the upper half, then linearly growing from b_min to 1
  for (int i=0; i<nh; ++i) {
    const Real a = Real(i)/(nh-1);
    bi[nh0+i] = (1-a)*b_min + a;
  }

  // eta=a+b is uniform in [eta_top,1], so that all layers have the same thickness
  for (int i=0; i<n; ++i) {
    const Real a = Real(i)/(n-1);
    const Real eta = (1-a)*eta_top + a;
    ai[i] = eta - bi[i];
    EKAT_REQUIRE_MSG (ai[i]>=0, "Error! Negative hybrid_a coefficient in smooth_init.\n");
  }

  for (int i=0; i<NUM_PHYSICAL_LEV; ++i) {
    am[i] = (ai[i] + ai[i+1])/2;
    bm[i] = (bi[i] + bi[i+1])/2;
  }

  init(ps0_in, am.data(), ai.data(), bm.data(), bi.data());
}

void HybridVCoord::compute_deltas ()
{
  // This is obviously not for speed (tiny amount of work, and only at setup),
  // but rather to delegate the logic to a single place (namely ColumnOps),
  // so that we avoid making mistakes by replicating the same algorithm
  hybrid_ai_delta = ExecViewManaged<RPack[NUM_LEV]>("delta hyai");
  hybrid_bi_delta = ExecViewManaged<RPack[NUM_LEV]>("delta hybi");
  dp0 = ExecViewManaged<RPack[NUM_LEV]>("dp0");

  // Create local copies, to avoid issue of lambda on GPU
  auto hyai = hybrid_ai_packed;
  auto hybi = hybrid_bi_packed;
  auto dhyai = hybrid_ai_delta;
  auto dhybi = hybrid_bi_delta;
  auto ldp0 = dp0;
  auto lps0 = ps0;
  auto policy = Homme::get_default_team_policy<ExecSpace>(1);
  Kokkos::parallel_for("[HybridVCoord::compute_deltas]",policy,
                       KOKKOS_LAMBDA(const TeamMember& team) {
    KernelVariables kv(team);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,1),[&](int /*unused*/){
      ColumnOps::compute_midpoint_delta(kv,hyai,dhyai);
      ColumnOps::compute_midpoint_delta(kv,hybi,dhybi);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        ldp0(ilev) = dhyai(ilev) * lps0 + dhybi(ilev) * lps0;
      });
    });
  });
}

void HybridVCoord::compute_eta ()
{
  // Create device views
  etai = ExecViewManaged<Real[NUM_INTERFACE_LEV]>("Eta coordinate at interfaces");
  etam = ExecViewManaged<RPack[NUM_LEV]>("Eta coordinate at midpoints");
  exner0 = ExecViewManaged<RPack[NUM_LEV]>("exner0");

  // Local copies, to avoid issues on GPU when accessing this-> members
  auto l_etai = etai;
  auto l_etam = etam;
  auto l_exner0 = exner0;
  auto l_hybrid_am = hybrid_am;
  auto l_hybrid_bm = hybrid_bm;
  auto l_hybrid_ai = hybrid_ai;
  auto l_hybrid_bi = hybrid_bi;
  const auto l_ps0 = ps0;
  const auto p0 = PhysicalConstants::p0;
  const auto kappa = PhysicalConstants::kappa;

  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,NUM_LEV),
                       KOKKOS_LAMBDA(const int& ilev){
    l_etam(ilev) = l_hybrid_am(ilev) + l_hybrid_bm(ilev);
    l_exner0(ilev) = pow(l_etam(ilev)*l_ps0/p0,kappa);
  });
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,NUM_INTERFACE_LEV),
                       KOKKOS_LAMBDA(const int& ilev){
    l_etai(ilev) = l_hybrid_ai(ilev) + l_hybrid_bi(ilev);
  });
}

} // namespace Homme
