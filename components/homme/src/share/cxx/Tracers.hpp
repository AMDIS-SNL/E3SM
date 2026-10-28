/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_TRACERS_HPP
#define HOMMEXX_TRACERS_HPP

#include "Types.hpp"
#include "utilities/Hash.hpp"

namespace Homme {

template<typename ST>
struct TracersST
{
  using PT = PackType<ST>;

  TracersST() : m_inited(false) {}
  TracersST(const int num_elems, const int num_tracers);

  void init (const int num_elems, const int num_tracers);

  void randomize (const int seed, const Real min = -1.0, const Real max = 1.0);

  void pull_qdp(CF90Ptr &state_qdp);
  void push_qdp(F90Ptr &state_qdp) const;
  // Push a single qdp time level: source level src_qdp_tl -> F90 level dst_qdp_tl.
  // The two differ in general, because the qdp levels are rotated before the
  // vertical remap while the dynamics levels are rotated after it.
  void push_qdp(F90Ptr &state_qdp, const int src_qdp_tl, const int dst_qdp_tl) const;

  KOKKOS_INLINE_FUNCTION
  int num_tracers() const {
    return nt;
  }

  int num_elems () const {
    return ne;
  }

  bool inited () const { return m_inited; }

  ExecViewManaged<PT*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]> qdp;
  ExecViewManaged<PT**[NP][NP][NUM_LEV]>                    qtens_biharmonic; // Also doubles as just qtens.
  ExecViewManaged<PT*[QSIZE_D][2][NUM_LEV]>                 qlim;
  ExecViewManaged<PT**[NP][NP][NUM_LEV]>                    Q;
  ExecViewManaged<PT**[NP][NP][NUM_LEV]>                    fq;

  HashType hash(const int qdp_time_level) const;

  // Copy values from one ElementStateST struct to another. All derivs get set to 0.
  // NOTE: only imports Q and qdp (for now)
  template<typename RST>
  void import_values (const TracersST<RST>& rhs, int tl);

private:
  int nt;
  int ne;
  bool m_inited;
};

using Tracers = TracersST<ScalarValue>;

// Copy values from one ElementStateST struct to another. All derivs get set to 0.
template<typename ST>
template<typename RST>
void TracersST<ST>::import_values (const TracersST<RST>& rhs, int tl)
{
  if constexpr (std::is_same_v<ST,RST>) {
    const void* lhs_ptr = this;
    const void* rhs_ptr = &rhs;
    if (lhs_ptr==rhs_ptr)
      return;
  }
  auto lhs_q = ekat::scalarize(Q);
  auto lhs_qdp = ekat::scalarize(qdp);

  auto rhs_q = ekat::scalarize(rhs.Q);
  auto rhs_qdp = ekat::scalarize(rhs.qdp);

  int nlev = NUM_PHYSICAL_LEV;
  auto copy = KOKKOS_LAMBDA(int ie, int iq, int ip, int jp, int k) {
    lhs_qdp(ie,tl,iq,ip,jp,k) = ADValue(rhs_qdp(ie,tl,iq,ip,jp,k));

    lhs_q(ie,iq,ip,jp,k) = ADValue(rhs_q(ie,iq,ip,jp,k));
  };
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>> p({0,0,0,0,0},{ne,nt,NP,NP,nlev});
  Kokkos::parallel_for(p,copy);
}

} // namespace Homme

#endif // HOMMEXX_TRACERS_HPP
