/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMME_TRACERS_DEF_HPP
#define HOMME_TRACERS_DEF_HPP

#include <random>

#include "Tracers.hpp"
#include "Context.hpp"
#include "Elements.hpp"
#include "SimulationParams.hpp"

#include "utilities/SyncUtils.hpp"
#include "utilities/TestUtils.hpp"

namespace Homme {

template<typename ST>
TracersST<ST>::TracersST(const int num_elems, const int num_tracers)
{
  init (num_elems, num_tracers);
}

template<typename ST>
void TracersST<ST>::init(const int num_elems, const int num_tracers)
{
  // Sanity check
  assert(num_elems>0);
  assert(num_tracers>=0);

  ne = num_elems;
  nt = num_tracers;

  qdp = decltype(qdp)("tracers mass", num_elems);
  // Also used in ComposeTransportImplEnhancedTrajectory for communication,
  // where 4 slots are needed.
  qtens_biharmonic = decltype(qtens_biharmonic)(
    "qtens(_biharmonic)", num_elems, std::max(4, num_tracers));
  qlim = decltype(qlim)("qlim", num_elems);

  Q = decltype(Q)("tracers concentration", num_elems,num_tracers);
  fq = decltype(fq)("fq",num_elems,num_tracers);

  m_inited = true;
}

template<typename ST>
void TracersST<ST>::randomize(const int seed, const Real min, const Real max) {
  // Check tracers were inited
  assert (m_inited);

  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<Real> random_dist(min, max);

  genRandArray(qdp, engine, random_dist);
  genRandArray(qtens_biharmonic, engine, random_dist);
  genRandArray(qlim, engine, random_dist);
  genRandArray(fq, engine, random_dist);
  genRandArray(Q, engine, random_dist);
}

template<typename ST>
void TracersST<ST>::pull_qdp(CF90Ptr &state_qdp) {
  HostViewUnmanaged<
      const Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>
  state_qdp_f90(state_qdp, qdp.extent_int(0));
  sync_to_device(state_qdp_f90, qdp);
}

template<typename ST>
void TracersST<ST>::push_qdp(F90Ptr &state_qdp) const {
  HostViewUnmanaged<
      Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>
  state_qdp_f90(state_qdp, qdp.extent_int(0));
  sync_to_host(qdp, state_qdp_f90);
}

template<typename ST>
void TracersST<ST>::push_qdp(F90Ptr &state_qdp, const int src_qdp_tl,
                             const int dst_qdp_tl) const {
  HostViewUnmanaged<
      Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>
  state_qdp_f90(state_qdp, qdp.extent_int(0));
  sync_to_host(qdp, state_qdp_f90, src_qdp_tl, dst_qdp_tl);
}

template<typename ST>
HashType TracersST<ST>::hash (const int tl) const {  
  HashType accum = 0;
  Homme::hash(tl, qdp, NUM_PHYSICAL_LEV, accum);
  Homme::hash(      Q, NUM_PHYSICAL_LEV, accum);
  Homme::hash(     fq, NUM_PHYSICAL_LEV, accum);
  return accum;
}

} // namespace Homme

#endif // HOMME_TRACERS_DEF_HPP
