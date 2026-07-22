#ifndef HOMMEXX_STATE_SNAPSHOT_HPP
#define HOMMEXX_STATE_SNAPSHOT_HPP

#include "Types.hpp"

namespace Homme {

struct StateSnapshot {
  using ST = Real;
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

} // namespace Homme

#endif  // HOMMEXX_STATE_SNAPSHOT_HPP
