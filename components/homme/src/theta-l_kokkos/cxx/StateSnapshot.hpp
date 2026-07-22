#ifndef HOMMEXX_STATE_SNAPSHOT_HPP
#define HOMMEXX_STATE_SNAPSHOT_HPP

#include "Types.hpp"

namespace Homme {

struct StateSnapshot {
  using ST = Real;
  using PT = PackType<ST>;

  StateSnapshot (int nelem, bool alloc_ps = false);

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
