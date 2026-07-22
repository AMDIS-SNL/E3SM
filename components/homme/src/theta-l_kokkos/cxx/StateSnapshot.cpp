#include "StateSnapshot.hpp"

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

} // namespace Homme
