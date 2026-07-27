#include "prim_advance_adj.hpp"

#include "mpi/MpiBuffersManager.hpp"
#include "SimulationParams.hpp"
#include "Context.hpp"

namespace Homme
{

std::shared_ptr<BoundaryExchangeST<Real>> create_adj_bex (StateSnapshot& adj_state)
{
  auto& c = Context::singleton();
  const auto& params = c.get<SimulationParams>();
  auto& bmm = c.get<MpiBuffersManagerMap>();

  auto be = std::make_shared<BoundaryExchangeST<Real>>();
  be->m_label = std::string("AdjState");
  be->set_buffers_manager(bmm[MPI_EXCHANGE]);
  be->m_diagnostics_level = params.internal_diagnostics_level;
  if (params.theta_hydrostatic_mode) {
    be->set_num_fields(0,0,4);
  } else {
    be->set_num_fields(0,0,4,2);
  }

  be->register_field(adj_state.v,2,0);
  be->register_field(adj_state.vtheta_dp);
  be->register_field(adj_state.dp3d);
  if (!params.theta_hydrostatic_mode) {
    // Note: phinh_i at the surface (last level) is constant, so it doesn't *need* bex.
    //       If bex(constant)=constant, we might just do it. This would not eliminate
    //       the need for halo-exchange of interface-based quantities though, since
    //       we would still need to exchange w_i.
    be->register_field(adj_state.w_i);
    be->register_field(adj_state.phinh_i);
  }
  be->registration_completed();

  return be;
}

} // namespace Homme
