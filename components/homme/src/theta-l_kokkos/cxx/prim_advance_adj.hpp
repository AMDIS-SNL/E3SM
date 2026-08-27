#ifndef HOMMEXX_PRIM_ADVANCE_ADJ_HPP
#define HOMMEXX_PRIM_ADVANCE_ADJ_HPP

#include "mpi/BoundaryExchange.hpp"
#include "StateSnapshot.hpp"
#include "Types.hpp"

#include <memory>

namespace Homme {

std::shared_ptr<BoundaryExchangeST<Real>> create_adj_bex (StateSnapshot& adj_state);

void ttype5_imex_adjoint(const Real dt_dyn,
                         const Real eta_ave_w,
                         StateSnapshot& adj_state);

void ttype10_imex_adjoint(const Real dt_dyn,
                          const Real eta_ave_w,
                          StateSnapshot& adj_state);
} // namespace Homme

#endif // HOMMEXX_PRIM_ADVANCE_ADJ_HPP
