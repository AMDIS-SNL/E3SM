#ifndef HOMMEXX_CHECKPOINT_IO_HPP
#define HOMMEXX_CHECKPOINT_IO_HPP

#include "Hommexx_config.h"

#include "StateSnapshot.hpp"

#include <ekat_comm.hpp>

#include <string>

namespace Homme {

// Identifies one on-disk checkpoint slot in the reverse-mode adjoint sweep's
// coarse-grained checkpoint schedule (see the HOMME adjoint battleplan,
// Step 3): which NN call, which of the M prim_run_subcycle_c calls within
// it, and which boundary within that call the snapshot was taken at (e.g.
// step 0 = call start, one entry per prim_advance_exp call end, and a final
// entry = call end -- the exact schedule is decided by the caller, not by
// this class).
struct CheckpointId {
  int nn_call       = 0;
  int subcycle_call = 0;
  int step          = 0;
};

// On-disk store for StateSnapshot checkpoints, using scorpio's PIOc C API --
// the same underlying library HOMME's existing Fortran restart
// infrastructure builds against (src/pio_io_mod.F90, src/restart_io_mod.F90),
// but a distinct, unrelated usage: this stores arbitrary intermediate
// StateSnapshots at chosen points during the adjoint sweep's forward re-run,
// not full-simulation restart files.
//
// Each checkpoint is written as one NetCDF file per (MPI rank, CheckpointId).
// Checkpoints exist purely to support this same run's own backward sweep
// (identical ranks, identical element decomposition, on reload), so unlike a
// real restart/output file there is no need to reassemble a global field
// across ranks: each rank writes/reads only the state of the elements it
// already owns. That lets each rank use its own private, single-task PIO
// subsystem (scoped to MPI_COMM_SELF) and write whole, non-decomposed
// variables -- no PIOc decomposition/rearranger machinery needed at all.
class CheckpointStore {
public:
  // dir: directory the checkpoint files are written to/read from. Must
  // already exist (this class does not create directories). comm: the
  // (possibly multi-rank) communicator identifying which rank this process
  // is, used only to make each rank's filenames distinct -- no collective
  // operation is ever performed on it.
  CheckpointStore (const std::string& dir, const ekat::Comm& comm);
  ~CheckpointStore ();

  CheckpointStore (const CheckpointStore&) = delete;
  CheckpointStore& operator= (const CheckpointStore&) = delete;

  // Writes snap to disk under id, creating or overwriting its file.
  void save (const CheckpointId& id, const StateSnapshot& snap) const;

  // Reads the checkpoint stored under id into snap. snap must already be
  // allocated (via StateSnapshot(num_elems,alloc_ps)) with the same
  // num_elems and ps_v allocation choice used when it was saved.
  void load (const CheckpointId& id, StateSnapshot& snap) const;

  // True if a checkpoint file for id exists (for this rank).
  bool has (const CheckpointId& id) const;

  // Removes the on-disk file for id, if present. Lets the backward sweep
  // keep disk usage bounded once a checkpoint has been consumed and won't
  // be revisited.
  void remove (const CheckpointId& id) const;

private:
  std::string filename (const CheckpointId& id) const;

  std::string m_dir;
  int         m_rank;
  int         m_iosysid;
};

} // namespace Homme

#endif // HOMMEXX_CHECKPOINT_IO_HPP
