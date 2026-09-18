#ifndef HOMMEXX_CHECKPOINT_IO_HPP
#define HOMMEXX_CHECKPOINT_IO_HPP

#include "Hommexx_config.h"

#include "StateSnapshot.hpp"

#include <ekat_comm.hpp>

#include <map>
#include <string>
#include <tuple>

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
// All ranks share ONE file per NN call (not one file per rank), written and
// read with genuine parallel, decomposed I/O (PIOc_write_darray/
// PIOc_read_darray) -- scorpio's own rearranger moves data between ranks as
// it sees fit, rather than this class funneling everything through one
// rank. The decomposition itself needs a global degrees-of-freedom map, but
// building one doesn't need HOMME's real (cube_mod.F90) element global IDs:
// since a checkpoint only ever has to round-trip within *this run's own*
// backward sweep, any consistent numbering works, so the constructor just
// assigns each rank a contiguous block -- rank 0 gets elements
// [0,num_elems_0), rank 1 gets [num_elems_0,num_elems_0+num_elems_1), and so
// on -- via a single exclusive prefix sum (MPI_Exscan) at construction time.
// The resulting decomposition (one per distinct per-element field shape,
// since e.g. v and vtheta_dp differ) is created once, in the constructor,
// and reused by every save()/load() call and every NN call's file for the
// life of this object.
//
// Successive checkpoints within the same NN call are appended as records
// along an unlimited "snap" dimension in that NN call's file; moving to a
// new NN call starts a new file (see save()).
class CheckpointStore {
public:
  // dir: directory the checkpoint files are written to/read from. Must
  // already exist (this class does not create directories) and must be on
  // a filesystem shared by every rank. comm: the communicator this store is
  // collective over. num_elems: how many elements *this* rank owns -- fixed
  // for the life of this object (matches a StateSnapshot's own num_elems,
  // which must agree with this value on every save()/load() call).
  CheckpointStore (const std::string& dir, const ekat::Comm& comm, int num_elems);
  ~CheckpointStore ();

  CheckpointStore (const CheckpointStore&) = delete;
  CheckpointStore& operator= (const CheckpointStore&) = delete;

  // Collective over comm. Appends snap as a new record in id.nn_call's file
  // (creating that file, and defining its dims/vars, the first time this
  // nn_call is seen; later calls for the same nn_call just append). Moving
  // to a *different* nn_call than the one currently open for writing closes
  // that file first -- so callers must save() in nn_call-nondecreasing order
  // (the natural order of a forward integration).
  void save (const CheckpointId& id, const StateSnapshot& snap);

  // Collective over comm. Reads the checkpoint stored under id into snap,
  // which must already be allocated (via StateSnapshot(num_elems,alloc_ps))
  // with the same num_elems (on every rank) and ps_v allocation choice used
  // when it was saved. id must have been save()'d earlier by this same
  // CheckpointStore instance (the snap-index mapping is kept in memory, not
  // persisted to the file -- see the class comment's "same run" guarantee).
  void load (const CheckpointId& id, StateSnapshot& snap);

  // True if id has been save()'d by this CheckpointStore instance.
  bool has (const CheckpointId& id) const;

  // Deletes the on-disk file holding all of nn_call's checkpoints, once the
  // backward sweep has consumed every one of them and won't revisit any --
  // lets it keep disk usage bounded. Collective over comm.
  void discard (int nn_call);

private:
  std::string filename (int nn_call) const;

  // (Re)opens, creating and defining dims/vars if new, the file for nn_call
  // in write mode, closing whatever was previously open for writing if it
  // was a different nn_call. Collective over comm.
  void open_for_write (int nn_call, const StateSnapshot& snap);

  // (Re)opens the file for nn_call in read mode, closing whatever was
  // previously open for reading if it was a different nn_call. Collective
  // over comm.
  void open_for_read (int nn_call);

  // varids for a checkpoint file's fields (one per StateSnapshot field --
  // see StateSnapshot.hpp). ps_v stays -1 for a file whose StateSnapshot had
  // no ps_v allocated.
  struct FieldVarIds {
    int v = -1, vtheta_dp = -1, dp3d = -1, w_i = -1, phinh_i = -1, ps_v = -1;
  };

  std::string m_dir;
  ekat::Comm  m_comm;
  int         m_iosysid = -1;

  int m_num_elems    = 0; // this rank's own element count (fixed)
  int m_elem_offset  = 0; // this rank's first element's 0-based global index
  int m_global_num_elems = 0;

  // PIOc decomposition ids (ioid), one per distinct per-element field shape,
  // created once in the constructor and reused for every field/file/call
  // sharing that shape: "v" (2 x np x np x lev), "mid" (np x np x lev, used
  // by both vtheta_dp and dp3d), "int" (np x np x ilev, used by both w_i and
  // phinh_i), "ps" (np x np, used by ps_v).
  int m_decomp_v   = -1;
  int m_decomp_mid = -1;
  int m_decomp_int = -1;
  int m_decomp_ps  = -1;

  // Persistent scratch buffers for staging a field's data to/from host (and,
  // when a field is packed with padding past its physical extent, for
  // stripping/expanding that padding on device first) -- reused by every
  // save()/load() call and every field sharing a given shape, so no device
  // or host allocation happens on the save()/load() hot path. Allocated
  // once, in the constructor, at each shape's physical (unpadded) size
  // (fixed for the life of this object, since m_num_elems is fixed): "v" (2
  // x np x np x lev), "mid" (np x np x lev, shared by vtheta_dp and dp3d),
  // "int" (np x np x ilev, shared by w_i and phinh_i), "ps" (np x np, used
  // by ps_v, which is never packed).
  Kokkos::View<Real*****,Kokkos::LayoutRight,ExecSpace> m_buf_v_dev;
  Kokkos::View<Real*****,Kokkos::LayoutRight,ExecSpace>::HostMirror m_buf_v_host;
  Kokkos::View<Real****, Kokkos::LayoutRight,ExecSpace> m_buf_mid_dev;
  Kokkos::View<Real****, Kokkos::LayoutRight,ExecSpace>::HostMirror m_buf_mid_host;
  Kokkos::View<Real****, Kokkos::LayoutRight,ExecSpace> m_buf_int_dev;
  Kokkos::View<Real****, Kokkos::LayoutRight,ExecSpace>::HostMirror m_buf_int_host;
  Kokkos::View<Real***,  Kokkos::LayoutRight,ExecSpace> m_buf_ps_dev;
  Kokkos::View<Real***,  Kokkos::LayoutRight,ExecSpace>::HostMirror m_buf_ps_host;

  // Currently-open file handles (a file is "open" here in the sense of "we
  // haven't moved on to another nn_call yet", not necessarily that its ncid
  // is live at every instant -- see the .cpp).
  int          m_write_ncid    = -1;
  int          m_write_nn_call = -1;
  FieldVarIds  m_write_var_ids;
  int          m_read_ncid     = -1;
  int          m_read_nn_call  = -1;
  FieldVarIds  m_read_var_ids;

  // Next snap-record index to use for a given nn_call's file, and the
  // (nn_call,subcycle_call,step) -> snap-index mapping save() has produced
  // so far (this run only; not persisted to disk). Every rank computes the
  // same values independently (every rank takes the same save()/load() call
  // sequence, so no communication is needed to keep them in sync).
  std::map<int, int> m_next_snap_idx;
  std::map<std::tuple<int,int,int>, int> m_snap_idx;
};

} // namespace Homme

#endif // HOMMEXX_CHECKPOINT_IO_HPP
