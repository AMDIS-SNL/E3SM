#include "CheckpointIO.hpp"

#ifdef HOMMEXX_ENABLE_CHECKPOINT_IO

#include "Dimensions.hpp"

#include <ekat_assert.hpp>
#include <ekat_pack_kokkos.hpp>

#include <pio.h>
#include <mpi.h>

#include <cstdio>
#include <sstream>
#include <sys/stat.h>
#include <type_traits>
#include <vector>

namespace Homme {

namespace {

void check_pio_noerr (const int err, const std::string& pioc_func) {
  EKAT_REQUIRE_MSG (err==PIO_NOERR,
      "Error! A PIOc call failed inside CheckpointStore.\n"
      " - pio error code   : " + std::to_string(err) + "\n"
      " - calling PIOc func: " + pioc_func + "\n");
}

int pio_real_type () {
  // HOMME's Real is always either float or double; pick the matching PIO
  // dtype so PIOc_put_var/PIOc_get_var never need an internal conversion.
  return std::is_same<Real,double>::value ? PIO_DOUBLE : PIO_FLOAT;
}

// Number of physical (unpadded) entries a rank 3/4/5 view holds, i.e. the
// size view_to_flat_buffer's/flat_buffer_to_view's buffer must have -- computed
// from extents alone (no device access), so callers can size a read buffer
// without paying for a throwaway device->host copy.
template<typename ViewT>
size_t flat_buffer_size (const ViewT& view, const int phys_last_extent) {
  constexpr int N = ViewT::rank;
  static_assert (N==3 or N==4 or N==5, "Unsupported rank for checkpoint I/O.\n");
  if constexpr (N==3) {
    return view.extent(0)*view.extent(1)*phys_last_extent;
  } else if constexpr (N==4) {
    return view.extent(0)*view.extent(1)*view.extent(2)*phys_last_extent;
  } else { // N==5
    return view.extent(0)*view.extent(1)*view.extent(2)*view.extent(3)*phys_last_extent;
  }
}

// Reads the physical (unpadded) entries of a (already-scalarized, so
// Real-valued) rank 3/4/5 view into a flat, row-major buffer suitable for
// PIOc_put_var. Packed fields (v, vtheta_dp, dp3d, w_i, phinh_i) must be
// passed through ekat::scalarize() first; their packing only ever pads the
// *last* dimension, so only phys_last_extent (<= view's last extent) entries
// along it are read here -- the rest is padding, and must not be written to
// disk (it may be uninitialized).
template<typename ViewT>
std::vector<Real> view_to_flat_buffer (const ViewT& view, const int phys_last_extent) {
  auto h = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(h,view);

  constexpr int N = ViewT::rank;
  static_assert (N==3 or N==4 or N==5, "Unsupported rank for checkpoint I/O.\n");

  std::vector<Real> buf(flat_buffer_size(view,phys_last_extent));
  size_t idx = 0;
  if constexpr (N==3) {
    for (int i=0; i<view.extent_int(0); ++i)
      for (int j=0; j<view.extent_int(1); ++j)
        for (int k=0; k<phys_last_extent; ++k)
          buf[idx++] = h(i,j,k);
  } else if constexpr (N==4) {
    for (int i=0; i<view.extent_int(0); ++i)
      for (int j=0; j<view.extent_int(1); ++j)
        for (int k=0; k<view.extent_int(2); ++k)
          for (int l=0; l<phys_last_extent; ++l)
            buf[idx++] = h(i,j,k,l);
  } else { // N==5
    for (int i=0; i<view.extent_int(0); ++i)
      for (int j=0; j<view.extent_int(1); ++j)
        for (int k=0; k<view.extent_int(2); ++k)
          for (int l=0; l<view.extent_int(3); ++l)
            for (int m=0; m<phys_last_extent; ++m)
              buf[idx++] = h(i,j,k,l,m);
  }
  return buf;
}

// Inverse of view_to_flat_buffer: fills the physical entries of view from
// buf, and zeroes the padding past phys_last_extent (matching the convention
// used elsewhere in HOMME of keeping pack padding at a well-defined zero,
// rather than leaving it as whatever garbage a fresh allocation contains).
template<typename ViewT>
void flat_buffer_to_view (const std::vector<Real>& buf, const ViewT& view, const int phys_last_extent) {
  auto h = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(h,Real(0));

  constexpr int N = ViewT::rank;
  static_assert (N==3 or N==4 or N==5, "Unsupported rank for checkpoint I/O.\n");

  size_t idx = 0;
  if constexpr (N==3) {
    for (int i=0; i<view.extent_int(0); ++i)
      for (int j=0; j<view.extent_int(1); ++j)
        for (int k=0; k<phys_last_extent; ++k)
          h(i,j,k) = buf[idx++];
  } else if constexpr (N==4) {
    for (int i=0; i<view.extent_int(0); ++i)
      for (int j=0; j<view.extent_int(1); ++j)
        for (int k=0; k<view.extent_int(2); ++k)
          for (int l=0; l<phys_last_extent; ++l)
            h(i,j,k,l) = buf[idx++];
  } else { // N==5
    for (int i=0; i<view.extent_int(0); ++i)
      for (int j=0; j<view.extent_int(1); ++j)
        for (int k=0; k<view.extent_int(2); ++k)
          for (int l=0; l<view.extent_int(3); ++l)
            for (int m=0; m<phys_last_extent; ++m)
              h(i,j,k,l,m) = buf[idx++];
  }
  Kokkos::deep_copy(view,h);
}

// Small RAII-free helper bundling the dim ids shared by all of a checkpoint
// file's variables.
struct CheckpointDims {
  int elem, cmp, np, lev, ilev;
};

CheckpointDims define_checkpoint_dims (const int ncid, const int num_elems) {
  CheckpointDims d;
  check_pio_noerr(PIOc_def_dim(ncid,"elem",num_elems,           &d.elem), "def_dim(elem)");
  check_pio_noerr(PIOc_def_dim(ncid,"cmp", 2,                   &d.cmp ), "def_dim(cmp)");
  check_pio_noerr(PIOc_def_dim(ncid,"np",  NP,                  &d.np  ), "def_dim(np)");
  check_pio_noerr(PIOc_def_dim(ncid,"lev", NUM_PHYSICAL_LEV,    &d.lev ), "def_dim(lev)");
  check_pio_noerr(PIOc_def_dim(ncid,"ilev",NUM_INTERFACE_LEV,   &d.ilev), "def_dim(ilev)");
  return d;
}

} // anonymous namespace

CheckpointStore::CheckpointStore (const std::string& dir, const ekat::Comm& comm)
 : m_dir  (dir)
 , m_rank (comm.rank())
{
  // Each rank owns its own single-task PIO subsystem, scoped to
  // MPI_COMM_SELF: checkpoint files are per-rank, so there is never any
  // cross-rank decomposition/rearranging to do (see class comment in the
  // header).
  const int num_iotasks = 1;
  const int stride = 1;
  const int base = 0;
  check_pio_noerr(
      PIOc_Init_Intracomm(MPI_COMM_SELF,num_iotasks,stride,base,PIO_REARR_SUBSET,&m_iosysid),
      "Init_Intracomm");
}

CheckpointStore::~CheckpointStore ()
{
  PIOc_finalize(m_iosysid);
}

std::string CheckpointStore::filename (const CheckpointId& id) const
{
  std::ostringstream oss;
  oss << m_dir << "/ckpt"
      << "_r"   << m_rank
      << "_nn"  << id.nn_call
      << "_sub" << id.subcycle_call
      << "_step"<< id.step
      << ".nc";
  return oss.str();
}

bool CheckpointStore::has (const CheckpointId& id) const
{
  struct stat buffer;
  return stat(filename(id).c_str(),&buffer)==0;
}

void CheckpointStore::remove (const CheckpointId& id) const
{
  if (has(id)) {
    std::remove(filename(id).c_str());
  }
}

void CheckpointStore::save (const CheckpointId& id, const StateSnapshot& snap) const
{
  const bool has_ps = snap.ps_v.data()!=nullptr;
  const auto fname = filename(id);

  int ncid;
  int iotype = PIO_IOTYPE_NETCDF;
  check_pio_noerr(
      PIOc_createfile(m_iosysid,&ncid,&iotype,fname.c_str(),PIO_64BIT_OFFSET),
      "createfile");

  const auto d = define_checkpoint_dims(ncid,snap.num_elems);
  const int dtype = pio_real_type();

  int v_id, vtheta_id, dp3d_id, w_id, phi_id, ps_id = -1;
  {
    std::vector<int> dims_v        = {d.elem,d.cmp,d.np,d.np,d.lev };
    std::vector<int> dims_mid      = {d.elem,      d.np,d.np,d.lev };
    std::vector<int> dims_int      = {d.elem,      d.np,d.np,d.ilev};
    check_pio_noerr(PIOc_def_var(ncid,"v",        dtype,(int)dims_v.size(),  dims_v.data(),  &v_id),     "def_var(v)");
    check_pio_noerr(PIOc_def_var(ncid,"vtheta_dp",dtype,(int)dims_mid.size(),dims_mid.data(),&vtheta_id),"def_var(vtheta_dp)");
    check_pio_noerr(PIOc_def_var(ncid,"dp3d",     dtype,(int)dims_mid.size(),dims_mid.data(),&dp3d_id),  "def_var(dp3d)");
    check_pio_noerr(PIOc_def_var(ncid,"w_i",      dtype,(int)dims_int.size(),dims_int.data(),&w_id),     "def_var(w_i)");
    check_pio_noerr(PIOc_def_var(ncid,"phinh_i",  dtype,(int)dims_int.size(),dims_int.data(),&phi_id),   "def_var(phinh_i)");
    if (has_ps) {
      std::vector<int> dims_ps = {d.elem,d.np,d.np};
      check_pio_noerr(PIOc_def_var(ncid,"ps_v",dtype,(int)dims_ps.size(),dims_ps.data(),&ps_id), "def_var(ps_v)");
    }
  }

  const int has_ps_flag = has_ps ? 1 : 0;
  check_pio_noerr(PIOc_put_att(ncid,PIO_GLOBAL,"has_ps",PIO_INT,1,&has_ps_flag), "put_att(has_ps)");

  check_pio_noerr(PIOc_enddef(ncid), "enddef");

  auto write_field = [&] (const int varid, const auto& scalarized_view, const int phys_ext) {
    auto buf = view_to_flat_buffer(scalarized_view,phys_ext);
    check_pio_noerr(PIOc_put_var(ncid,varid,buf.data()), "put_var");
  };

  write_field(v_id,       ekat::scalarize(snap.v),        NUM_PHYSICAL_LEV);
  write_field(vtheta_id,  ekat::scalarize(snap.vtheta_dp),NUM_PHYSICAL_LEV);
  write_field(dp3d_id,    ekat::scalarize(snap.dp3d),     NUM_PHYSICAL_LEV);
  write_field(w_id,       ekat::scalarize(snap.w_i),      NUM_INTERFACE_LEV);
  write_field(phi_id,     ekat::scalarize(snap.phinh_i),  NUM_INTERFACE_LEV);
  if (has_ps) {
    write_field(ps_id, snap.ps_v, NP);
  }

  check_pio_noerr(PIOc_sync(ncid),      "sync");
  check_pio_noerr(PIOc_closefile(ncid), "closefile");
}

void CheckpointStore::load (const CheckpointId& id, StateSnapshot& snap) const
{
  const auto fname = filename(id);
  EKAT_REQUIRE_MSG (has(id), "Error! No checkpoint file found for the given id.\n"
                             " - filename: " + fname + "\n");

  int ncid;
  int iotype = PIO_IOTYPE_NETCDF;
  check_pio_noerr(PIOc_openfile(m_iosysid,&ncid,&iotype,fname.c_str(),PIO_NOWRITE), "openfile");

  int has_ps_flag = 0;
  check_pio_noerr(PIOc_get_att(ncid,PIO_GLOBAL,"has_ps",&has_ps_flag), "get_att(has_ps)");
  const bool has_ps = has_ps_flag!=0;
  EKAT_REQUIRE_MSG (has_ps==(snap.ps_v.data()!=nullptr),
      "Error! Checkpoint's ps_v allocation does not match the target StateSnapshot's.\n"
      " - filename: " + fname + "\n");

  int v_id, vtheta_id, dp3d_id, w_id, phi_id, ps_id = -1;
  check_pio_noerr(PIOc_inq_varid(ncid,"v",        &v_id),     "inq_varid(v)");
  check_pio_noerr(PIOc_inq_varid(ncid,"vtheta_dp",&vtheta_id),"inq_varid(vtheta_dp)");
  check_pio_noerr(PIOc_inq_varid(ncid,"dp3d",     &dp3d_id),  "inq_varid(dp3d)");
  check_pio_noerr(PIOc_inq_varid(ncid,"w_i",      &w_id),     "inq_varid(w_i)");
  check_pio_noerr(PIOc_inq_varid(ncid,"phinh_i",  &phi_id),   "inq_varid(phinh_i)");
  if (has_ps) {
    check_pio_noerr(PIOc_inq_varid(ncid,"ps_v",&ps_id), "inq_varid(ps_v)");
  }

  auto read_field = [&] (const int varid, auto scalarized_view, const int phys_ext) {
    std::vector<Real> buf(flat_buffer_size(scalarized_view,phys_ext));
    check_pio_noerr(PIOc_get_var(ncid,varid,buf.data()), "get_var");
    flat_buffer_to_view(buf,scalarized_view,phys_ext);
  };

  read_field(v_id,      ekat::scalarize(snap.v),        NUM_PHYSICAL_LEV);
  read_field(vtheta_id, ekat::scalarize(snap.vtheta_dp),NUM_PHYSICAL_LEV);
  read_field(dp3d_id,   ekat::scalarize(snap.dp3d),     NUM_PHYSICAL_LEV);
  read_field(w_id,      ekat::scalarize(snap.w_i),      NUM_INTERFACE_LEV);
  read_field(phi_id,    ekat::scalarize(snap.phinh_i),  NUM_INTERFACE_LEV);
  if (has_ps) {
    read_field(ps_id, snap.ps_v, NP);
  }

  check_pio_noerr(PIOc_closefile(ncid), "closefile");
}

} // namespace Homme

#endif // HOMMEXX_ENABLE_CHECKPOINT_IO
