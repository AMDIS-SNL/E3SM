#include "CheckpointIO.hpp"

#include "Dimensions.hpp"

#include <ekat_assert.hpp>
#include <ekat_pack_kokkos.hpp>

#include <pio.h>
#include <mpi.h>

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <sstream>
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
  // dtype so PIOc_write_darray/PIOc_read_darray never need an internal
  // conversion.
  return std::is_same<Real,double>::value ? PIO_DOUBLE : PIO_FLOAT;
}

// Number of physical (unpadded) Real entries *per element* a rank 3/4/5
// (elem, ...) view holds, i.e. everything but the leading "elem" extent.
// Computed from extents alone (no device access).
template<typename ViewT>
size_t per_elem_stride (const ViewT& view, const int phys_last_extent) {
  constexpr int N = ViewT::rank;
  static_assert (N==3 or N==4 or N==5, "Unsupported rank for checkpoint I/O.\n");
  if constexpr (N==3) {
    return static_cast<size_t>(view.extent(1))*phys_last_extent;
  } else if constexpr (N==4) {
    return static_cast<size_t>(view.extent(1))*view.extent(2)*phys_last_extent;
  } else { // N==5
    return static_cast<size_t>(view.extent(1))*view.extent(2)*view.extent(3)*phys_last_extent;
  }
}

// Reads the physical (unpadded) entries of a (already-scalarized, so
// Real-valued) rank 3/4/5 view into a flat, row-major (element-major) host
// buffer -- exactly the layout PIOc_write_darray/PIOc_read_darray expect
// for a rank's local contribution, given a decomposition built by
// build_compmap() below. Packed fields (v, vtheta_dp, dp3d, w_i, phinh_i)
// must be passed through ekat::scalarize() first; their packing only ever
// pads the *last* dimension, so only phys_last_extent (<= the view's last
// extent) entries along it are read -- the rest is padding, and must not be
// written to disk (it may be uninitialized). When there is no padding at
// all (ps_v, which was never packed; or any packed field on a build with
// pack size 1, e.g. GPU builds), the mirror's own storage already *is* the
// buffer we want, so no per-index copy is needed -- just one bulk copy of
// its data.
template<typename ViewT>
std::vector<Real> view_to_flat_buffer (const ViewT& view, const int phys_last_extent) {
  auto h = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(h,view);

  constexpr int N = ViewT::rank;
  static_assert (N==3 or N==4 or N==5, "Unsupported rank for checkpoint I/O.\n");

  std::vector<Real> buf(static_cast<size_t>(view.extent(0))*per_elem_stride(view,phys_last_extent));

  if (view.extent_int(N-1)==phys_last_extent) {
    std::copy(h.data(),h.data()+buf.size(),buf.begin());
    return buf;
  }

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

// Inverse of view_to_flat_buffer. Deliberately does *not* touch the padding
// past phys_last_extent (when there is any): a fresh create_mirror_view's
// padding is whatever the allocator happened to give it, most likely
// uninitialized garbage, and leaving it that way is more likely to make any
// accidental use of it visibly break (NaN, crash) than a quietly-plausible
// zero would.
template<typename ViewT>
void flat_buffer_to_view (const std::vector<Real>& buf, const ViewT& view, const int phys_last_extent) {
  auto h = Kokkos::create_mirror_view(view);

  constexpr int N = ViewT::rank;
  static_assert (N==3 or N==4 or N==5, "Unsupported rank for checkpoint I/O.\n");

  if (view.extent_int(N-1)==phys_last_extent) {
    std::copy(buf.begin(),buf.begin()+buf.size(),h.data());
    Kokkos::deep_copy(view,h);
    return;
  }

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

// One contiguous block of `stride` consecutive 0-based offsets per local
// element, starting at stride*(elem_offset+i) for local element i -- i.e.
// "this rank's local elements occupy a contiguous block of the flattened
// [global_elem, ...tail dims...] array, starting at elem_offset". This is
// exactly the map PIOc_init_decomp expects (see PIOc_write_darray/
// PIOc_read_darray below): scorpio's rearranger uses it to figure out which
// rank owns which piece of the global array, and does the actual data
// movement -- no manual gather/scatter needed on our end.
std::vector<PIO_Offset> build_compmap (const int local_num_elems, const int elem_offset, const size_t stride) {
  std::vector<PIO_Offset> compmap(static_cast<size_t>(local_num_elems)*stride);
  for (int i=0; i<local_num_elems; ++i) {
    const PIO_Offset base = static_cast<PIO_Offset>(stride)*static_cast<PIO_Offset>(elem_offset+i);
    const auto beg = compmap.begin()+static_cast<size_t>(i)*stride;
    std::iota(beg,beg+stride,base);
  }
  return compmap;
}

// Creates a PIOc decomposition for a field shaped (global_num_elems,
// tail_dims...), with this rank owning local_num_elems elements starting at
// elem_offset. Collective over the iosystem's communicator.
int init_decomp (const int iosysid, const int rearranger, const int global_num_elems,
                  const int local_num_elems, const int elem_offset,
                  const std::vector<int>& tail_dims) {
  std::vector<int> gdimlen;
  gdimlen.push_back(global_num_elems);
  gdimlen.insert(gdimlen.end(),tail_dims.begin(),tail_dims.end());

  size_t stride = 1;
  for (auto d : tail_dims) stride *= static_cast<size_t>(d);

  const auto compmap = build_compmap(local_num_elems,elem_offset,stride);

  int ioid;
  const int err = PIOc_init_decomp(iosysid,pio_real_type(),static_cast<int>(gdimlen.size()),gdimlen.data(),
                                    static_cast<int>(compmap.size()),compmap.data(),&ioid,rearranger,
                                    nullptr,nullptr);
  check_pio_noerr(err,"init_decomp");
  return ioid;
}

// Writes local_view's (already scalarized) local data as record snap_idx of
// varid, via the given decomposition. Collective (every rank must call this
// with its own local slice; scorpio's rearranger does the rest).
template<typename ViewT>
void write_darray_field (const int ncid, const int varid, const int decompid, const int snap_idx,
                          const ViewT& local_view, const int phys_ext) {
  check_pio_noerr(PIOc_setframe(ncid,varid,snap_idx),"setframe");
  const auto buf = view_to_flat_buffer(local_view,phys_ext);
  check_pio_noerr(PIOc_write_darray(ncid,varid,decompid,buf.size(),buf.data(),nullptr),"write_darray");
}

// Inverse of write_darray_field.
template<typename ViewT>
void read_darray_field (const int ncid, const int varid, const int decompid, const int snap_idx,
                         ViewT local_view, const int phys_ext) {
  check_pio_noerr(PIOc_setframe(ncid,varid,snap_idx),"setframe");
  std::vector<Real> buf(static_cast<size_t>(local_view.extent(0))*per_elem_stride(local_view,phys_ext));
  check_pio_noerr(PIOc_read_darray(ncid,varid,decompid,buf.size(),buf.data()),"read_darray");
  flat_buffer_to_view(buf,local_view,phys_ext);
}

} // anonymous namespace

CheckpointStore::CheckpointStore (const std::string& dir, const ekat::Comm& comm, const int num_elems)
 : m_dir       (dir)
 , m_comm      (comm)
 , m_num_elems (num_elems)
{
  // This rank's block of "fake" global element ids is [m_elem_offset,
  // m_elem_offset+m_num_elems) -- an exclusive prefix sum of every rank's
  // num_elems, computed once here (see class comment for why this doesn't
  // need HOMME's real global element ids).
  MPI_Exscan(&m_num_elems,&m_elem_offset,1,MPI_INT,MPI_SUM,m_comm.mpi_comm());
  if (m_comm.rank()==0) {
    m_elem_offset = 0; // MPI_Exscan leaves rank 0's result value undefined
  }
  MPI_Allreduce(&m_num_elems,&m_global_num_elems,1,MPI_INT,MPI_SUM,m_comm.mpi_comm());

  // A single iosystem spanning every rank in comm (unlike a per-rank
  // MPI_COMM_SELF subsystem, since real decomposed I/O needs every rank
  // participating): every rank is also an IO task, the simplest correct
  // configuration (not necessarily the fastest at very large scale, but
  // that's a tuning knob for later, not a correctness concern now).
  const int num_iotasks = m_comm.size();
  const int stride = 1;
  const int base = 0;
  check_pio_noerr(
      PIOc_Init_Intracomm(m_comm.mpi_comm(),num_iotasks,stride,base,PIO_REARR_SUBSET,&m_iosysid),
      "Init_Intracomm");

  m_decomp_v   = init_decomp(m_iosysid,PIO_REARR_SUBSET,m_global_num_elems,m_num_elems,m_elem_offset,{2,NP,NP,NUM_PHYSICAL_LEV});
  m_decomp_mid = init_decomp(m_iosysid,PIO_REARR_SUBSET,m_global_num_elems,m_num_elems,m_elem_offset,{NP,NP,NUM_PHYSICAL_LEV});
  m_decomp_int = init_decomp(m_iosysid,PIO_REARR_SUBSET,m_global_num_elems,m_num_elems,m_elem_offset,{NP,NP,NUM_INTERFACE_LEV});
  m_decomp_ps  = init_decomp(m_iosysid,PIO_REARR_SUBSET,m_global_num_elems,m_num_elems,m_elem_offset,{NP,NP});
}

CheckpointStore::~CheckpointStore ()
{
  if (m_write_ncid>=0) {
    PIOc_sync(m_write_ncid);
    PIOc_closefile(m_write_ncid);
  }
  if (m_read_ncid>=0) {
    PIOc_closefile(m_read_ncid);
  }
  if (m_decomp_v>=0)   PIOc_freedecomp(m_iosysid,m_decomp_v);
  if (m_decomp_mid>=0) PIOc_freedecomp(m_iosysid,m_decomp_mid);
  if (m_decomp_int>=0) PIOc_freedecomp(m_iosysid,m_decomp_int);
  if (m_decomp_ps>=0)  PIOc_freedecomp(m_iosysid,m_decomp_ps);
  if (m_iosysid>=0) {
    PIOc_finalize(m_iosysid);
  }
}

std::string CheckpointStore::filename (const int nn_call) const
{
  std::ostringstream oss;
  oss << m_dir << "/ckpt_nn" << nn_call << ".nc";
  return oss.str();
}

void CheckpointStore::open_for_write (const int nn_call, const StateSnapshot& snap)
{
  if (m_write_ncid>=0 && m_write_nn_call==nn_call) {
    return;
  }
  if (m_write_ncid>=0) {
    check_pio_noerr(PIOc_sync(m_write_ncid),"sync");
    check_pio_noerr(PIOc_closefile(m_write_ncid),"closefile");
    m_write_ncid = -1;
  }

  const auto fname = filename(nn_call);
  const bool has_ps = snap.ps_v.data()!=nullptr;

  int ncid;
  int iotype = PIO_IOTYPE_NETCDF;
  check_pio_noerr(
      PIOc_createfile(m_iosysid,&ncid,&iotype,fname.c_str(),PIO_64BIT_OFFSET),
      "createfile");

  int snap_dim, elem_dim, uv_cmp_dim, np_dim, lev_dim, ilev_dim;
  check_pio_noerr(PIOc_def_dim(ncid,"snap",  PIO_UNLIMITED,      &snap_dim  ), "def_dim(snap)");
  check_pio_noerr(PIOc_def_dim(ncid,"elem",  m_global_num_elems, &elem_dim  ), "def_dim(elem)");
  check_pio_noerr(PIOc_def_dim(ncid,"uv_cmp",2,                  &uv_cmp_dim), "def_dim(uv_cmp)");
  check_pio_noerr(PIOc_def_dim(ncid,"np",    NP,                 &np_dim    ), "def_dim(np)");
  check_pio_noerr(PIOc_def_dim(ncid,"lev",   NUM_PHYSICAL_LEV,   &lev_dim   ), "def_dim(lev)");
  check_pio_noerr(PIOc_def_dim(ncid,"ilev",  NUM_INTERFACE_LEV,  &ilev_dim  ), "def_dim(ilev)");

  const int dtype = pio_real_type();
  FieldVarIds ids;
  {
    std::vector<int> dims_v   = {snap_dim,elem_dim,uv_cmp_dim,np_dim,np_dim,lev_dim };
    std::vector<int> dims_mid = {snap_dim,elem_dim,           np_dim,np_dim,lev_dim };
    std::vector<int> dims_int = {snap_dim,elem_dim,           np_dim,np_dim,ilev_dim};
    check_pio_noerr(PIOc_def_var(ncid,"v",        dtype,(int)dims_v.size(),  dims_v.data(),  &ids.v),        "def_var(v)");
    check_pio_noerr(PIOc_def_var(ncid,"vtheta_dp",dtype,(int)dims_mid.size(),dims_mid.data(),&ids.vtheta_dp),"def_var(vtheta_dp)");
    check_pio_noerr(PIOc_def_var(ncid,"dp3d",     dtype,(int)dims_mid.size(),dims_mid.data(),&ids.dp3d),     "def_var(dp3d)");
    check_pio_noerr(PIOc_def_var(ncid,"w_i",      dtype,(int)dims_int.size(),dims_int.data(),&ids.w_i),      "def_var(w_i)");
    check_pio_noerr(PIOc_def_var(ncid,"phinh_i",  dtype,(int)dims_int.size(),dims_int.data(),&ids.phinh_i),  "def_var(phinh_i)");
    if (has_ps) {
      std::vector<int> dims_ps = {snap_dim,elem_dim,np_dim,np_dim};
      check_pio_noerr(PIOc_def_var(ncid,"ps_v",dtype,(int)dims_ps.size(),dims_ps.data(),&ids.ps_v), "def_var(ps_v)");
    }
  }

  const int has_ps_flag = has_ps ? 1 : 0;
  check_pio_noerr(PIOc_put_att(ncid,PIO_GLOBAL,"has_ps",PIO_INT,1,&has_ps_flag), "put_att(has_ps)");
  check_pio_noerr(PIOc_enddef(ncid), "enddef");

  m_write_ncid    = ncid;
  m_write_nn_call = nn_call;
  m_write_var_ids = ids;
}

void CheckpointStore::open_for_read (const int nn_call)
{
  if (m_read_ncid>=0 && m_read_nn_call==nn_call) {
    return;
  }
  if (m_read_ncid>=0) {
    check_pio_noerr(PIOc_closefile(m_read_ncid),"closefile");
    m_read_ncid = -1;
  }

  const auto fname = filename(nn_call);
  EKAT_REQUIRE_MSG (std::filesystem::exists(fname),
      "Error! No checkpoint file found for the given nn_call.\n"
      " - filename: " + fname + "\n");

  int ncid;
  int iotype = PIO_IOTYPE_NETCDF;
  check_pio_noerr(PIOc_openfile(m_iosysid,&ncid,&iotype,fname.c_str(),PIO_NOWRITE), "openfile");

  FieldVarIds ids;
  check_pio_noerr(PIOc_inq_varid(ncid,"v",        &ids.v),        "inq_varid(v)");
  check_pio_noerr(PIOc_inq_varid(ncid,"vtheta_dp",&ids.vtheta_dp),"inq_varid(vtheta_dp)");
  check_pio_noerr(PIOc_inq_varid(ncid,"dp3d",     &ids.dp3d),     "inq_varid(dp3d)");
  check_pio_noerr(PIOc_inq_varid(ncid,"w_i",      &ids.w_i),      "inq_varid(w_i)");
  check_pio_noerr(PIOc_inq_varid(ncid,"phinh_i",  &ids.phinh_i),  "inq_varid(phinh_i)");

  int has_ps_flag = 0;
  check_pio_noerr(PIOc_get_att(ncid,PIO_GLOBAL,"has_ps",&has_ps_flag), "get_att(has_ps)");
  if (has_ps_flag!=0) {
    check_pio_noerr(PIOc_inq_varid(ncid,"ps_v",&ids.ps_v), "inq_varid(ps_v)");
  }

  m_read_ncid    = ncid;
  m_read_nn_call = nn_call;
  m_read_var_ids = ids;
}

void CheckpointStore::save (const CheckpointId& id, const StateSnapshot& snap)
{
  EKAT_REQUIRE_MSG (snap.num_elems==m_num_elems,
      "Error! StateSnapshot's num_elems does not match this CheckpointStore's (fixed at construction).\n");

  open_for_write(id.nn_call,snap);

  const int snap_idx = m_next_snap_idx[id.nn_call]++;
  m_snap_idx[std::make_tuple(id.nn_call,id.subcycle_call,id.step)] = snap_idx;

  write_darray_field(m_write_ncid,m_write_var_ids.v,        m_decomp_v,  snap_idx,ekat::scalarize(snap.v),        NUM_PHYSICAL_LEV);
  write_darray_field(m_write_ncid,m_write_var_ids.vtheta_dp,m_decomp_mid,snap_idx,ekat::scalarize(snap.vtheta_dp),NUM_PHYSICAL_LEV);
  write_darray_field(m_write_ncid,m_write_var_ids.dp3d,     m_decomp_mid,snap_idx,ekat::scalarize(snap.dp3d),     NUM_PHYSICAL_LEV);
  write_darray_field(m_write_ncid,m_write_var_ids.w_i,      m_decomp_int,snap_idx,ekat::scalarize(snap.w_i),      NUM_INTERFACE_LEV);
  write_darray_field(m_write_ncid,m_write_var_ids.phinh_i,  m_decomp_int,snap_idx,ekat::scalarize(snap.phinh_i),  NUM_INTERFACE_LEV);
  if (snap.ps_v.data()!=nullptr) {
    write_darray_field(m_write_ncid,m_write_var_ids.ps_v,m_decomp_ps,snap_idx,snap.ps_v,NP);
  }

  check_pio_noerr(PIOc_sync(m_write_ncid),"sync");
}

void CheckpointStore::load (const CheckpointId& id, StateSnapshot& snap)
{
  EKAT_REQUIRE_MSG (snap.num_elems==m_num_elems,
      "Error! StateSnapshot's num_elems does not match this CheckpointStore's (fixed at construction).\n");

  int ncid;
  FieldVarIds ids;
  if (m_write_ncid>=0 && m_write_nn_call==id.nn_call) {
    // id.nn_call's file is still open for writing: read straight from that
    // same handle rather than opening the file a second time (PIO's
    // internal buffering makes two simultaneously-open handles to the same
    // file, one write and one read, risky).
    check_pio_noerr(PIOc_sync(m_write_ncid),"sync");
    ncid = m_write_ncid;
    ids  = m_write_var_ids;
  } else {
    open_for_read(id.nn_call);
    ncid = m_read_ncid;
    ids  = m_read_var_ids;
  }

  const auto it = m_snap_idx.find(std::make_tuple(id.nn_call,id.subcycle_call,id.step));
  EKAT_REQUIRE_MSG (it!=m_snap_idx.end(),
      "Error! No checkpoint found for the given id -- was it save()'d by this CheckpointStore?\n");
  const int snap_idx = it->second;

  EKAT_REQUIRE_MSG ((ids.ps_v>=0)==(snap.ps_v.data()!=nullptr),
      "Error! Checkpoint's ps_v allocation does not match the target StateSnapshot's.\n");

  read_darray_field(ncid,ids.v,        m_decomp_v,  snap_idx,ekat::scalarize(snap.v),        NUM_PHYSICAL_LEV);
  read_darray_field(ncid,ids.vtheta_dp,m_decomp_mid,snap_idx,ekat::scalarize(snap.vtheta_dp),NUM_PHYSICAL_LEV);
  read_darray_field(ncid,ids.dp3d,     m_decomp_mid,snap_idx,ekat::scalarize(snap.dp3d),     NUM_PHYSICAL_LEV);
  read_darray_field(ncid,ids.w_i,      m_decomp_int,snap_idx,ekat::scalarize(snap.w_i),      NUM_INTERFACE_LEV);
  read_darray_field(ncid,ids.phinh_i,  m_decomp_int,snap_idx,ekat::scalarize(snap.phinh_i),  NUM_INTERFACE_LEV);
  if (snap.ps_v.data()!=nullptr) {
    read_darray_field(ncid,ids.ps_v,m_decomp_ps,snap_idx,snap.ps_v,NP);
  }
}

bool CheckpointStore::has (const CheckpointId& id) const
{
  return m_snap_idx.count(std::make_tuple(id.nn_call,id.subcycle_call,id.step))>0;
}

void CheckpointStore::discard (const int nn_call)
{
  if (m_write_ncid>=0 && m_write_nn_call==nn_call) {
    check_pio_noerr(PIOc_sync(m_write_ncid),"sync");
    check_pio_noerr(PIOc_closefile(m_write_ncid),"closefile");
    m_write_ncid = -1;
    m_write_nn_call = -1;
  }
  if (m_read_ncid>=0 && m_read_nn_call==nn_call) {
    check_pio_noerr(PIOc_closefile(m_read_ncid),"closefile");
    m_read_ncid = -1;
    m_read_nn_call = -1;
  }

  // Every rank reaches this point together (the closefile calls above are
  // collective), so it's safe for just one rank to unlink the shared file.
  MPI_Barrier(m_comm.mpi_comm());
  if (m_comm.rank()==0) {
    const auto fname = filename(nn_call);
    if (std::filesystem::exists(fname)) {
      std::filesystem::remove(fname);
    }
  }

  m_next_snap_idx.erase(nn_call);
  for (auto it=m_snap_idx.begin(); it!=m_snap_idx.end(); ) {
    if (std::get<0>(it->first)==nn_call) {
      it = m_snap_idx.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace Homme
