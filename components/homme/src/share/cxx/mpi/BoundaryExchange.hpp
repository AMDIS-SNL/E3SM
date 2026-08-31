/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_BOUNDARY_EXCHANGE_HPP
#define HOMMEXX_BOUNDARY_EXCHANGE_HPP

#include "BoundaryExchangeBase.hpp"
#include "Connectivity.hpp"

#include "Types.hpp"
#include "MpiHelpers.hpp"
#include "ErrorDefs.hpp"
#include "Hommexx_Debug.hpp"

#include <ekat_string_utils.hpp>
#include <ekat_assert.hpp>

#include <memory>

#include <vector>
#include <map>
#include <string>

#include <assert.h>

namespace Homme
{

/*
 * BoundaryExchange: a class to handle the pack/exchange/unpack process
 *
 * This class (BE) takes care of exchanging the values of one or more fields in the
 * GP on the boundary of the elements with the neighboring elements. In particular,
 * it takes care of packing the values into some buffers, performing the exchange
 * (usually, this includes some MPI calls, unless running in serial mode),
 * and then unpacking the values, accumulating them into the receiving elements.
 * This process can be done for an arbitrary number of 2d fields (that is,
 * no vertical levels) and 3d fields (with vertical levels). If you have a
 * vector field of dimension DIM, you need to register DIM separate scalar
 * fields. Internally, for each input field the BE object stores a bunch of
 * separate Views, that view the input field at each element and component
 * (if vector field). When the exchange method is called, ALL the stored
 * fields are packed/exchanged/unpacked. Therefore, if you have two sets
 * of fields that need to be exchanged at different times, you need to
 * register them into two separate BE objects.
 *
 * The registration happens in three steps:
 *
 *  - a call to set_num_fields, which sets the number of 1d, 2d and 3d fields
 *    that will be exchanged. Once this method is called, it cannot be
 *    called again, unless the method clean_up is called first.
 *  - a number of calls to one of more of the register_field(...), methods,
 *    which set the fields into the BE class. You cannot register more fields
 *    than declared in the set_num_fields call. However you can, if you want,
 *    register less fields, although this scenario is not tested, and may
 *    be buggy, so you are probably better off calling set_num_fields with
 *    the actual number of fields you are going to register. Note that you
 *    are not allowed to call register_field(...) before set_num_fields.
 *    NOTE: you are NOT allowed to register 1d fields together with
 *          2d/3d fields. The idea is that 1d fields are exchanged only as
 *          min/max quantities (so they are not accumulated). Please, use
 *          two different BE objects for accumulation and for min/max.
 *  - a call to registration_completed, which ends the registration phase,
 *    and sets up all the internal structure to prepare for calls to
 *    exchange(). This method MUST be called BEFORE any call to exchange.
 *
 * This class relies on the MpiBuffersManager (BM) class for the handling of the buffers.
 * See MpiBuffersManager header for more info on that. As explained above,
 * you may have different BE objects, which are however never used at the
 * same time. Therefore, it makes sense to reuse the same BM for all of them.
 * For this reason, the BM can serve multiple 'customers'. The BE and BM
 * classes are linked by a provider-customer relationship. It is up to the BE
 * to register itself as a customer in the stored BM, and to unregister
 * itself before going out of scope, to make sure the BM does not keep
 * a dangling pointer to a non-existent customer. This is why BE's destructor
 * automatically removes the 'this' object from the stored BM's customers list
 * (assuming there is a stored BM, otherwise nothing happens).
 *
 * In order to work correctly, BE needs a valid Connectivity and a valid
 * BM (both stored as shared_ptr). They can be set at construction time
 * or later, via a setter method. There are only a few rules:
 *
 *  - once one is set, you cannot reset it
 *  - the Connectivity must be set BEFORE any call to set_num_fields
 *  - the BM must be set BEFORE any call to registration_completed
 *
 */

template<typename ST>
class BoundaryExchangeST : public BoundaryExchangeBase
{
public:
  using PT = PackType<ST>;

  BoundaryExchangeST();
  BoundaryExchangeST(const std::shared_ptr<Connectivity>& connectivity,
                     const std::shared_ptr<MpiBuffersManager>& buffers_manager);

  // Thou shall not copy this class
  BoundaryExchangeST(const BoundaryExchangeST&) = delete;
  BoundaryExchangeST& operator= (const BoundaryExchangeST&) = delete;

  ~BoundaryExchangeST();

  // These number refers to *scalar* fields. A 2-vector field counts as 2 fields.
  void set_num_fields (const int num_1d_fields, const int num_2d_fields, const int num_3d_fields, const int num_3d_int_fields = 0);

  // Clean up MPI stuff and registered fields (but leaves connectivity and buffers manager)
  void clean_up ();

  // Notes:
  // - the first runtime dimension (if present) is always the number of elements
  // - num_dims is the # of dimensions to exchange
  // - start_dim is the first to exchange
  // - in case of tensors (more than 1 rank between the element dim and the first NP),
  //   the outer dimension MUST be sliced, while the inner dimension can be fully exchanged

  // 2d fields (no vertical level dimension)
  template<typename... Properties>
  void register_field (ExecView<ST*[NP][NP], Properties...> field);
  template<typename... Properties>
  void register_field (ExecView<ST**[NP][NP], Properties...> field, int num_dims, int start_dim);
  template<typename... Properties>
  void register_field (ExecView<ST***[NP][NP], Properties...> field, int idim_out, int num_dims, int start_dim);
  template<int DIM, typename... Properties>
  void register_field (ExecView<ST*[DIM][NP][NP], Properties...> field, int num_dims, int start_dim);

  // 3d fields (with vertical level dimension at the end)
  template<int OUTER_DIM, int DIM, typename... Properties>
  void register_field (ExecView<PT*[OUTER_DIM][DIM][NP][NP][NUM_LEV], Properties...> field, int idim_out, int num_dims, int start_dim, int nlev=NUM_LEV);
  template<typename... Properties>
  void register_field (ExecView<PT***[NP][NP][NUM_LEV], Properties...> field, int idim_out, int num_dims, int start_dim, int nlev=NUM_LEV);

  // Handle both NUM_LEV and NUM_LEV_P.
  template<int NUM_LEV_IN, typename... Properties>
  void register_field (ExecView<PT*[NP][NP][NUM_LEV_IN], Properties...> field, int nlev=NUM_LEV_IN);

  template<int NUM_LEV_IN, typename... Properties>
  void register_field (ExecView<PT**[NP][NP][NUM_LEV_IN], Properties...> field, int num_dims, int start_dim, int nlev=NUM_LEV_IN);

  template<int NUM_LEV_IN, int DIM, typename... Properties>
  void register_field (ExecView<PT*[DIM][NP][NP][NUM_LEV_IN], Properties...> field, int num_dims, int start_dim, int nlev=NUM_LEV_IN);

  // This registration method should be used for the exchange of min/max fields
  template<int DIM, typename... Properties>
  void register_min_max_fields (ExecView<PT*[DIM][2][NUM_LEV], Properties...> field_min_max, int num_dims, int start_dim);

  // Allow to swap a registered field with a new one. This allows to echange different views,
  // without having to rebuild all the MPI structures.
  template<int NUM_LEV_IN, typename... Properties>
  void replace_field (ExecView<PT*[NP][NP][NUM_LEV_IN], Properties...> field);
  template<int DIM, typename... Properties>
  void replace_field (ExecView<PT*[DIM][NP][NP][NUM_LEV], Properties...> field);

  bool can_replace_fields () const { return m_replace_allowed; }

  // Exchange all registered 2d and 3d fields
  void exchange ();
  void exchange (ExecViewUnmanaged<const Real * [NP][NP]> rspheremp);

  // Exchange all registered 1d fields, performing min/max operations with neighbors
  void exchange_min_max ();

  template<typename ptr_type, typename raw_type>
  struct Pointer {

    Pointer& operator= (const ptr_type& p) { ptr = p; return *this; }

    KOKKOS_FORCEINLINE_FUNCTION
    raw_type& operator[] (int i) { return ptr[i]; }

    KOKKOS_FORCEINLINE_FUNCTION
    ptr_type get() { return ptr; }

    ptr_type ptr;
  };

  // Perform the pack_and_send and recv_and_unpack for boundary exchange of 2d/3d fields
  void pack_and_send ();
  void recv_and_unpack ();

  // Perform the pack_and_send and recv_and_unpack for min/max boundary exchange of 1d fields
  void pack_and_send_min_max ();
  void recv_and_unpack_min_max ();

private:

  // Make MpiBuffersManager a friend, so it can call the method underneath
  void clear_buffer_views_and_requests () override;

  void build_buffer_views_and_requests () override;

  // We store some specs for each field, so that, if we replace it, we know exactly where in the
  // exec views below we need to replace the subviews. Only offset is ALWAYS needed, while
  // the others only make sense for vector/tensor fields
  struct FieldProps {
    int offset;        // Offset in m_Nd_fields
    int cmp_beg = -1;  // 1st cmp of field we exchange
    int cmp_end = -1;  // Number of cmps of field we exchange
    int out_dim = -1;  // If tensor field, the index of outer dim we're exchanging
  };

  std::map<std::string, FieldProps> m_fields_props;
  bool m_replace_allowed = true; // If all registered fields store a diff name, replacing will be allowed

  ExecViewManaged<ExecViewManaged<PT[2][NUM_LEV]>**>            m_1d_fields;
  ExecViewManaged<ExecViewManaged<ST[NP][NP]>**>                m_2d_fields;
  ExecViewManaged<ExecViewManaged<PT[NP][NP][NUM_LEV]>**>       m_3d_fields;
  ExecViewManaged<ExecViewManaged<PT[NP][NP][NUM_LEV_P]>**>     m_3d_int_fields;

  // send_buffer(..) points to the right area of one of the three buffers in the
  // buffers manager:
  // * local connection: point to local_buffer (on both send and recv views);
  // * shared connection: point to the corresponding mpi buffer;
  // * missing connection: point to the send/recv blackhole.

  // Index as m_*_?d_buffers(ifield, iconn), where iconn ranges over all
  // connections in this rank.
  ExecViewManaged<ExecViewUnmanaged<PT[2][NUM_LEV]>**>  m_send_1d_buffers;
  ExecViewManaged<ExecViewUnmanaged<PT[2][NUM_LEV]>**>  m_recv_1d_buffers;

  ExecViewManaged<ExecViewUnmanaged<ST*>**>        m_send_2d_buffers;
  ExecViewManaged<ExecViewUnmanaged<ST*>**>        m_recv_2d_buffers;

  ExecViewManaged<ExecViewUnmanaged<PT**>**>            m_send_3d_buffers;
  ExecViewManaged<ExecViewUnmanaged<PT**>**>            m_recv_3d_buffers;
  
  // TODO: optimize: you only need to pack/unpack the first entry of the NUM_LEV_P-th pack.
  //       This is because, if NUM_LEV!=NUM_LEV_P, then the NUM_LEV_P-th pack contains
  //       only one meaningful value (the rest is garbage, most likely nan's).
  ExecViewManaged<ExecViewUnmanaged<PT**>**>  m_send_3d_int_buffers;
  ExecViewManaged<ExecViewUnmanaged<PT**>**>  m_recv_3d_int_buffers;  

  // Only the impl knows about the raw pointer.
  void exchange(const ExecViewUnmanaged<const Real * [NP][NP]>* rspheremp);
public: // This is semantically private but must be public for nvcc.
  void recv_and_unpack(const ExecViewUnmanaged<const Real * [NP][NP]>* rspheremp);
};

using BoundaryExchange = BoundaryExchangeST<ScalarValue>;

// ============================ REGISTER METHODS ========================= //

// --- 2d fields --- //

template<typename ST>
template<typename... Properties>
void BoundaryExchangeST<ST>::register_field (ExecView<ST*[NP][NP], Properties...> field)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_2d_fields+1<=m_2d_fields.extent_int(1));
  assert (m_num_1d_fields==0);

  {
    auto l_num_2d_fields = m_num_2d_fields;
    auto l_2d_fields = m_2d_fields;
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, m_connectivity->get_num_local_elements()),
                         KOKKOS_LAMBDA(const int ie){
      l_2d_fields(ie, l_num_2d_fields) = Kokkos::subview(field, ie, ALL, ALL);
    });
  }

  ++m_num_2d_fields;
}

template<typename ST>
template<int DIM, typename... Properties>
void BoundaryExchangeST<ST>::register_field (ExecView<ST*[DIM][NP][NP], Properties...> field, int num_dims, int start_dim)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_2d_fields+num_dims<=m_2d_fields.extent_int(1));
  assert (m_num_1d_fields==0);

  {
    auto l_num_2d_fields = m_num_2d_fields;
    auto l_2d_fields = m_2d_fields;
    Kokkos::parallel_for(MDRangePolicy<ExecSpace, 2>({0, 0}, {m_connectivity->get_num_local_elements(), num_dims}, {1, 1}),
                         KOKKOS_LAMBDA(const int ie, const int idim){
        l_2d_fields(ie, l_num_2d_fields+idim) = Kokkos::subview(field, ie, start_dim+idim, ALL, ALL);
    });
  }

  m_num_2d_fields += num_dims;
}

template<typename ST>
template<typename... Properties>
void BoundaryExchangeST<ST>::register_field (ExecView<ST**[NP][NP], Properties...> field, int num_dims, int start_dim)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0);
  assert (start_dim+num_dims<=field.extent_int(1));
  assert (m_num_2d_fields+num_dims<=m_2d_fields.extent_int(1));
  assert (m_num_1d_fields==0);

  {
    auto l_num_2d_fields = m_num_2d_fields;
    auto l_2d_fields = m_2d_fields;
    Kokkos::parallel_for(MDRangePolicy<ExecSpace, 2>({0, 0}, {m_connectivity->get_num_local_elements(), num_dims}, {1, 1}),
                         KOKKOS_LAMBDA(const int ie, const int idim){
        l_2d_fields(ie, l_num_2d_fields+idim) = Kokkos::subview(field, ie, start_dim+idim, ALL, ALL);
    });
  }

  m_num_2d_fields += num_dims;
}

template<typename ST>
template<typename... Properties>
void BoundaryExchangeST<ST>::register_field (ExecView<ST***[NP][NP], Properties...> field, int idim_out, int num_dims, int start_dim)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0);
  assert (idim_out>=0 && idim_out<field.extent_int(1));
  assert (start_dim+num_dims<=field.extent_int(2));
  assert (m_num_2d_fields+num_dims<=m_2d_fields.extent_int(1));
  assert (m_num_1d_fields==0);

  {
    auto l_num_2d_fields = m_num_2d_fields;
    auto l_2d_fields = m_2d_fields;
    Kokkos::parallel_for(MDRangePolicy<ExecSpace, 2>({0, 0}, {m_connectivity->get_num_local_elements(), num_dims}, {1, 1}),
                         KOKKOS_LAMBDA(const int ie, const int idim){
        l_2d_fields(ie, l_num_2d_fields+idim) = Kokkos::subview(field, ie, idim_out, start_dim+idim, ALL, ALL);
    });
  }

  m_num_2d_fields += num_dims;
}

// --- 3d NUM_LEV fields --- //

template<typename ST>
template<int OUTER_DIM, int DIM, typename... Properties>
void BoundaryExchangeST<ST>::register_field (ExecView<PT*[OUTER_DIM][DIM][NP][NP][NUM_LEV], Properties...> field, int outer_dim, int num_dims, int start_dim, int nlev)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && outer_dim>=0 && DIM>0 && OUTER_DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_3d_fields+num_dims<=m_3d_fields.extent_int(1));
  assert (m_num_1d_fields==0);

  {
    auto l_num_3d_fields = m_num_3d_fields;
    auto l_3d_fields = m_3d_fields;
    Kokkos::parallel_for(MDRangePolicy<ExecSpace, 2>({0, 0}, {m_connectivity->get_num_local_elements(), num_dims}, {1, 1}),
                         KOKKOS_LAMBDA(const int ie, const int idim){
        l_3d_fields(ie, l_num_3d_fields+idim) = Kokkos::subview(field, ie, outer_dim, start_dim+idim, ALL, ALL, ALL);
    });
  }

  for (int i = 0; i < num_dims; ++i) m_3d_nlev_pack.push_back(nlev);
  m_num_3d_fields += num_dims;
}

template<typename ST>
template<typename... Properties>
void BoundaryExchangeST<ST>::register_field (ExecView<PT***[NP][NP][NUM_LEV], Properties...> field, int outer_dim, int num_dims, int start_dim, int nlev)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && outer_dim>=0);
  assert (start_dim+num_dims<=field.extent_int(2));
  assert (m_num_3d_fields+num_dims<=m_3d_fields.extent_int(1));
  assert (m_num_1d_fields==0);

  auto it_bool = m_fields_props.emplace(field.label(),{m_num_3d_fields,start_dim,num_dims,outer_dim});
  m_replace_allowed &= it_bool.second;

  {
    auto l_num_3d_fields = m_num_3d_fields;
    auto l_3d_fields = m_3d_fields;
    Kokkos::parallel_for(MDRangePolicy<ExecSpace, 2>({0, 0}, {m_connectivity->get_num_local_elements(), num_dims}, {1, 1}),
                         KOKKOS_LAMBDA(const int ie, const int idim){
        l_3d_fields(ie, l_num_3d_fields+idim) = Kokkos::subview(field, ie, outer_dim, start_dim+idim, ALL, ALL, ALL);
    });
  }

  for (int i = 0; i < num_dims; ++i) m_3d_nlev_pack.push_back(nlev);
  m_num_3d_fields += num_dims;
}

template<typename ST>
template<int NUM_LEV_IN, typename... Properties>
void BoundaryExchangeST<ST>::
register_field (ExecView<PT*[NP][NP][NUM_LEV_IN], Properties...> field, int nlev)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_1d_fields==0);

  auto it_bool = m_fields_props.emplace(field.label(),FieldProps{m_num_3d_fields});
  m_replace_allowed &= it_bool.second;

  int nelems = m_connectivity->get_num_local_elements();
  auto do_register = [=](auto fields, auto f, int& num_fields) {
    assert (num_fields+1<=fields.extent_int(1));
    auto policy = Kokkos::RangePolicy<ExecSpace>(0, nelems);
    auto lambda = KOKKOS_LAMBDA(const int ie){
      fields(ie, num_fields) = Kokkos::subview(field, ie, ALL, ALL, ALL);
    };
    Kokkos::parallel_for(policy,lambda);
    ++num_fields;
  };

  if constexpr (NUM_LEV_IN==NUM_LEV) {
    do_register(m_3d_fields,field,m_num_3d_fields);
    m_3d_nlev_pack.push_back(nlev);
  } else if constexpr (NUM_LEV_IN==NUM_LEV_P) {
    do_register(m_3d_int_fields,field,m_num_3d_int_fields);
  } else {
    EKAT_ERROR_MSG (
        "Error! Invalid value for template arg NUM_LEV_IN.\n"
        " - NUM_LEV_IN: " + std::to_string(NUM_LEV_IN) + "\n");
  }
}

// Workaround for SCREAM issue
//     https://github.com/E3SM-Project/scream/issues/2146
// concerning the Perlmutter GPU system.
template<typename ST, typename... Properties>
struct RegisterFieldImpl {
  using PT = PackType<ST>;

  int num_dims, num_3d_fields, start_dim;
  ExecViewManaged<ExecViewManaged<PT[NP][NP][NUM_LEV]>**> fields;
  ExecView<PT**[NP][NP][NUM_LEV], Properties...> field;
  KOKKOS_INLINE_FUNCTION void operator() (const int k) const {
    using Kokkos::ALL;
    const int ie = k / num_dims, idim = k % num_dims;
    fields(ie, num_3d_fields+idim) = Kokkos::subview(field, ie, start_dim+idim, ALL, ALL, ALL);
  }
};

template<typename ST>
template<int NUM_LEV_IN, typename... Properties>
void BoundaryExchangeST<ST>::
register_field (ExecView<PT**[NP][NP][NUM_LEV_IN], Properties...> field,
                int num_dims, int start_dim, int nlev)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_1d_fields==0);
  assert (num_dims>0 && start_dim>=0);
  assert (start_dim+num_dims<=field.extent_int(1));

  auto it_bool = m_fields_props.emplace(field.label(),FieldProps{m_num_3d_fields,start_dim,start_dim+num_dims});
  m_replace_allowed &= it_bool.second;

  int nelems = m_connectivity->get_num_local_elements();
  auto do_register = [=](auto fields, auto f, int& num_fields) {
    assert (num_fields+num_dims<=fields.extent_int(1));
    MDRangePolicy<ExecSpace, 2> policy({0, 0},{nelems,num_dims});
    auto lambda = KOKKOS_LAMBDA(const int ie,const int idim){
      fields(ie, num_fields+idim) = Kokkos::subview(field, ie, start_dim+idim, ALL, ALL, ALL);
    };
    Kokkos::parallel_for(policy,lambda);
    num_fields += num_dims;
  };

  if constexpr (NUM_LEV_IN==NUM_LEV) {
    do_register(m_3d_fields,field,m_num_3d_fields);
  } else if constexpr (NUM_LEV_IN==NUM_LEV_P) {
    do_register(m_3d_int_fields,field,m_num_3d_int_fields);
  } else {
    EKAT_ERROR_MSG (
        "Error! Invalid value for template arg NUM_LEV_IN.\n"
        " - NUM_LEV_IN: " + std::to_string(NUM_LEV_IN) + "\n");
  }

  m_3d_nlev_pack.push_back(nlev);
}

template<typename ST>
template<int NUM_LEV_IN, int DIM, typename... Properties>
void BoundaryExchangeST<ST>::
register_field (ExecView<PT*[DIM][NP][NP][NUM_LEV_IN], Properties...> field,
                int num_dims, int start_dim, int nlev)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_1d_fields==0);
  assert (num_dims>0 && start_dim>=0);
  assert (start_dim+num_dims<=DIM);

  auto it_bool = m_fields_props.emplace(field.label(),FieldProps{m_num_3d_fields,start_dim,start_dim+num_dims});
  m_replace_allowed &= it_bool.second;

  int nelems = m_connectivity->get_num_local_elements();
  auto do_register = [=](auto fields, auto f, int& num_fields) {
    assert (num_fields+num_dims<=fields.extent_int(1));
    MDRangePolicy<ExecSpace, 2> policy({0, 0},{nelems,num_dims});
    auto lambda = KOKKOS_LAMBDA(const int ie,const int idim){
      fields(ie, num_fields+idim) = Kokkos::subview(field, ie, start_dim+idim, ALL, ALL, ALL);
    };
    Kokkos::parallel_for(policy,lambda);
    num_fields += num_dims;
  };

  if constexpr (NUM_LEV_IN==NUM_LEV) {
    do_register(m_3d_fields,field,m_num_3d_fields);
  } else if constexpr (NUM_LEV_IN==NUM_LEV_P) {
    do_register(m_3d_int_fields,field,m_num_3d_int_fields);
  } else {
    EKAT_ERROR_MSG (
        "Error! Invalid value for template arg NUM_LEV_IN.\n"
        " - NUM_LEV_IN: " + std::to_string(NUM_LEV_IN) + "\n");
  }

  for (int i = 0; i < num_dims; ++i)
    m_3d_nlev_pack.push_back(nlev);
}

// --- min-max fields --- //

template<typename ST>
template<int DIM, typename... Properties>
void BoundaryExchangeST<ST>::
register_min_max_fields (ExecView<PT*[DIM][2][NUM_LEV], Properties...> field_min_max,
                         int num_dims, int start_dim)
{
  using Kokkos::ALL;

  // Sanity checks
  assert(m_registration_started && !m_registration_completed);
  assert(m_num_2d_fields == 0 && m_num_3d_fields == 0);

  {
    auto l_num_1d_fields = m_num_1d_fields;
    auto l_1d_fields     = m_1d_fields;
    Kokkos::parallel_for(MDRangePolicy<ExecSpace, 2>({0, 0}, {m_connectivity->get_num_local_elements(), num_dims}, {1, 1}),
                         KOKKOS_LAMBDA(const int ie, const int idim){
      l_1d_fields(ie, l_num_1d_fields+idim) = Kokkos::subview(field_min_max, ie, start_dim+idim, ALL, ALL);
    });
  }

  m_num_1d_fields += num_dims;
}

// ================================ REPLACE FIELDS ===================================== //

template<typename ST>
template<int NUM_LEV_IN, typename... Properties>
void BoundaryExchangeST<ST>::
replace_field (ExecView<PT*[NP][NP][NUM_LEV_IN], Properties...> field)
{
  EKAT_REQUIRE_MSG (m_registration_completed,
      "Error! Cannot replace a field until registration has ended.\n");
  EKAT_REQUIRE_MSG (m_replace_allowed,
      "Error! Attempting to replace a field, but replace was not allowed.\n");

  auto it = m_fields_props.find(field.label());
  EKAT_REQUIRE_MSG (it!=m_fields_props.end(),
      "Error! Could not locate offsets for this field.\n"
      " - field name: " + field.label() + "\n");

  const auto props = it->second;

  int offset   = props.offset;
  int nelems = m_connectivity->get_num_local_elements();
  auto do_replace = [&](auto fields, auto f_in) {
    auto lambda = KOKKOS_LAMBDA (int ie) {
      using Kokkos::ALL;
      fields(ie, offset) = Kokkos::subview(field, ie, ALL, ALL, ALL);
    };
    Kokkos::RangePolicy<ExecSpace> policy(0, nelems);

    Kokkos::parallel_for(policy,lambda);
  };

  if constexpr (NUM_LEV_IN==NUM_LEV) {
    do_replace(m_3d_fields,field);
  } else if constexpr (NUM_LEV_IN==NUM_LEV_P) {
    do_replace(m_3d_int_fields,field);
  } else {
    EKAT_ERROR_MSG (
        "Error! Invalid value for template arg NUM_LEV_IN.\n"
        " - NUM_LEV_IN: " + std::to_string(NUM_LEV_IN) + "\n");
  }
}

template<typename ST>
template<int DIM, typename... Properties>
void BoundaryExchangeST<ST>::
replace_field (ExecView<PT*[DIM][NP][NP][NUM_LEV], Properties...> field)
{
  EKAT_REQUIRE_MSG (m_registration_completed,
      "Error! Cannot replace a field until registration has ended.\n");
  EKAT_REQUIRE_MSG (m_replace_allowed,
      "Error! Attempting to replace a field, but replace was not allowed.\n");

  auto it = m_fields_props.find(field.label());
  EKAT_REQUIRE_MSG (it!=m_fields_props.end(),
      "Error! Could not locate offsets for this field.\n"
      " - field name: " + field.label() + "\n");

  const auto props = it->second;

  int cmp_beg  = props.cmp_beg;
  int cmp_end  = props.cmp_end;
  int offset   = props.offset;
  int nelems = m_connectivity->get_num_local_elements();
  auto l_3d_fields = m_3d_fields;
  auto lambda = KOKKOS_LAMBDA (int ie, int idim) {
    using Kokkos::ALL;
    l_3d_fields(ie, offset+idim) = Kokkos::subview(field, ie, idim, ALL, ALL, ALL);
  };
  MDRangePolicy<ExecSpace, 2> policy({0, cmp_beg},{nelems,cmp_end});

  Kokkos::parallel_for(policy,lambda);
}

} // namespace Homme

#endif // HOMMEXX_BOUNDARY_EXCHANGE_HPP
