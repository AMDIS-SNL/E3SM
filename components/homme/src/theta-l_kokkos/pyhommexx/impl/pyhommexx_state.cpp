#include "pyhommexx.hpp"
#include "pyhommexx_utils.hpp"

#include "Context.hpp"
#include "ElementsGeometry.hpp"
#include "ElementsState.hpp"
#include "Hommexx_Session.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "TimeLevel.hpp"
#include "Tracers.hpp"
#include "Types.hpp"

#include <ekat_assert.hpp>

#include <nanobind/ndarray.h>

namespace pyhommexx {

using namespace Homme;

void init_dp3d_from_ps ()
{
  auto& c = Context::singleton();
  auto state = c.get<ElementsState>();
  auto hvcoord = c.get<HybridVCoord>();
  int n0 = c.get<TimeLevel>().n0;

  auto p = get_default_team_policy<ExecSpace>(state.num_elems());
  auto init_dp = KOKKOS_LAMBDA(const TeamMember& team) {
    KernelVariables kv(team);
    hvcoord.compute_dp_ref (kv,Homme::subview(state.m_ps_v,kv.ie,n0),
                               Homme::subview(state.m_dp3d,kv.ie,n0));
  };
  Kokkos::parallel_for(p,init_dp);
}

template<typename ST>
void get_state_var_impl (nb::ndarray<double>& arr, const nb::str& name, const int which_tl)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsStateST<ST>> ();
  const auto& tracers = c.get<TracersST<ST>> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();

  std::vector<int> vector3dm_shape = {nelem,2,NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3dm_shape = {nelem,  NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3di_shape = {nelem,  NP,NP,NUM_INTERFACE_LEV};

  constexpr int flag_u   = 0;
  constexpr int flag_v   = 1;
  constexpr int flag_uv  = 2;
  constexpr int flag_vth = 3;
  constexpr int flag_dp  = 4;
  constexpr int flag_w   = 5;
  constexpr int flag_phi = 6;
  constexpr int flag_qv  = 7;
  std::map<std::string,int> which_map = {
    {"u",  flag_u},
    {"v",  flag_v},
    {"uv", flag_uv},
    {"vtheta_dp",flag_vth},
    {"dp", flag_dp},
    {"w",  flag_w},
    {"phi",flag_phi},
    {"qv", flag_qv},
  };

  // nanobind has no operator== with const char[]
  std::string n (name.c_str());
  if (n=="u" or n=="v" or n=="dp" or n=="qv" or n=="vtheta_dp") {
    check_shape(arr,scalar3dm_shape);
  } else if (n=="uv") {
    check_shape(arr,vector3dm_shape);
  } else if (n=="phi" or n=="w") {
    check_shape(arr,scalar3di_shape);
  } else {
    EKAT_ERROR_MSG("Unrecognized/unsupported state var '" + n + "'.\n");
  }
  assert ((int)arr.dtype().bits==64);

  int slice = -1;
  if (n=="qv") {
    EKAT_REQUIRE_MSG (which_tl==0,
        "[pyhommexx::set_state_var] Invalid time slice.\n"
        " - state var: " + n + "\n"
        " = time slice: " + std::to_string(which_tl) + "\n"
        " - valid values: 0\n");
  } else {
    EKAT_REQUIRE_MSG (which_tl>=-1 and which_tl<=1,
        "[pyhommexx::set_state_var] Invalid time slice.\n"
        " - state var: " + n + "\n"
        " = time slice: " + std::to_string(which_tl) + "\n"
        " - valid values: -1, 0, 1\n");
    slice = which_tl==0 ? tl.n0 : (which_tl==-1 ? tl.nm1 : tl.np1);
  }

  // They are unmanaged, so we can create all of them, even if we don't use them
  ExecViewUnmanaged<double*****> vec_mid_v (vp2dp(arr.data()),nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  scl_mid_v (vp2dp(arr.data()),nelem,  NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double****>  scl_int_v (vp2dp(arr.data()),nelem,  NP,NP,NUM_INTERFACE_LEV);

  auto uv  = state.m_v;
  auto vth = state.m_vtheta_dp;
  auto dp  = state.m_dp3d;
  auto w   = state.m_w_i;
  auto phi = state.m_phinh_i;
  auto Q  = tracers.Q;

  auto which = which_map.at(n);
  int nlev = (which==flag_phi or which==flag_w) ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  policy_t p ({0,0,0,0},{nelem,NP,NP,nlev});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    int lev = k / VECTOR_SIZE;
    int vec = k % VECTOR_SIZE;
    switch (which) {
      case flag_u:
        scl_mid_v(ie,ip,jp,k) = ADValue(uv(ie,slice,0,ip,jp,lev)[vec]); break;
      case flag_v:
        scl_mid_v(ie,ip,jp,k) = ADValue(uv(ie,slice,1,ip,jp,lev)[vec]); break;
      case flag_uv:
        vec_mid_v(ie,0,ip,jp,k) = ADValue(uv(ie,slice,0,ip,jp,lev)[vec]);
        vec_mid_v(ie,1,ip,jp,k) = ADValue(uv(ie,slice,1,ip,jp,lev)[vec]); break;
      case flag_vth:
        scl_mid_v(ie,ip,jp,k) = ADValue(vth(ie,slice,ip,jp,lev)[vec]); break;
      case flag_dp:
        scl_mid_v(ie,ip,jp,k) = ADValue(dp(ie,slice,ip,jp,lev)[vec]); break;
      case flag_w:
        scl_int_v(ie,ip,jp,k) = ADValue(w(ie,slice,ip,jp,lev)[vec]); break;
      case flag_phi:
        scl_int_v(ie,ip,jp,k) = ADValue(phi(ie,slice,ip,jp,lev)[vec]); break;
      case flag_qv:
        // qv is the FIRST tracer
        scl_mid_v(ie,ip,jp,k) = ADValue(Q(ie,0,ip,jp,lev)[vec]); break;
      default:
        Kokkos::abort("Unsupported value for 'which' in get_state_var.\n");
    }
  };
  Kokkos::parallel_for(p,copy);
}

void get_state_var (nb::ndarray<double>& arr, const nb::str& name, const nb::str& dtype, const int which_tl)
{
  std::string dtype_str(dtype.c_str());

  if (dtype_str=="real") {
    get_state_var_impl<Real>(arr,name,which_tl);
  } else if (dtype_str=="dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    get_state_var_impl<DpFadType>(arr,name,which_tl);
#else
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
      EKAT_ERROR_MSG("[get_state_var] Error! Unrecognized/unsupported dtype name.\n"
                    " - input dtype: " + dtype_str + "\n"
                    " - valid dtype(s): real, dpfad\n");
  }
}

template<typename ST>
void set_state_var_impl (const nb::ndarray<double>& arr, const nb::str& name, const int which_tl)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsStateST<ST>> ();
  const auto& tracers = c.get<TracersST<ST>> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();

  std::vector<int> vector3dm_shape = {nelem,2,NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3dm_shape = {nelem,  NP,NP,NUM_PHYSICAL_LEV};
  std::vector<int> scalar3di_shape = {nelem,  NP,NP,NUM_INTERFACE_LEV};

  constexpr int flag_u   = 0;
  constexpr int flag_v   = 1;
  constexpr int flag_uv  = 2;
  constexpr int flag_vth = 3;
  constexpr int flag_dp  = 4;
  constexpr int flag_w   = 5;
  constexpr int flag_phi = 6;
  constexpr int flag_qv  = 7;
  std::map<std::string,int> which_map = {
    {"u",  flag_u},
    {"v",  flag_v},
    {"uv", flag_uv},
    {"vtheta_dp",flag_vth},
    {"dp", flag_dp},
    {"w",  flag_w},
    {"phi",flag_phi},
    {"qv", flag_qv},
  };

  // nanobind has no operator== with const char[]
  std::string n (name.c_str());
  if (n=="u" or n=="v" or n=="dp" or n=="qv" or n=="vtheta_dp") {
    check_shape(arr,scalar3dm_shape);
  } else if (n=="uv") {
    check_shape(arr,vector3dm_shape);
  } else if (n=="phi" or n=="w") {
    check_shape(arr,scalar3di_shape);
  } else {
    EKAT_ERROR_MSG("Unrecognized/unsupported state var '" + n + "'.\n");
  }
  assert ((int)arr.dtype().bits==64);

  int slice = -1;
  if (n=="qv") {
    EKAT_REQUIRE_MSG (which_tl==0,
        "[pyhommexx::set_state_var] Invalid time slice.\n"
        " - state var: " + n + "\n"
        " = time slice: " + std::to_string(which_tl) + "\n"
        " - valid values: 0\n");
  } else {
    EKAT_REQUIRE_MSG (which_tl>=-1 and which_tl<=1,
        "[pyhommexx::set_state_var] Invalid time slice.\n"
        " - state var: " + n + "\n"
        " = time slice: " + std::to_string(which_tl) + "\n"
        " - valid values: -1, 0, 1\n");
    slice = which_tl==0 ? tl.n0 : (which_tl==-1 ? tl.nm1 : tl.np1);
  }

  // They are unmanaged, so we can create all of them, even if we don't use them
  ExecViewUnmanaged<const double*****> vec_mid_v (vp2cdp(arr.data()),nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  scl_mid_v (vp2cdp(arr.data()),nelem,  NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  scl_int_v (vp2cdp(arr.data()),nelem,  NP,NP,NUM_INTERFACE_LEV);

  auto uv  = state.m_v;
  auto vth = state.m_vtheta_dp;
  auto dp  = state.m_dp3d;
  auto w   = state.m_w_i;
  auto phi = state.m_phinh_i;
  auto qv  = tracers.Q;

  auto which = which_map.at(n);
  int nlev = (which==flag_phi or which==flag_w) ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
  policy_t p ({0,0,0,0},{nelem,NP,NP,nlev});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    int lev = k / VECTOR_SIZE;
    int vec = k % VECTOR_SIZE;
    switch (which) {
      case flag_u:
        uv(ie,slice,0,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      case flag_v:
        uv(ie,slice,1,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      case flag_uv:
        uv(ie,slice,0,ip,jp,lev)[vec] = vec_mid_v(ie,0,ip,jp,k);
        uv(ie,slice,1,ip,jp,lev)[vec] = vec_mid_v(ie,1,ip,jp,k); break;
      case flag_vth:
        vth(ie,slice,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      case flag_dp:
        dp(ie,slice,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      case flag_w:
        w(ie,slice,ip,jp,lev)[vec] = scl_int_v(ie,ip,jp,k); break;
      case flag_phi:
        phi(ie,slice,ip,jp,lev)[vec] = scl_int_v(ie,ip,jp,k); break;
      case flag_qv:
        // qv is the FIRST tracer
        qv(ie,0,ip,jp,lev)[vec] = scl_mid_v(ie,ip,jp,k); break;
      default:
        Kokkos::abort("Unsupported value for 'which' in set_state_var.\n");
    }
  };
  Kokkos::parallel_for(p,copy);
}

template<typename ST>
void set_state_var_value_impl (const double value, const nb::str& name, const int which_tl)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsStateST<ST>> ();
  const auto& tracers = c.get<TracersST<ST>> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();

  constexpr int flag_u   = 0;
  constexpr int flag_v   = 1;
  constexpr int flag_uv  = 2;
  constexpr int flag_vth = 3;
  constexpr int flag_dp  = 4;
  constexpr int flag_w   = 5;
  constexpr int flag_phi = 6;
  constexpr int flag_qv  = 7;
  std::map<std::string,int> which_map = {
    {"u",  flag_u},
    {"v",  flag_v},
    {"uv", flag_uv},
    {"vtheta_dp",flag_vth},
    {"dp", flag_dp},
    {"w",  flag_w},
    {"phi",flag_phi},
    {"qv", flag_qv},
  };

  // nanobind has no operator== with const char[]
  std::string n (name.c_str());

  int slice = -1;
  if (n=="qv") {
    EKAT_REQUIRE_MSG (which_tl==0,
        "[pyhommexx::set_state_var] Invalid time slice.\n"
        " - state var: " + n + "\n"
        " = time slice: " + std::to_string(which_tl) + "\n"
        " - valid values: 0\n");
  } else {
    EKAT_REQUIRE_MSG (which_tl>=-1 and which_tl<=1,
        "[pyhommexx::set_state_var] Invalid time slice.\n"
        " - state var: " + n + "\n"
        " = time slice: " + std::to_string(which_tl) + "\n"
        " - valid values: 0\n");
    slice = which_tl==0 ? tl.n0 : (which_tl==-1 ? tl.nm1 : tl.np1);
  }

  auto uv  = state.m_v;
  auto vth = state.m_vtheta_dp;
  auto dp  = state.m_dp3d;
  auto w   = state.m_w_i;
  auto phi = state.m_phinh_i;
  auto qv  = tracers.Q;

  auto which = which_map.at(n);
  const auto A = Kokkos::ALL();
  ST v (value);
  switch (which) {
    case flag_u:
      Kokkos::deep_copy(Kokkos::subview(uv,A,slice,0,A,A,A),v); break;
    case flag_v:
      Kokkos::deep_copy(Kokkos::subview(uv,A,slice,1,A,A,A),v); break;
    case flag_uv:
      Kokkos::deep_copy(Kokkos::subview(uv,A,slice,0,A,A,A),v);
      Kokkos::deep_copy(Kokkos::subview(uv,A,slice,1,A,A,A),v); break;
    case flag_vth:
      Kokkos::deep_copy(Kokkos::subview(vth,A,slice,A,A,A),v); break;
    case flag_dp:
      Kokkos::deep_copy(Kokkos::subview(dp,A,slice,A,A,A),v); break;
    case flag_w:
      Kokkos::deep_copy(Kokkos::subview(w,A,slice,A,A,A),v); break;
    case flag_phi:
      Kokkos::deep_copy(Kokkos::subview(phi,A,slice,A,A,A),v); break;
    case flag_qv:
      // qv is the FIRST tracer
      Kokkos::deep_copy(Kokkos::subview(qv,A,0,A,A,A),v); break;
    default:
      EKAT_ERROR_MSG("Unsupported value for 'which' in set_state_var_value.\n");
  }
}

void set_state_var (const nb::ndarray<double>& arr, const nb::str& name, const nb::str& dtype, const int tl)
{
  std::string dtype_str(dtype.c_str());

  if (dtype_str=="real") {
    set_state_var_impl<Real>(arr,name,tl);
  } else if (dtype_str=="dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    set_state_var_impl<DpFadType>(arr,name,tl);
#else
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
      EKAT_ERROR_MSG("[set_state_var] Error! Unrecognized/unsupported dtype name.\n"
                    " - input dtype: " + dtype_str + "\n"
                    " - valid dtype(s): real, dpfad\n");
  }
}

void set_state_var_value (const double value, const nb::str& name, const nb::str& dtype, const int tl)
{
  std::string dtype_str(dtype.c_str());

  if (dtype_str=="real") {
    set_state_var_value_impl<Real>(value,name,tl);
  } else if (dtype_str=="dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    set_state_var_value_impl<DpFadType>(value,name,tl);
#else
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
      EKAT_ERROR_MSG("[set_state_var_value] Error! Unrecognized/unsupported dtype name.\n"
                    " - input dtype: " + dtype_str + "\n"
                    " - valid dtype(s): real, dpfad\n");
  }
}

void get_state_var_dp_sens (nb::ndarray<double>& arr, const nb::str& name)
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  const auto& c = Context::singleton();
  if (not c.has<ElementsStateST<DpFadType>>() or not c.has<TracersST<DpFadType>>()) {
    EKAT_ERROR_MSG("Cannot retrieve dp sensitivities. No DpFadType tracers/state present");
  }
  const auto& state = c.get<ElementsStateST<DpFadType>> ();
  const auto& tracers = c.get<TracersST<DpFadType>> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();
  const int np1 = tl.np1;
  const int qnp1 = tl.np1_qdp;

  std::vector<int> vector3dm_shape = {nelem,2,NP,NP,NUM_PHYSICAL_LEV,HOMMEXX_DP_SFAD_SIZE};
  std::vector<int> scalar3dm_shape = {nelem,  NP,NP,NUM_PHYSICAL_LEV,HOMMEXX_DP_SFAD_SIZE};
  std::vector<int> scalar3di_shape = {nelem,  NP,NP,NUM_INTERFACE_LEV,HOMMEXX_DP_SFAD_SIZE};

  constexpr int flag_u   = 0;
  constexpr int flag_v   = 1;
  constexpr int flag_uv  = 2;
  constexpr int flag_vth = 3;
  constexpr int flag_dp  = 4;
  constexpr int flag_w   = 5;
  constexpr int flag_phi = 6;
  constexpr int flag_qv  = 7;
  std::map<std::string,int> which_map = {
    {"u",  flag_u},
    {"v",  flag_v},
    {"uv", flag_uv},
    {"vtheta_dp",flag_vth},
    {"dp", flag_dp},
    {"w",  flag_w},
    {"phi",flag_phi},
    {"qv", flag_qv},
  };

  // nanobind has no operator== with const char[]
  std::string n (name.c_str());
  if (n=="u" or n=="v" or n=="dp" or n=="qv" or n=="vtheta_dp") {
    check_shape(arr,scalar3dm_shape);
  } else if (n=="uv") {
    check_shape(arr,vector3dm_shape);
  } else if (n=="phi" or n=="w") {
    check_shape(arr,scalar3di_shape);
  } else {
    EKAT_ERROR_MSG("Unrecognized/unsupported state var '" + n + "'.\n");
  }
  assert ((int)arr.dtype().bits==64);

  // They are unmanaged, so we can create all of them, even if we don't use them
  ExecViewUnmanaged<double******> vec_mid_v (vp2dp(arr.data()),nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double*****>  scl_mid_v (vp2dp(arr.data()),nelem,  NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<double*****>  scl_int_v (vp2dp(arr.data()),nelem,  NP,NP,NUM_INTERFACE_LEV);

  auto uv  = state.m_v;
  auto vth = state.m_vtheta_dp;
  auto dp  = state.m_dp3d;
  auto w   = state.m_w_i;
  auto phi = state.m_phinh_i;
  auto qv  = tracers.Q;

  auto which = which_map.at(n);
  int nlev = (which==flag_phi or which==flag_w) ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<5>>;
  policy_t p ({0,0,0,0,0},{nelem,NP,NP,nlev,HOMMEXX_DP_SFAD_SIZE});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k, int ider) {
    int lev = k / VECTOR_SIZE;
    int vec = k % VECTOR_SIZE;
    switch (which) {
      case flag_u:
        scl_mid_v(ie,ip,jp,k,ider) = uv(ie,np1,0,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_v:
        scl_mid_v(ie,ip,jp,k,ider) = uv(ie,np1,1,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_uv:
        vec_mid_v(ie,0,ip,jp,k,ider) = uv(ie,np1,0,ip,jp,lev)[vec].fastAccessDx(ider);
        vec_mid_v(ie,1,ip,jp,k,ider) = uv(ie,np1,1,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_vth:
        scl_mid_v(ie,ip,jp,k,ider) = vth(ie,np1,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_dp:
        scl_mid_v(ie,ip,jp,k,ider) = dp(ie,np1,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_w:
        scl_int_v(ie,ip,jp,k,ider) = w(ie,np1,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_phi:
        scl_int_v(ie,ip,jp,k,ider) = phi(ie,np1,ip,jp,lev)[vec].fastAccessDx(ider); break;
      case flag_qv:
        scl_mid_v(ie,ip,jp,k,ider) = qv(ie,qnp1,ip,jp,lev)[vec].fastAccessDx(ider); break;
      default:
        Kokkos::abort("Unsupported value for 'which' in get_state_var.\n");
    }
  };
  Kokkos::parallel_for(p,copy);
#else
  EKAT_ERROR_MSG("[pyhommexx::get_state_var_sens] not implemented unless HOMMEXX_ENABLE_FAD_TYPES is defined.\n");
  (void)arr;
  (void)name;
#endif
}

template<typename ST>
void perturb_state_var_impl (const nb::str& name,
                             const nb::ndarray<double>& delta,
                             const double factor)
{
  const auto& c = Context::singleton();
  const auto& state = c.get<ElementsStateST<ST>> ();
  const auto& geo = c.get<ElementsGeometry> ();
  const auto& tracers = c.get<TracersST<ST>> ();
  const auto& tl = c.get<TimeLevel> ();

  const int nelem = state.num_elems();
  const int n0 = tl.n0;
  const int qn0 = tl.n0_qdp;

  constexpr int flag_u   = 0;
  constexpr int flag_v   = 1;
  constexpr int flag_uv  = 2;
  constexpr int flag_vth = 3;
  constexpr int flag_dp  = 4;
  constexpr int flag_w   = 5;
  constexpr int flag_phi = 6;
  constexpr int flag_qv  = 7;
  std::map<std::string,int> which_map = {
    {"u",  flag_u},
    {"v",  flag_v},
    {"uv", flag_uv},
    {"vtheta_dp",flag_vth},
    {"dp", flag_dp},
    {"w",  flag_w},
    {"phi",flag_phi},
    {"qv", flag_qv},
  };

  auto uv  = state.m_v;
  auto vth = state.m_vtheta_dp;
  auto dp  = state.m_dp3d;
  auto w   = state.m_w_i;
  auto phi = state.m_phinh_i;
  auto qv  = tracers.Q;

  std::string n (name.c_str());
  auto which = which_map.at(n);
  int nlev = (which==flag_phi or which==flag_w) ? NUM_INTERFACE_LEV : NUM_PHYSICAL_LEV;
  using policy_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;

  auto ptr = vp2cdp(delta.data());
  ExecViewUnmanaged<const double*****> vec_mid (ptr,nelem,2,NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  scl_mid (ptr,nelem,  NP,NP,NUM_PHYSICAL_LEV);
  ExecViewUnmanaged<const double****>  scl_int (ptr,nelem,  NP,NP,NUM_INTERFACE_LEV);

  ST p = factor;
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  if constexpr (Sacado::IsFad<ST>::value) {
    p.fastAccessDx(0) = 1;
  }
#endif
  policy_t policy ({0,0,0,0},{nelem,NP,NP,nlev});
  auto copy = KOKKOS_LAMBDA (int ie, int ip, int jp, int k) {
    int lev = k / VECTOR_SIZE;
    int vec = k % VECTOR_SIZE;

    switch (which) {
      case flag_u:
        uv(ie,n0,0,ip,jp,lev)[vec] *= (1+p*scl_mid(ie,ip,jp,k)); break;
      case flag_v:
        uv(ie,n0,1,ip,jp,lev)[vec] *= (1+p*scl_mid(ie,ip,jp,k)); break;
      case flag_uv:
        uv(ie,n0,0,ip,jp,lev)[vec] *= (1+p*vec_mid(ie,0,ip,jp,k)); break;
        uv(ie,n0,1,ip,jp,lev)[vec] *= (1+p*vec_mid(ie,1,ip,jp,k)); break;
      case flag_vth:
        vth(ie,n0,ip,jp,lev)[vec]  *= (1+p*scl_mid(ie,ip,jp,k)); break;
      case flag_dp:
        dp(ie,n0,ip,jp,lev)[vec]   *= (1+p*scl_mid(ie,ip,jp,k)); break;
      case flag_w:
        w(ie,n0,ip,jp,lev)[vec]    *= (1+p*scl_int(ie,ip,jp,k)); break;
      case flag_phi:
        phi(ie,n0,ip,jp,lev)[vec]  *= (1+p*scl_int(ie,ip,jp,k)); break;
      case flag_qv:
        qv(ie,n0,ip,jp,lev)[vec]   *= (1+p*scl_mid(ie,ip,jp,k)); break;
      default:
        Kokkos::abort("Unsupported value for 'which' in get_state_var.\n");
    }
  };
  Kokkos::parallel_for(policy,copy);
}

void perturb_state_var (const nb::str& name,
                        const nb::ndarray<double>& delta,
                        const double factor,
                        const nb::str& dtype)
{
  std::string dtype_str(dtype.c_str());

  if (dtype_str=="real") {
    perturb_state_var_impl<Real>(name,delta,factor);
  } else if (dtype_str=="dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    perturb_state_var_impl<DpFadType>(name,delta,factor);
#else
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
      EKAT_ERROR_MSG("[perturb_state_var] Error! Unrecognized/unsupported dtype name.\n"
                    " - input dtype: " + dtype_str + "\n"
                    " - valid dtype(s): real, dpfad\n");
  }
}

void copy_state (const nb::str& from_dtype, const nb::str& to_dtype)
{
  auto& c = Context::singleton();
  const auto& tl = c.get<TimeLevel>();
  std::string from_dtype_str(from_dtype.c_str());
  std::string to_dtype_str(to_dtype.c_str());
  if (from_dtype_str=="real") {
    const auto& from_st = c.get<ElementsStateST<Real>>();
    const auto& from_tr = c.get<TracersST<Real>>();
    if (to_dtype_str=="real") {
      return;
    } else if (to_dtype_str=="dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
      auto& to_st = c.get<ElementsStateST<DpFadType>>();
      auto& to_tr = c.get<TracersST<DpFadType>>();
      to_st.import_values(from_st,tl.nm1);
      to_st.import_values(from_st,tl.n0);
      to_st.import_values(from_st,tl.np1);
      to_tr.import_values(from_tr,tl.n0_qdp);
      to_tr.import_values(from_tr,tl.np1_qdp);
#else
      EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
    } else {
      EKAT_ERROR_MSG(
        "[copy_state] Error! Unrecognized/unsupported to_dtype name.\n"
        " - input dtype: " + to_dtype_str + "\n"
        " - valid dtype(s): real, dpfad\n");
    }
  } else if (from_dtype_str=="dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    auto& from_st = c.get<ElementsStateST<DpFadType>>();
    auto& from_tr = c.get<TracersST<DpFadType>>();
    if (to_dtype_str=="real") {
      auto& to_st = c.get<ElementsStateST<Real>>();
      auto& to_tr = c.get<TracersST<Real>>();
      to_st.import_values(from_st,tl.nm1);
      to_st.import_values(from_st,tl.n0);
      to_st.import_values(from_st,tl.np1);
      to_tr.import_values(from_tr,tl.n0_qdp);
      to_tr.import_values(from_tr,tl.np1_qdp);
    } else if (to_dtype_str=="dpfad") {
      return;
    } else {
      EKAT_ERROR_MSG(
        "[copy_state] Error! Unrecognized/unsupported to_dtype name.\n"
        " - input dtype: " + to_dtype_str + "\n"
        " - valid dtype(s): real, dpfad\n");
    }
#else
      EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
    EKAT_ERROR_MSG(
      "[copy_state] Error! Unrecognized/unsupported from_dtype name.\n"
      " - input dtype: " + from_dtype_str + "\n"
      " - valid dtype(s): real, dpfad\n");
  }
}

} // namespace pyhommexx
