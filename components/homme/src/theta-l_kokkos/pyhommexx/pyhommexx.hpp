#ifndef PYHOMMEXX_HPP
#define PYHOMMEXX_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace pyhommexx {

namespace nb = nanobind;

// General session functions
void init_session (const bool do_print_to_screen = true);
void enable_scalar_type (const nb::str& dtype);
void toggle_screen_output (const bool enabled);
double get_phys_constant (const nb::str& name);
void finalize();

// Input parameters utilities
void read_params (const nb::str& nml_filename);
nb::dict get_params();
void set_params(const nb::dict& params);

// Model geometry utilities
int get_nelemd();
void get_num_unique_pts (nb::ndarray<int>& n);
void get_unique_pts (nb::ndarray<int>& ia,
                     nb::ndarray<int>& ja);
void get_dyn_latlon (nb::ndarray<double>& lat,
                     nb::ndarray<double>& lon);

// State handling utils
void get_state_var (nb::ndarray<double>& arr, const nb::str& name, const nb::str& dtype, const int tl);
void get_state_var_dp_sens (nb::ndarray<double>& arr, const nb::str& name);
void set_state_var (const nb::ndarray<double>& arr, const nb::str& name, const nb::str& dtype, const int tl);
void set_state_var_value (const double value, const nb::str& name, const nb::str& dtype, const int tl);
void init_dp3d_from_ps ();
void copy_state (const nb::str& from_dtype, const nb::str& to_dtype);

// Tracers (qdp). The array shape is (nelem,qsize,np,np,nlev) (with an extra trailing
// dimension of size DP_SFAD_SIZE for the *_dp_sens versions). Unlike the dynamics state,
// qdp has no time level that moves with n0/np1: it is made of two slots that flip at each tracers
// step, and the caller must say which one it wants. prim_step reads the tracers from the first slot
// given by get_prim_step_qdp_slots() and writes them to the second, but the answer
// is only valid *before* calling prim_step (after it, the TimeLevel is already updated,
// and the slots are the usual n0_qdp/np1_qdp).
nb::tuple get_prim_step_qdp_slots ();
void get_qdp (nb::ndarray<double>& arr, const nb::str& dtype, const int slot);
void set_qdp (const nb::ndarray<double>& arr, const nb::str& dtype, const int slot);
void get_qdp_dp_sens (nb::ndarray<double>& arr, const int slot);
void set_qdp_dp_sens (const nb::ndarray<double>& arr, const int slot);

// Perturbs state array as arr *= (1+C(lat,lon)*eps), where eps~N(0,1),
// and C(lat,lon) = p_max*exp(-d/2*sigma^2), where d is the distance
// of the 2d point (lat,lon) from (lat0,lon0).
void perturb_state_var (const nb::str& name,
                        const nb::ndarray<double>& delta,
                        const double factor,
                        const nb::str& dtype);

// Forcing handling utils. Recognized names: fm, fm_x, fm_y, fm_z, fvtheta, fphi.
// ElementsForcingST<ST> has no time-level dimension. Values persist across
// forward()/apply_dynamics_forcing() calls until overwritten from Python.
void get_forcing (nb::ndarray<double>& arr, const nb::str& name, const nb::str& dtype);
void set_forcing (const nb::ndarray<double>& arr, const nb::str& name, const nb::str& dtype);
void set_forcing_value (const double value, const nb::str& name, const nb::str& dtype);

// Apply the (states, i.e. non-tracer) forcing arrays to the state at tl.n0.
// Equivalent to CamForcing.cpp's apply_cam_forcing_dynamics for Real, and
// analogous for DpFadType.
void apply_dynamics_forcing (const double dt, const nb::str& dtype);

// Init/run a functor or the whole model
void run_functor(const nb::str& name,const nb::dict& params,const nb::str& dtype);
void model_init ();
void forward(const double dt);

// One call to prim_step (dt_tracer_factor dynamics steps, followed by a tracers step), on the
// objects of the given scalar type. The time levels of the TimeLevel in the Context are updated
// as in the f90/C++ drivers (see get_prim_step_qdp_slots for the tracers' ones).
// For dtype!=real: only ttype10_imex, Eulerian transport, no diagnostics, no adjoint taping.
void prim_step (const double dt, const nb::str& dtype, const bool compute_diagnostics);

// Dynamics advance (fwd/adj). Only ttype10_imex (time_step_type=10) has an adjoint.
// Neither routine rotates time levels: prim_advance_exp reads n0 (and nm1 as scratch)
// and writes np1. Call update_dynamics_levels() to move to the next step.
void update_dynamics_levels ();
void enable_adjoint ();
void prim_advance_exp (const double dt, const bool compute_diagnostics, const nb::str& dtype);
void prim_advance_adj (const double dt);

// Adjoint state utils. Recognized names: u, v, uv, vtheta_dp, dp, w, phi
void zero_adj_state ();
void get_adj_state_var (nb::ndarray<double>& arr, const nb::str& name);
void set_adj_state_var (nb::ndarray<double>& arr, const nb::str& name);

// Adjoint of prim_step (Eulerian tracers). Requires enable_adjoint() (with qsize>0, transport_alg=0)
// and a build with HOMMEXX_ENABLE_FAD_TYPES=ON.
//  - prim_step_checkpoint_take() must be called right BEFORE the prim_step to differentiate
//    (it captures dynamics state, qdp and time levels, all Real-typed).
//  - the adjoint state (see above) and the adjoint qdp hold dJ/d(state after prim_step) on entry
//    and dJ/d(state before prim_step) on exit, in place.
//  - prim_step_adj recomputes the fwd run from the checkpoint, and puts the live (Real) state
//    and time levels back at the beginning of the prim_step when it's done.
// The adjoint qdp has shape (nelem,qsize,np,np,nlev)
void prim_step_checkpoint_take ();
void prim_step_adj (const double dt);
void zero_adj_qdp ();
void get_adj_qdp (nb::ndarray<double>& arr);
void set_adj_qdp (const nb::ndarray<double>& arr);

} // namespace pyhommexx

#endif // PYHOMMEXX_HPP
