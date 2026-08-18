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

} // namespace pyhommexx

#endif // PYHOMMEXX_HPP
