#include "pyhommexx.hpp"

#include <nanobind/nanobind.h>

namespace pyhommexx {

NB_MODULE (pyhommexx,m) {

  m.doc() = "Python interface to theta-l_kokkos Hommexx target";

  // General session functions
  m.def("init_session",&init_session,
        "Initialize the session. If do_print_to_screen is True, print to screen.",
        nb::arg("do_print_to_screen") = true);
  m.def("enable_scalar_type",&enable_scalar_type);
  m.def("toggle_screen_output",&toggle_screen_output);
  m.def("get_phys_constant",&get_phys_constant);
  m.def("finalize",&finalize);

  // Input parameters utilities
  m.def("read_params",&read_params);
  m.def("get_params",&get_params);
  m.def("set_params",&set_params,
      "Overrides some of the parameters read from the namelist. Valid keys:\n"
      " - alloc_sphere_coords (bool)\n"
      " - nu_p (float): to be set after read_params and before model_init. Meant for tests that need a value\n"
      "   the init checks reject (nu_p=0 for the adjoint of prim_step, which does not support nu_p>0 yet).");

  // Model geometry utilities
  m.def("get_nelemd",&get_nelemd);
  m.def("get_num_unique_pts",&get_num_unique_pts);
  m.def("get_unique_pts",&get_unique_pts);
  m.def("get_dyn_latlon",&get_dyn_latlon);

  // State handling utils
  m.def("get_state_var",&get_state_var,
      "Retrieves a state variable from the data structure with the given scalar type, at the given time slice.\n"
      "The time slice value can be -1 (nm1) 0 (n0) or 1 (np1) for state vars, and 0 for tracers (no time slices).\n"
      "By default, we retrieve slice np1 for states",
        nb::arg("arr"),
        nb::arg("name"),
        nb::arg("dtype") = "real",
        nb::arg("tl") = 1);
  m.def("get_state_var_dp_sens",&get_state_var_dp_sens);
  m.def("get_prim_step_qdp_slots",&get_prim_step_qdp_slots,
      "Returns (in,out): the qdp slots prim_step will read the tracers from, and write them to.\n"
      "Only valid BEFORE calling prim_step (afterward, the TimeLevel has been updated, and these\n"
      "are the current n0_qdp/np1_qdp).");
  m.def("get_qdp",&get_qdp,
      "Copies qdp, at the given slot (0 or 1), of the given scalar type into arr, of shape (nelem,qsize,np,np,nlev).",
        nb::arg("arr"), nb::arg("dtype") = "real", nb::arg("slot"));
  m.def("set_qdp",&set_qdp,
      "Sets qdp, at the given slot (0 or 1), of the given scalar type from arr, of shape (nelem,qsize,np,np,nlev).\n"
      "For 'dpfad', the derivatives are not modified (see set_qdp_dp_sens).",
        nb::arg("arr"), nb::arg("dtype") = "real", nb::arg("slot"));
  m.def("get_qdp_dp_sens",&get_qdp_dp_sens,
      "Copies the derivs of the dpfad qdp, at the given slot, into arr, of shape (nelem,qsize,np,np,nlev,DP_SFAD_SIZE).",
        nb::arg("arr"), nb::arg("slot"));
  m.def("set_qdp_dp_sens",&set_qdp_dp_sens,
      "Sets the derivs of the dpfad qdp, at the given slot, from arr, of shape (nelem,qsize,np,np,nlev,DP_SFAD_SIZE).\n"
      "The values of qdp are not modified (see set_qdp).",
        nb::arg("arr"), nb::arg("slot"));
  m.def("copy_state",&copy_state);
  m.def("init_dp3d_from_ps",&init_dp3d_from_ps);
  m.def("set_state_var",&set_state_var,
      "Sets a state variable in the data structure with the given scalar type, at the given time slice.\n"
      "The time slice value can be -1 (nm1) 0 (n0) or 1 (np1) for state vars, and 0 for tracers (no time slices).\n"
      "By default, we set slice n0 for states",
        nb::arg("arr"),
        nb::arg("name"),
        nb::arg("dtype") = "real",
        nb::arg("tl") = 0);
  m.def("set_state_var_value",&set_state_var_value,
      "Sets a state variable in the data structure with the given scalar type, at the given time slice.\n"
      "The time slice value can be -1 (nm1) 0 (n0) or 1 (np1) for state vars, and 0 for tracers (no time slices).\n"
      "By default, we set slice n0 for states",
        nb::arg("value"),
        nb::arg("name"),
        nb::arg("dtype") = "real",
        nb::arg("tl") = 0);
  m.def("perturb_state_var",&perturb_state_var,
      "Perturbs the given variable of the given scalar type by multiply it by a given perturbation field.\n"
      "For 'dpfad' scalar, initialize the 1st dx/dp fad deriv, following x = x*(1+p*delta)\n"
      "The input relative perturbation field 'delta' must have the same shape as the state we perturb\n",
        nb::arg("name"),
        nb::arg("delta"),
        nb::arg("factor"),
        nb::arg("dtype") = "real");

  // Forcing handling
  m.def("get_forcing",&get_forcing,
      "Retrieves a forcing array from ElementsForcingST<ST>.\n"
      "Recognized names: fm (vector), fm_x/fm_y/fm_z (scalar components), fvtheta, fphi.\n"
      "fphi is at interfaces; the rest are at midpoints.",
        nb::arg("arr"),
        nb::arg("name"),
        nb::arg("dtype") = "real");
  m.def("set_forcing",&set_forcing,
      "Writes a forcing array into ElementsForcingST<ST>.\n"
      "Values persist across forward()/apply_dynamics_forcing() calls until overwritten.",
        nb::arg("arr"),
        nb::arg("name"),
        nb::arg("dtype") = "real");
  m.def("set_forcing_value",&set_forcing_value,
      "Fills a forcing array with a single constant value.",
        nb::arg("value"),
        nb::arg("name"),
        nb::arg("dtype") = "real");
  m.def("apply_dynamics_forcing",&apply_dynamics_forcing,
      "Apply the (non-tracer) forcing arrays to the state at tl.n0 in-place:\n"
      "  u,v,w      += dt * fm\n"
      "  vtheta_dp  += dt * fvtheta\n"
      "  phi        += dt * fphi\n"
      "then re-imposes the surface w boundary condition. Isolates the states_forcing\n"
      "portion of apply_cam_forcing (bypasses the tracer path and the RK loop).",
        nb::arg("dt"),
        nb::arg("dtype") = "real");

  // Init/run a functor or the whole model
  m.def("model_init",&model_init);
  m.def("run_functor",&run_functor,
      "Runs a named functor with the given parameters and scalar type.\n"
      "The dtype arg specifies which instantiation of the functor to use",
        nb::arg("name"), nb::arg("params"), nb::arg("dtype") = "real");
  m.def("forward",&forward);
  m.def("prim_step",&prim_step,
      "Runs prim_step(dt) on the objects with the given scalar type: dt_tracer_factor dynamics steps\n"
      "followed by one tracers step. Time levels are updated as in the drivers (no dynamics levels rotation\n"
      "after the last dynamics step, the tracers slots flip).\n"
      "For dtype='dpfad' (fwd sensitivities), only ttype10_imex, transport_alg=0 and no diagnostics are supported.",
        nb::arg("dt"),
        nb::arg("dtype") = "real",
        nb::arg("compute_diagnostics") = false);

  // Dynamics advance, fwd and adjoint
  m.def("update_dynamics_levels",&update_dynamics_levels,
      "Rotates the dynamics time levels (leapfrog), as done at the end of each step.");
  m.def("enable_adjoint",&enable_adjoint,
      "Turns on fwd state taping and allocates the tape and the adjoint state.\n"
      "Must be called after the model is initialized. Requires time_step_type=10 (ttype10_imex)\n"
      "and a build with HOMMEXX_ENABLE_FAD_TYPES=ON.");
  m.def("prim_advance_exp",&prim_advance_exp,
      "Advances the dynamics by dt, from time level n0 to np1. Does NOT rotate time levels.\n"
      "If the adjoint is enabled, the fwd trajectory is taped (only the LAST call is kept).\n"
      "The scalar type can be chosen via dtype; for dtype!=real, only ttype10_imex is supported,\n"
      "and no diagnostics (the adjoint taping, if on, is skipped for the call).",
        nb::arg("dt"),
        nb::arg("compute_diagnostics") = false,
        nb::arg("dtype") = "real");
  m.def("prim_advance_adj",&prim_advance_adj,
      "Applies the adjoint of the LAST prim_advance_exp call to the adjoint state, in place.\n"
      "On entry the adjoint state holds dJ/dstate(np1); on exit, dJ/dstate(n0).",
        nb::arg("dt"));
  m.def("zero_adj_state",&zero_adj_state);
  m.def("get_adj_state_var",&get_adj_state_var,
      "Copies a field of the adjoint state into arr (same names/shapes as get_state_var).\n"
      "Valid names: u, v, uv, vtheta_dp, dp, w, phi",
        nb::arg("arr"), nb::arg("name"));
  m.def("prim_step_checkpoint_take",&prim_step_checkpoint_take,
      "Captures (Real-typed) dynamics state, qdp and time levels, as the start of the prim_step to be differentiated.\n"
      "Call it right BEFORE the (Real) prim_step that prim_step_adj will differentiate.");
  m.def("prim_step_adj",&prim_step_adj,
      "Applies the adjoint of one prim_step(dt), from the last checkpoint, to the adjoint state and adjoint qdp, in place.\n"
      "On entry they hold dJ/d(state after prim_step), on exit dJ/d(state before prim_step).\n"
      "The live Real state and time levels are restored to the beginning of the prim_step.",
        nb::arg("dt"));
  m.def("zero_adj_qdp",&zero_adj_qdp);
  m.def("get_adj_qdp",&get_adj_qdp,
      "Copies the adjoint qdp into arr, of shape (nelem,qsize,np,np,nlev).",
        nb::arg("arr"));
  m.def("set_adj_qdp",&set_adj_qdp,
      "Sets the adjoint qdp from arr, of shape (nelem,qsize,np,np,nlev).",
        nb::arg("arr"));
  m.def("set_adj_state_var",&set_adj_state_var,
      "Sets a field of the adjoint state from arr (same names/shapes as set_state_var).",
        nb::arg("arr"), nb::arg("name"));
}

} // namespace pyhommexx
