# Pyhommexx: nanobind-based library to allow calling hommexx from python

In order to build this library, you need nanobind and mpi4py, which can
easily be installed via pip. To build the library, add the following
lines to your cmake configuration script:

  -D CMAKE_C_FLAGS:STRING="-fPIC"                           \
  -D CMAKE_CXX_FLAGS:STRING="-fPIC"                         \
  -D CMAKE_Fortran_FLAGS:STRING="-fPIC"                     \
  -D HOMME_BUILD_PYHOMMEXX:BOOL=ON                          \
  -D PYHOMMEXX_NLEV:STRING=128                              \
  -D PYHOMMEXX_QSIZE:STRING=10                              \

where you can change the value of PYHOMMEXX_NLEV and PYHOMMEXX_QSIZE
to match your desided configuration. Notice that the -fPIC flag is
needed if you are not already building ALL homme libraries as shared.
That's because nanobind ONLY builds a shared library, and in order to
link a static lib to a shared lib, the static lib MUST be built with
the -fPIC flag (for compilers other than GNU, this flag may be different).

Once built, in your python code simply do

import sys
sys.append('/path/to/homme/binary/dir/src/theta-l_kokkos/pyhommexx')
import pyhommexx

and you can later access functions from pyhommexx.

## Forward/adjoint dynamics step

`prim_advance_exp(dt)` and `prim_advance_adj(dt)` are available. The adjoint
currently supports only `time_step_type=10` (ttype10_imex), no `prescribed_wind`,
and requires `-D HOMMEXX_ENABLE_FAD_TYPES:BOOL=ON`. Usage:

    pyhommexx.enable_adjoint()            # after model_init: enables taping, allocs adj state
    pyhommexx.prim_advance_exp(dt)        # n0 -> np1 (no time level rotation); taped
    pyhommexx.set_adj_state_var(seed,'dp')  # seed = dJ/dstate(np1), for each field
    pyhommexx.prim_advance_adj(dt)        # in place: adj state now holds dJ/dstate(n0)
    pyhommexx.get_adj_state_var(out,'dp')

Only the LAST `prim_advance_exp` call is taped, so call `prim_advance_adj`
right after the fwd step it differentiates. Use `update_dynamics_levels()` to
rotate time levels between fwd steps.

## Choosing the scalar type: `dtype`

Python has no templates, so functions that can run on different scalar types take a runtime
`dtype` string (`'real'` or `'dpfad'`) and dispatch to the matching instantiation of the templated
C++ implementation (see `dispatch_dtype` in `impl/pyhommexx_utils.hpp`). The scalar type must have
been enabled with `pyhommexx.enable_scalar_type(dtype)` before `model_init()`. `dpfad` needs
`-D HOMMEXX_ENABLE_FAD_TYPES:BOOL=ON`.

pyhommexx cannot be built with `-D HOMMEXX_ENABLE_FWD_SENS:BOOL=ON` (configuration fails with an error):
that mode aliases `ScalarValue` to a Fad type for the Fortran-driven runs, while here we turn on
`HOMMEXX_ENABLE_FAD_TYPES` and select the scalar type of each call at runtime.

The C++ side does not depend on what `ScalarValue` is: `prim_step<ST>`, `prim_advance_exp<ST>` and
`prim_advec_tracers_remap<ST>` grab the `ST`-typed objects from the Context.
For `ST!=ScalarValue` only `time_step_type=10`, Eulerian tracers transport (`transport_alg=0`),
no diagnostics and no adjoint taping are supported.

## Forward sensitivities and adjoint of `prim_step`

    pyhommexx.enable_scalar_type('real'); pyhommexx.enable_scalar_type('dpfad')
    ... read_params, model_init ...
    pyhommexx.enable_adjoint()                 # also sets up the tracers adjoint, if qsize>0 and transport_alg=0
    slot_in, slot_out = pyhommexx.get_prim_step_qdp_slots()   # BEFORE prim_step
    pyhommexx.set_qdp(qdp0, 'real', slot_in)   # qdp: (nelem,qsize,np,np,nlev)
    pyhommexx.prim_step_checkpoint_take()      # start of the prim_step to differentiate

    # Forward: tangent of the Fad run
    pyhommexx.copy_state('real','dpfad')
    pyhommexx.perturb_state_var('u',delta,1.0,'dpfad')   # seeds d/dp of u0*(1+p*delta)
    pyhommexx.set_qdp_dp_sens(dqdp0,slot_in)
    pyhommexx.prim_step(dt,'dpfad')
    pyhommexx.get_state_var_dp_sens(out,'u')   # d(u at np1)/dp
    pyhommexx.get_qdp_dp_sens(out_q,slot_out)

    # Adjoint, in place, from the Real checkpoint
    pyhommexx.set_adj_state_var(seed,'u'); pyhommexx.set_adj_qdp(seed_q)
    pyhommexx.prim_step_adj(dt)
    pyhommexx.get_adj_state_var(out,'u'); pyhommexx.get_adj_qdp(out_q)

`prim_step_adj` works on the Real-typed objects, so `'real'` must be enabled (it is, unless disabled).
It recomputes the forward run from the checkpoint and restores the live Real state and time levels to the
beginning of the `prim_step` on exit. Note that `prim_step('real')` is NOT needed before it.
