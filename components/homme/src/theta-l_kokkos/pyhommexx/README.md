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
