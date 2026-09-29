SET(SRC_DIR          ${HOMME_SOURCE_DIR}/src)
SET(SRC_SHARE_DIR    ${HOMME_SOURCE_DIR}/src/share)
SET(SRC_THETA_DIR    ${HOMME_SOURCE_DIR}/src/theta-l_kokkos)
SET(SHARE_UT_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/../share_kokkos_ut)
SET(THETA_UT_DIR     ${CMAKE_CURRENT_SOURCE_DIR})
SET(UTILS_TIMING_SRC_DIR ${HOMME_SOURCE_DIR}/utils/cime/CIME/non_py/src/timing)
SET(UTILS_TIMING_BIN_DIR ${HOMME_BINARY_DIR}/utils/cime/CIME/non_py/src/timing)

### Build a 'theta-l_kokkos' library ###
# We do this once, and link against it for all unit tests. This reduces compilation times
THETAL_KOKKOS_SETUP()

# This is needed to compile the lib and test executables with the correct options
#these vars shared between all targets, so changing one var
#for one test only won't work, config is built once and for the last test
SET(THIS_CONFIG_IN ${HOMME_SOURCE_DIR}/src/theta-l_kokkos/config.h.cmake.in)
SET(THIS_CONFIG_HC ${CMAKE_CURRENT_BINARY_DIR}/config.h.c)
SET(THIS_CONFIG_H ${CMAKE_CURRENT_BINARY_DIR}/config.h)
SET (NUM_POINTS 4)
SET (NUM_PLEV 12)
SET (QSIZE_D 4)
SET (PIO_INTERP TRUE)

HommeConfigFile (${THIS_CONFIG_IN} ${THIS_CONFIG_HC} ${THIS_CONFIG_H} )

ADD_LIBRARY(thetal_kokkos_ut_lib
  ${THETAL_DEPS}
  ${TEST_SRC_F90}
  ${SHARE_UT_DIR}/geometry_interface.F90
  ${THETA_UT_DIR}/thetal_test_interface.F90
  ${THETA_UT_DIR}/thetal_ut_session.cpp
  ${SRC_DIR}/checksum_mod.F90
  ${SRC_DIR}/common_io_mod.F90
  ${SRC_DIR}/common_movie_mod.F90
  ${SRC_DIR}/interpolate_driver_mod.F90
  ${SRC_DIR}/interp_movie_mod.F90
  ${SRC_DIR}/netcdf_io_mod.F90
  ${SRC_DIR}/pio_io_mod.F90
  ${SRC_DIR}/prim_movie_mod.F90
  ${SRC_DIR}/theta_restart_mod.F90
  ${SRC_DIR}/restart_io_mod.F90
  ${SRC_DIR}/surfaces_mod.F90
  ${SRC_DIR}/test_mod.F90
)
# Fortran modules
SET(THETA_LIB_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/thetal_kokkos_ut_lib_modules)
SET_TARGET_PROPERTIES(thetal_kokkos_ut_lib PROPERTIES Fortran_MODULE_DIRECTORY ${THETA_LIB_MODULE_DIR})
TARGET_INCLUDE_DIRECTORIES(thetal_kokkos_ut_lib PUBLIC
  ${EXEC_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${SRC_THETA_DIR}
  ${THETA_LIB_MODULE_DIR}
  )
TARGET_COMPILE_DEFINITIONS(thetal_kokkos_ut_lib PUBLIC "HAVE_CONFIG_H")
TARGET_LINK_LIBRARIES(thetal_kokkos_ut_lib ekat::AllLibs)
TARGET_LINK_LIBRARIES(thetal_kokkos_ut_lib timing csm_share ${COMPOSE_LIBRARY_CPP} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
if (HOMMEXX_ENABLE_FAD_TYPES)
  target_link_libraries(thetal_kokkos_ut_lib Sacado::sacado)
endif()

IF (HOMME_USE_MKL)
  IF (MKL_TYPE STREQUAL "oneMKL")
    TARGET_LINK_LIBRARIES(thetal_kokkos_ut_lib -qmkl)
  ELSEIF (MKL_TYPE STREQUAL "Intel MKL")
    TARGET_LINK_LIBRARIES(thetal_kokkos_ut_lib -mkl)
  ENDIF ()
ENDIF()
IF(BUILD_HOMME_WITHOUT_PIOLIBRARY)
  TARGET_COMPILE_DEFINITIONS(thetal_kokkos_ut_lib PUBLIC HOMME_WITHOUT_PIOLIBRARY)
ELSE ()
  IF(HOMME_USE_SCORPIO)
    TARGET_LINK_LIBRARIES(thetal_kokkos_ut_lib piof pioc)
  ELSE ()
    TARGET_LINK_LIBRARIES(thetal_kokkos_ut_lib pio)
  ENDIF ()
ENDIF ()

# Setting this once and for all, good for all unit tests.
SET (CONFIG_DEFINES HAVE_CONFIG_H)

# EkatCreateUnitTest is defined by ekat's own CMake module. Included here,
# unconditionally and before any use, rather than down where the
# state_ut/sacado tests use it (its previous spot): some uses of it (e.g.
# forcing_adjoint_ut, checkpoint_io_ut below) are gated inside
# IF(HOMMEXX_BFB_TESTING), and using a CMake command before its module is
# include()'d is an "Unknown CMake command" configure-time error regardless
# of where in the file that first use is.
include (EkatCreateUnitTest)

# Include dirs common to all the unit tests built with cxx_unit_test
SET (THETAL_UT_INCLUDE_DIRS
  ${SRC_THETA_DIR}/cxx
  ${SRC_SHARE_DIR}
  ${SRC_SHARE_DIR}/cxx
  ${THETA_UT_DIR}
  ${THETA_LIB_MODULE_DIR}
  ${UTILS_TIMING_SRC_DIR}
  ${UTILS_TIMING_BIN_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/src/share/cxx
)

# Thin wrapper around cxx_unit_test, which also links against thetal_kokkos_ut_lib.
# Note: thetal_test_interface.F90 and geometry_interface.F90 are part of the lib,
#       so they must NOT be listed in F90_SRCS.
#  - CXX_SRCS:     the C++ sources of the test
#  - F90_SRCS:     the test-specific F90 sources (if any)
#  - INCLUDE_DIRS: additional include dirs (on top of THETAL_UT_INCLUDE_DIRS)
#  - NUM_CPUS:     number of MPI ranks (defaults to USE_NUM_PROCS, or 1 if not set)
FUNCTION (thetal_cxx_unit_test test_name)
  CMAKE_PARSE_ARGUMENTS (TUT "" "NUM_CPUS" "CXX_SRCS;F90_SRCS;INCLUDE_DIRS" ${ARGN})
  IF (NOT TUT_NUM_CPUS)
    IF (USE_NUM_PROCS)
      SET (TUT_NUM_CPUS ${USE_NUM_PROCS})
    ELSE ()
      SET (TUT_NUM_CPUS 1)
    ENDIF ()
  ENDIF ()
  cxx_unit_test (${test_name} "${TUT_F90_SRCS}" "${TUT_CXX_SRCS}"
                 "${THETAL_UT_INCLUDE_DIRS};${TUT_INCLUDE_DIRS}" "${CONFIG_DEFINES}" ${TUT_NUM_CPUS})
  TARGET_LINK_LIBRARIES (${test_name} thetal_kokkos_ut_lib)
ENDFUNCTION ()

IF (HOMMEXX_BFB_TESTING)
  ### Equation of state unit tests

  thetal_cxx_unit_test (eos_ut
    CXX_SRCS ${THETA_UT_DIR}/eos_ut.cpp
    F90_SRCS ${THETA_UT_DIR}/eos_interface.F90)

  ### Element ops unit tests

  thetal_cxx_unit_test (elem_ops_ut
    CXX_SRCS ${THETA_UT_DIR}/elem_ops_ut.cpp
    F90_SRCS ${THETA_UT_DIR}/elem_ops_interface.F90)

  # ### HyperViscosity unit tests

  thetal_cxx_unit_test (hv_ut
    CXX_SRCS ${THETA_UT_DIR}/hv_ut.cpp
    F90_SRCS ${THETA_UT_DIR}/hv_interface.F90)

  ### Forcing unit tests

  # Unfortunately, due to F90 dependencies, we end up including most of the f90
  # srcs, and one of them happens to include config.h explicitly
  # We already set all the config options by hand, so we can simply touch an empty config.h
  EXECUTE_PROCESS (COMMAND touch config.h WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  thetal_cxx_unit_test (forcing_ut
    CXX_SRCS ${THETA_UT_DIR}/forcing_ut.cpp
    F90_SRCS ${THETA_UT_DIR}/forcing_interface.F90)

  ### Forcing adjoint unit test (dJ/dF, states_forcing_adj) ###

  if (NOT HOMMEXX_ENABLE_FWD_SENS)
    # states_forcing_adj is Real-only (see the static_assert in
    # ForcingFunctor.hpp), so it isn't usable when ForcingFunctor's
    # ScalarValue is a Fad type. Purely local (no MPI/connectivity), so
    # unlike most other thetal_kokkos_ut tests it needs no F90 sources.
    EkatCreateUnitTest ("forcing_adjoint_ut"
      "forcing_adjoint_ut.cpp"
      LIBS thetal_kokkos_ut_lib
      LABELS AMDIS
    )
  endif()

  ### Checkpoint I/O unit test (on-disk StateSnapshot round trip, battleplan Step 3) ###

  if (NOT BUILD_HOMME_WITHOUT_PIOLIBRARY AND HOMME_USE_SCORPIO)
    # CheckpointIO.{hpp,cpp} needs a PIO/scorpio library to link against
    # (thetal_kokkos_ut_lib already links piof/pioc in this configuration --
    # see above). Its I/O (PIOc_write_darray/read_darray, through a real
    # decomposition) is collective over its communicator and all ranks
    # share one file per NN call -- this just happens to run single-rank
    # here, so the collectives degenerate to the trivial case.
    EkatCreateUnitTest ("checkpoint_io_ut"
      "checkpoint_io_ut.cpp"
      LIBS thetal_kokkos_ut_lib
      LABELS AMDIS
    )
  endif()

  ### Euler step (tracer advection limiter) adjoint unit test (battleplan Step 4) ###

  # euler_step_adjoint_ut.cpp builds its own explicit TracersST<Real>/
  # ElementsDerivedStateST<Real>/EulerStepFunctorST<Real> (and, under
  # HOMMEXX_ENABLE_FAD_TYPES, TracersST<DpFadType>/
  # ElementsDerivedStateST<DpFadType>/EulerStepFunctorST<DpFadType> too, for
  # its dot-product TEST_CASE), independent of whatever ScalarValue the
  # HOMMEXX_ENABLE_FWD_SENS build setting would otherwise alias
  # Tracers/ElementsDerivedState/EulerStepFunctor to -- see the file's own
  # header comment. So, unlike before, this test needs no
  # HOMMEXX_ENABLE_FWD_SENS gate at all: it builds and passes either way.
  # Needs the F90 mesh/connectivity init (unlike forcing_adjoint_ut):
  # euler_step always runs a regular DSS boundary exchange, even when the
  # neighbor min-max exchange itself isn't exercised.
  set (EULER_STEP_ADJOINT_SOURCES
    euler_step_adjoint_ut.cpp
  )
  EkatCreateUnitTest ("euler_step_adjoint_ut"
    "${EULER_STEP_ADJOINT_SOURCES}"
    LIBS thetal_kokkos_ut_lib
    LABELS AMDIS
  )

  # ### Caar functor unit test

  thetal_cxx_unit_test (caar_ut
    CXX_SRCS ${THETA_UT_DIR}/caar_ut.cpp
    F90_SRCS ${THETA_UT_DIR}/caar_interface.F90)

  # ### Remap functor unit test

  thetal_cxx_unit_test (remap_theta_ut
    CXX_SRCS ${THETA_UT_DIR}/remap_theta_ut.cpp
    F90_SRCS ${THETA_UT_DIR}/remap_theta_interface.F90)
ENDIF ()

# ### DIRK functor unit test

thetal_cxx_unit_test (dirk_ut
  CXX_SRCS ${THETA_UT_DIR}/dirk_ut.cpp
  F90_SRCS ${THETA_UT_DIR}/dirk_interface.F90)

if (NOT HOMMEXX_ENABLE_FWD_SENS)
  # ### Compose semi-Lagrangian transport unit tests
  # NOTE: we don't use it for now in the AMDIS effort, so hold off conversion to Fad types

  thetal_cxx_unit_test (compose_ut
    CXX_SRCS ${THETA_UT_DIR}/compose_ut.cpp
    F90_SRCS ${THETA_UT_DIR}/compose_interface.F90
    INCLUDE_DIRS ${SRC_SHARE_DIR}/compose
    NUM_CPUS 1)

  # ### GllFvRemap unit tests
  # NOTE: we don't use it in hommexx standalone, so hold off conversion to Fad types
  thetal_cxx_unit_test (gllfvremap_ut
    CXX_SRCS ${THETA_UT_DIR}/gllfvremap_ut.cpp
    F90_SRCS ${THETA_UT_DIR}/gllfvremap_interface.F90 ${THETA_UT_DIR}/compose_interface.F90)
  cxx_unit_test_add_test(gllfvremap_planar_ut gllfvremap_ut ${NUM_CPUS} "hommexx -planar")
endif()

# EkatCreateUnitTest is now included earlier in this file (see above, near
# forcing_adjoint_ut/checkpoint_io_ut).
# State generic tests
EkatCreateUnitTest (state_ut
  state_ut.cpp
  LIBS thetal_kokkos_ut_lib
  LABELS AMDIS
)

if (HOMMEXX_ENABLE_FAD_TYPES)

  # Caar sacado derivatives tests
  EkatCreateUnitTest (caar_sacado_ut
    caar_sacado_ut.cpp
    COMPILER_CXX_DEFS HOMMEXX_SPHERE_OP_ST_FACTORS
    LIBS thetal_kokkos_ut_lib
    LABELS AMDIS
  )

  # Eqn of state sacado derivatives tests
  EkatCreateUnitTest (eos_sacado_ut
    eos_sacado_ut.cpp
    LIBS thetal_kokkos_ut_lib
    LABELS AMDIS
  )

  # Dirk sacado derivatives tests
  EkatCreateUnitTest ("dirk_sacado_ut"
    dirk_sacado_ut.cpp
    LIBS thetal_kokkos_ut_lib
    LABELS AMDIS
  )

  # Hyperviscosity sacado derivatives tests
  EkatCreateUnitTest ("hv_sacado_ut"
    "hv_sacado_ut.cpp;hv_interface.F90"
    LIBS thetal_kokkos_ut_lib
    LABELS AMDIS
  )

  # Tridiag sacado derivatives tests
  EkatCreateUnitTest ("tridiag_sacado_ut"
    tridiag_sacado_ut.cpp
    LIBS thetal_kokkos_ut_lib
    LABELS AMDIS
  )

  # Test the gradient calculation via adjoints
  set (IMEX_ADJOINT_SOURCES
    imex_adjoint_ut.cpp
    ${SRC_THETA_DIR}/cxx/prim_advance_adj.cpp
  )
  EkatCreateUnitTest ("imex_adjoint_ut"
    "${IMEX_ADJOINT_SOURCES}"
    LIBS thetal_kokkos_ut_lib
    LABELS AMDIS
  )

  # Test the full prim_advance_exp adjoint (ttype10 CAAR/DIRK stages + HV +
  # the w_i(n0) surface fix), via prim_advance_adj
  set (PRIM_ADVANCE_SACADO_SOURCES
    prim_advance_sacado_ut.cpp
    ${SRC_THETA_DIR}/cxx/prim_advance_adj.cpp
  )
  EkatCreateUnitTest ("prim_advance_sacado_ut"
    "${PRIM_ADVANCE_SACADO_SOURCES}"
    LIBS thetal_kokkos_ut_lib
    LABELS AMDIS
  )

  if (NOT HOMMEXX_ENABLE_FWD_SENS)
    # Test the adjoint of prim_step (Eulerian tracers): prim_advance_adj (K times),
    # the 3 euler_step_adj, and the adjoints of the glue in between. The adjoint
    # recomputes the fwd trajectory with the Real-typed objects in the Context (the
    # ones a run would use), so ScalarValue must be Real (like for the
    # Real-only euler_step_adj/states_forcing_adj themselves).
    # Note: prim_step_adj.cpp is only compiled into this test for now (it has no
    # other users yet), unlike prim_advance_adj.cpp, which is in the lib.
    set (PRIM_STEP_ADJOINT_SOURCES
      prim_step_adjoint_ut.cpp
      thetal_test_interface.F90
      ${SHARE_UT_DIR}/geometry_interface.F90
      ${SRC_THETA_DIR}/cxx/prim_step_adj.cpp
    )
    EkatCreateUnitTest ("prim_step_adjoint_ut"
      "${PRIM_STEP_ADJOINT_SOURCES}"
      LIBS thetal_kokkos_ut_lib
      LABELS AMDIS
    )
  endif()
endif()
