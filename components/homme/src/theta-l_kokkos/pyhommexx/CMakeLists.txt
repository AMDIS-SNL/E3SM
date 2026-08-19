# First, create a theta-l_kokkos target library with the requested compile-time specs
set(PYHOMMEXX_NLEV 128 CACHE STRING "Number of levs used to build the py interface")
set(PYHOMMEXX_QSIZE 10 CACHE STRING "Upper limit to qsize in the tracers array")

THETAL_KOKKOS_SETUP()

include(HommeMacros)

# Remove prim_main
list(REMOVE_ITEM EXEC_SOURCES ${SRC_DIR}/prim_main.F90)

# Create the theta-l_kokkos target library
set(hommeLib theta-l-kokkos)
createExecLib(${hommeLib} theta-l_kokkos "${EXEC_SOURCES}" "${EXEC_INCLUDE_DIRS}"
  4 ${PYHOMMEXX_NLEV} FALSE ${PYHOMMEXX_QSIZE})

# Set homme lib target properties
target_compile_definitions(${hommeLib} PUBLIC
  HOMMEXX_CONFIG_IS_CMAKE
  # SCREAM
  HAVE_CONFIG_H
)
target_link_libraries (${hommeLib} PUBLIC csm_share)
if (HOMME_ENABLE_COMPOSE)
  target_link_libraries (${hommeLib} PUBLIC composec++)
endif()

if(NOT BUILD_HOMME_WITHOUT_PIOLIBRARY)
  if(HOMME_USE_SCORPIO)
    target_link_libraries(${hommeLib} PUBLIC piof pioc)
  else ()
    target_link_libraries(${hommeLib} pio)
  endif ()
endif ()

# Second, ensure we have the python modules
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind REQUIRED)

nanobind_add_module(pyhommexx
  pyhommexx.cpp
  impl/pyhommexx_mod.F90
  impl/pyhommexx_forcing.cpp
  impl/pyhommexx_functors.cpp
  impl/pyhommexx_grid.cpp
  impl/pyhommexx_params.cpp
  impl/pyhommexx_session.cpp
  impl/pyhommexx_state.cpp
  impl/pyhommexx_theta_model.cpp
)

target_link_libraries(pyhommexx PUBLIC ${hommeLib})
target_include_directories(pyhommexx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
