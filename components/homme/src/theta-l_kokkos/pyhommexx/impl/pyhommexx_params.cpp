#include "pyhommexx.hpp"

#include "Context.hpp"
#include "ElementsState.hpp"
#include "SimulationParams.hpp"
#include "Types.hpp"

#include <nanobind/ndarray.h>

#include <ekat_assert.hpp>

extern "C"
{
void read_params_f90 (const char*& filename);
} // extern "C"

namespace pyhommexx {

void read_params (const nb::str& nml_filename)
{
  auto nml_filename_c = nml_filename.c_str();
  read_params_f90(nml_filename_c);
}

nb::dict get_params()
{
  using namespace Homme;

  const auto& c = Context::singleton();
  const auto& s = c.get<SimulationParams>();

  nb::dict params;

  params["rsplit"] = s.rsplit;
  params["qsplit"] = s.qsplit;
  params["qsize"] = s.qsize;
  params["qsize_d"] = QSIZE_D;
  params["np"] = NP;
  params["nlev"] = NUM_PHYSICAL_LEV;
  params["hydrostatic"] = s.theta_hydrostatic_mode;
  params["nu"] = s.nu;
  params["ne"] = s.ne;

  return params;
}

void set_params(const nb::dict& params)
{
  using namespace Homme;

  auto& c = Context::singleton();
  auto& s = c.get<SimulationParams>();

  for (const auto& [key,value] : params) {
    EKAT_REQUIRE_MSG (nb::isinstance<nb::str>(key),
        "[pyhommexx::set_params] Functor params dict keys should be strings.\n");

    std::string key_str(nb::cast<nb::str>(key).c_str());
    if (key_str=="alloc_sphere_coords") {
      EKAT_REQUIRE_MSG(nb::isinstance<bool>(value),
          "[pyhommexx::set_params] Functor param 'alloc_sphere_coords' should be a boolean.\n");
      s.alloc_sphere_coords = nb::cast<bool>(value);
    } else if (key_str=="nu_p") {
      // Overrides the value read from the namelist, which the shared init checks require to be >0.
      // This is meant for tests that need a value the checks reject (e.g., the adjoint of prim_step
      // only supports nu_p=0 for now). The functors read nu_p when they are created, so it has to be
      // set after read_params, but before model_init.
      EKAT_REQUIRE_MSG(nb::isinstance<double>(value),
          "[pyhommexx::set_params] Param 'nu_p' should be a float (e.g., 0.0, not 0).\n");
      EKAT_REQUIRE_MSG(not c.has<ElementsState>(),
          "[pyhommexx::set_params] Param 'nu_p' must be set before model_init(): it would be ignored.\n");
      s.nu_p = nb::cast<double>(value);
    } else {
      EKAT_ERROR_MSG("[pyhommexx::set_params] Invalid key for set_params.\n"
          " - key: " + key_str + "\n" +
          " - valid keys: alloc_sphere_coords, nu_p\n");
    }
  }
}

} // namespace pyhommexx
