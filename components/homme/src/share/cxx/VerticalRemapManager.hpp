/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_VERTICAL_REMAP_MANAGER_HPP
#define HOMMEXX_VERTICAL_REMAP_MANAGER_HPP

#include <memory>

#include "Types.hpp"

namespace Homme {

struct FunctorsBuffersManager;

namespace Remap {
template <typename ST> struct RemapperST;
using Remapper = RemapperST<ScalarValue>;
}

template <typename ST>
struct VerticalRemapManagerST {
  VerticalRemapManagerST(const bool remap_tracers=true);

  VerticalRemapManagerST(const int num_elems, const bool remap_tracers=true);

  void run_remap(int np1, int np1_qdp, double dt) const;

  int requested_buffer_size () const;
  void init_buffers(const FunctorsBuffersManager& fbm);

  std::shared_ptr<Remap::RemapperST<ST>> get_remapper() const;

  bool setup_needed () { return !is_setup; }

  void setup ();

private:
  struct Impl;
  std::shared_ptr<Impl> p_;

  int m_num_elems;
  bool is_setup;
};

using VerticalRemapManager = VerticalRemapManagerST<ScalarValue>;

}

#endif
