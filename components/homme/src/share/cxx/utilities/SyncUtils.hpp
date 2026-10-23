/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_SYNC_UTILS_HPP
#define HOMMEXX_SYNC_UTILS_HPP

#include "Types.hpp"

namespace Homme {

// ===================== SYNC FROM DEVICE TO HOST ============================ //

template<typename ST, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<ST*[NP][NP],SProps...>& src,
                   const HostView<Real*[NP][NP],DProps...>& dst)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        dst(ie, igp, jgp) = ADValue(src_h(ie, igp, jgp));
      }
    }
  }
}

template<typename ST, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<ST*[NUM_TIME_LEVELS][NP][NP],SProps...>& src,
                   const HostView<Real*[NUM_TIME_LEVELS][NP][NP],DProps...>& dst)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dst(ie, tl, igp, jgp) = ADValue(src_h(ie, tl, igp, jgp));
        }
      }
    }
  }
}

template<typename ST, int NLEVS, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>*[NUM_TIME_LEVELS][NP][NP][ColInfo<NLEVS>::NumPacks],SProps...>& src,
                   const HostView<Real*[NUM_TIME_LEVELS][NLEVS][NP][NP],DProps...>& dst)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int level = 0; level < NLEVS; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dst(ie, tl, level, igp, jgp) = ADValue(src_h(ie, tl, igp, jgp, ilev)[ivec]);
          }
        }
      }
    }
  }
}

template<typename ST, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>*[NUM_TIME_LEVELS][2][NP][NP][NUM_LEV],SProps...>& src,
                   const HostView<Real*[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP],DProps...>& dst)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dst(ie, tl, level, 0, igp, jgp) = ADValue(src_h(ie, tl, 0, igp, jgp, ilev)[ivec]);
            dst(ie, tl, level, 1, igp, jgp) = ADValue(src_h(ie, tl, 1, igp, jgp, ilev)[ivec]);
          }
        }
      }
    }
  }
}

template<typename ST, int DIM, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>*[DIM][NP][NP][NUM_LEV],SProps...>& src,
                   const HostView<Real*[NUM_PHYSICAL_LEV][DIM][NP][NP],DProps...>& dst)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int idim = 0; idim<DIM; ++idim) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dst(ie, level, idim, igp, jgp) = ADValue(src_h(ie, idim, igp, jgp, ilev)[ivec]);
          }
        }
      }
    }
  }
}

template<typename ST, int NLEVS, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>[NP][NP][ColInfo<NLEVS>::NumPacks],SProps...>& src,
                   const HostView<Real[NLEVS][NP][NP],DProps...>& dst)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int level = 0; level < NLEVS; ++level) {
    const int ilev = level / VECTOR_SIZE;
    const int ivec = level % VECTOR_SIZE;
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        dst(level, igp, jgp) = ADValue(src_h(igp, jgp, ilev)[ivec]);
      }
    }
  }
}

template<typename ST, int NLEVS, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>*[NP][NP][ColInfo<NLEVS>::NumPacks],SProps...>& src,
                   const HostView<Real*[NLEVS][NP][NP],DProps...>& dst)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int level = 0; level < NLEVS; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dst(ie, level, igp, jgp) = ADValue(src_h(ie, igp, jgp, ilev)[ivec]);
        }
      }
    }
  }
}

template<typename ST, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>**[NP][NP][NUM_LEV],SProps...>& src,
                   const HostView<Real**[NUM_PHYSICAL_LEV][NP][NP],DProps...>& dst)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int idim = 0; idim < src.extent_int(1); ++idim) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dst(ie, idim, level, igp, jgp) = ADValue(src_h(ie, idim, igp, jgp, ilev)[ivec]);
          }
        }
      }
    }
  }
}

template<typename ST, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV],SProps...>& src,
                   const HostView<Real*[Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP],DProps...>& dst)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int tl = 0; tl < Q_NUM_TIME_LEVELS; ++tl) {
      for (int iq = 0; iq < QSIZE_D; ++iq) {
        for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
          const int ilev = level / VECTOR_SIZE;
          const int ivec = level % VECTOR_SIZE;
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dst(ie, tl, iq, level, igp, jgp) = ADValue(src_h(ie, tl, iq, igp, jgp, ilev)[ivec]);
            }
          }
        }
      }
    }
  }
}

// ============ SYNC A SINGLE TIME LEVEL FROM DEVICE TO HOST ================= //
//
// These are trimmed twins of the sync_to_host overloads above, for the case where
// only one time level of a state array is actually consumed on the Fortran side
// (which is all CAM/EAM and SCREAM ever read after a step). They copy source time
// level src_tl into destination time level dst_tl and leave the other destination
// slots untouched, so the Fortran array keeps its full NUM_TIME_LEVELS extent and
// its layout/strides are unchanged.
//
// src_tl and dst_tl are separate because they are not always the same slot: the
// qdp levels are rotated by TimeLevel::update_tracers_levels() before the vertical
// remap, while the dynamics levels are rotated afterwards, so the freshly-remapped
// tracer mass lives in np1_qdp while Fortran readers index with TimeLevel_Qdp().
// Pass the same value twice when they do coincide.
//
// NOTE: the create_mirror_view/deep_copy pair below is deliberately left full
// width. On a host build it is a no-op alias plus a self-copy that Kokkos
// short-circuits, so the entire cost is the repack loop that we are trimming here.
// Narrowing the transfer itself would require packing one time level on the device
// first, since a Kokkos::subview slice is LayoutStride and cross-space deep_copy
// does not accept that.

// Single time level of m_dp3d / m_vtheta_dp (NLEVS=NUM_PHYSICAL_LEV)
// or m_w_i / m_phinh_i (NLEVS=NUM_INTERFACE_LEV)
template<typename ST, int NLEVS, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>*[NUM_TIME_LEVELS][NP][NP][ColInfo<NLEVS>::NumPacks],SProps...>& src,
                   const HostView<Real*[NUM_TIME_LEVELS][NLEVS][NP][NP],DProps...>& dst,
                   const int src_tl, const int dst_tl)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int level = 0; level < NLEVS; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dst(ie, dst_tl, level, igp, jgp) = ADValue(src_h(ie, src_tl, igp, jgp, ilev)[ivec]);
        }
      }
    }
  }
}

// Single time level of m_v (2 horizontal components)
template<typename ST, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>*[NUM_TIME_LEVELS][2][NP][NP][NUM_LEV],SProps...>& src,
                   const HostView<Real*[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP],DProps...>& dst,
                   const int src_tl, const int dst_tl)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dst(ie, dst_tl, level, 0, igp, jgp) = ADValue(src_h(ie, src_tl, 0, igp, jgp, ilev)[ivec]);
          dst(ie, dst_tl, level, 1, igp, jgp) = ADValue(src_h(ie, src_tl, 1, igp, jgp, ilev)[ivec]);
        }
      }
    }
  }
}

// Single qdp time level (all tracers)
template<typename ST, typename... SProps, typename... DProps>
void sync_to_host (const ExecView<PackType<ST>*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV],SProps...>& src,
                   const HostView<Real*[Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP],DProps...>& dst,
                   const int src_tl, const int dst_tl)
{
  auto src_h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(src_h, src);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int iq = 0; iq < QSIZE_D; ++iq) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dst(ie, dst_tl, iq, level, igp, jgp) = ADValue(src_h(ie, src_tl, iq, igp, jgp, ilev)[ivec]);
          }
        }
      }
    }
  }
}

// ===================== SYNC FROM HOST TO DEVICE ============================ //

template <typename ST, typename... SProps, typename... DProps>
void sync_to_device (const HostView<const Real**[NUM_PHYSICAL_LEV][NP][NP],SProps...>& src,
                     const ExecView<PackType<ST>**[NP][NP][NUM_LEV],DProps...>& dst)
{
  int dim1 = std::min(src.extent(1),dst.extent(1));
  auto dst_h = Kokkos::create_mirror_view(dst);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int idim = 0; idim < dim1; ++idim) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dst_h(ie, idim, igp, jgp, ilev)[ivec] = src(ie, idim, level, igp, jgp);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dst,dst_h);
}

template <typename ST, typename... SProps, typename... DProps>
void sync_to_device (const HostView<const Real*[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP],SProps...>& src,
                     const ExecView<PackType<ST>*[NUM_TIME_LEVELS][2][NP][NP][NUM_LEV],DProps...>& dst)
{
  auto dst_h = Kokkos::create_mirror_view(dst);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int tl=0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dst_h(ie, tl, 0, igp, jgp, ilev)[ivec] = src(ie, tl, level, 0, igp, jgp);
            dst_h(ie, tl, 1, igp, jgp, ilev)[ivec] = src(ie, tl, level, 1, igp, jgp);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dst,dst_h);
}

template <typename ST, int NLEVS, typename... SProps, typename... DProps>
void sync_to_device (const HostView<const Real*[NUM_TIME_LEVELS][NLEVS][NP][NP],SProps...>& src,
                     const ExecView<PackType<ST>*[NUM_TIME_LEVELS][NP][NP][ColInfo<NLEVS>::NumPacks],DProps...>& dst)
{
  auto dst_h = Kokkos::create_mirror_view(dst);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int tl=0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int level = 0; level < NLEVS; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dst_h(ie, tl, igp, jgp, ilev)[ivec] = src(ie, tl, level, igp, jgp);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dst,dst_h);
}


template<typename ST, int NLEVS, typename... SProps, typename... DProps>
void sync_to_device (const HostView<const Real*[NLEVS][NP][NP],SProps...>& src,
                     const ExecView<PackType<ST>*[NP][NP][ColInfo<NLEVS>::NumPacks],DProps...>& dst)
{
  auto dst_h = Kokkos::create_mirror_view(dst);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int level = 0; level < NLEVS; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dst_h(ie, igp, jgp, ilev)[ivec] = src(ie, level, igp, jgp);
        }
      }
    }
  }
  Kokkos::deep_copy(dst,dst_h);
}

template<typename ST, int DIM, int NLEVS, typename... SProps, typename... DProps>
void sync_to_device (const HostView<const Real*[NLEVS][DIM][NP][NP],SProps...>& src,
                     const ExecView<PackType<ST>*[DIM][NP][NP][ColInfo<NLEVS>::NumPacks],DProps...>& dst)
{
  auto dst_h = Kokkos::create_mirror_view(dst);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int idim=0; idim < DIM; ++idim) {
      for (int level = 0; level < NLEVS; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dst_h(ie, idim, igp, jgp, ilev)[ivec] = src(ie, idim, level, igp, jgp);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dst,dst_h);
}

template<typename ST, typename... SProps, typename... DProps>
void sync_to_device (const HostView<const Real[2][NP][NP],SProps...>& src,
                     const ExecView<ST[2][NP][NP],DProps...>& dst)
{
  auto dst_h = Kokkos::create_mirror_view(dst);
  for (int igp = 0; igp < NP; ++igp) {
    for (int jgp = 0; jgp < NP; ++jgp) {
      dst_h(0, igp, jgp) = src(0, igp, jgp);
      dst_h(1, igp, jgp) = src(1, igp, jgp);
    }
  }
  Kokkos::deep_copy(dst,dst_h);
}

template<typename ST, typename... SProps, typename... DProps>
void sync_to_device (const HostView<const Real*[Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP],SProps...>& src,
                     const ExecView<PackType<ST>*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV],DProps...>& dst)
{
  auto dst_h = Kokkos::create_mirror_view(dst);
  for (int ie = 0; ie < src.extent_int(0); ++ie) {
    for (int tl = 0; tl < Q_NUM_TIME_LEVELS; ++tl) {
      for (int iq = 0; iq < QSIZE_D; ++iq) {
        for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
          const int ilev = level / VECTOR_SIZE;
          const int ivec = level % VECTOR_SIZE;
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dst_h(ie, tl, iq, igp, jgp, ilev)[ivec] = src(ie, tl, iq, level, igp, jgp);
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dst,dst_h);
}

} // namespace Homme

#endif // HOMMEXX_SYNC_UTILS_HPP
