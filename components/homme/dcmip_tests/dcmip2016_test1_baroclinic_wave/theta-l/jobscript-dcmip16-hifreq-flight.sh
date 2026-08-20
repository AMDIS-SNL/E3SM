#!/bin/bash
#
# DCMIP-2016 test 1 (moist BW) — high-frequency training-data run for pyhommexx D8.
# Emits prognostic state + ground-truth physics tendencies (FM/FT/FQ) every
# physics step on the native GLL grid.
#
# Target: Sandia Flight cluster.
#   112 physical cores/node (Intel Xeon Silver 4210R, 2 sockets * 56 cores, 8 NUMA),
#   256 GB/node (2.3 GB/core), Cornelis Omni-Path, TOSS 4 / RHEL 8, SLURM.
#
# Usage:
#   sbatch jobscript-dcmip16-hifreq-flight.sh smoke   # ne2, 2 physics steps, ~seconds
#   sbatch jobscript-dcmip16-hifreq-flight.sh prod    # ne30, ndays=60, hours
#
#SBATCH --job-name dcmip16-hifreq
#SBATCH --account=fy210162
#SBATCH --reservation=flight-cldera
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Sizing rationale (prod):
#   ne=30 gives 6 * 30^2 = 5400 elements. 4 nodes * 112 = 448 MPI ranks
#   -> ~12 elements/rank, comfortable. Per-physics-step NetCDF output is
#   gated to 16 IO ranks (num_io_procs=16 in the namelist), so I/O contention
#   is bounded regardless of node count. 8h wall is loose for ndays=60 at ne30.

set -euo pipefail

MODE="${1:-smoke}"

case "$MODE" in
  smoke) NAMELIST=namelist-dcmip16-ne2-hifreq.nl  ;;
  prod)  NAMELIST=namelist-dcmip16-ne30-hifreq.nl ;;
  *) echo "usage: $0 {smoke|prod}"; exit 2 ;;
esac

# nlev=30 native-output executable. `-native` = USE_PIO=TRUE = prim_movie_mod,
# which supports interp_type=0 (native GLL output) — required by our namelist.
# The plain `theta-l-nlev30` target uses interp_movie_mod (interpolated-only) and
# would silently ignore interp_type=0.
# Build with HOMME_AMDIS_PROJECT=ON so the FM/FT/FQ output_varnames1 dispatch is
# compiled in (E10). The `-native` target's QSIZE_D was bumped 3 -> 6 to match
# DCMIP-2016 test 1's qsize=6 (Kessler + toy chemistry indices consumed
# unconditionally by dcmip16_wrapper.F90:494-497).
EXEC=../../../test_execs/theta-l-nlev30-native/theta-l-nlev30-native
if [ ! -x "$EXEC" ]; then
  echo "ERROR: $EXEC not found. Build theta-l-nlev30-native with HOMME_AMDIS_PROJECT=ON."
  exit 1
fi

# ne=2 has only 24 elements total (6 faces * 4 elems/face). HOMME needs
# >= 1 element per rank, so cap smoke at 24 ranks regardless of allocation.
NCPU=$(( SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE ))
if [[ "$MODE" == "smoke" && "$NCPU" -gt 24 ]]; then
  NCPU=24
fi

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=16M

mkdir -p movies
cp -f "$NAMELIST" input.nl

echo "== dcmip16 hifreq: mode=$MODE  namelist=$NAMELIST  NCPU=$NCPU  nodes=$SLURM_JOB_NUM_NODES =="
date
srun --kill-on-bad-exit --cpu-bind=cores -n "$NCPU" "$EXEC" < input.nl
date

echo "== output ($MODE) =="
ls -lh movies/dcmip16-t1-*hifreq*.nc || true
