#!/bin/bash
#
# DCMIP-2016 test 1 (moist BW) with the C++ theta-l_kokkos dycore,
# ttype10 IMEX, Eulerian transport, qsplit=rsplit=1, native GLL output,
# per-6-dyn-step tendency emission (E10).
#
# Companion to jobscript-dcmip16-hifreq-flight.sh: same target cluster,
# same E10 output schema (FM_x/y/z, FT, FQ1..3), but the *kokkos* dycore
# instead of Fortran, exercising ttype10_imex and the Eulerian
# EulerStepFunctor path.
#
# Target: Sandia Flight cluster.
#   112 physical cores/node (Intel Xeon Silver 4210R, 2 sockets * 56 cores, 8 NUMA),
#   256 GB/node (2.3 GB/core), Cornelis Omni-Path, TOSS 4 / RHEL 8, SLURM.
#
# Usage:
#   sbatch jobscript-eulerian-ttype10-flight.sh smoke    # ne2, 12 dyn steps, ~seconds
#   sbatch jobscript-eulerian-ttype10-flight.sh prod     # ne8, ndays=9,      ~30 min
#
#SBATCH --job-name eulerian-ttype10
#SBATCH --account=fy210162
#SBATCH --reservation=flight-cldera
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Sizing rationale:
#   prod:  ne=8  -> 6 * 8^2 = 384 elements. 1 node * 96 MPI ranks -> 4
#          elements/rank, comfortable. tstep=300 s * ndays=9 = 2592 dyn steps,
#          ttype10 IMEX ~2x cost of ttype5; ~30 min wall expected, 2h loose.
#   smoke: ne=2  -> 24 elements total; NCPU is capped below at 24 regardless
#          of allocation. nmax=12 -> ~seconds.

set -euo pipefail

MODE="${1:-smoke}"

case "$MODE" in
  smoke) NAMELIST=namelist-eulerian-ttype10-smoke.nl ;;
  prod)  NAMELIST=namelist-eulerian-ttype10.nl       ;;
  *) echo "usage: $0 {smoke|prod}"; exit 2 ;;
esac

# nlev=30 kokkos native-output executable — USE_PIO=TRUE routes analysis output
# through prim_movie_mod (native GLL, interp_type=0), which is where the E10
# tendency-write hook (#ifdef HOMME_AMDIS_PROJECT) lives. The plain
# theta-l-nlev30-kokkos target goes through interp_movie_mod, which has no
# tendency dispatch and would silently drop FM_x/FT/FQ* from output_varnames1.
# Build with HOMME_AMDIS_PROJECT=ON and BUILD_HOMME_THETA_KOKKOS=ON.
EXEC=../../../test_execs/theta-l-nlev30-native-kokkos/theta-l-nlev30-native-kokkos
if [ ! -x "$EXEC" ]; then
  echo "ERROR: $EXEC not found."
  echo "  Build the theta-l-nlev30-native-kokkos target with"
  echo "  -DHOMME_AMDIS_PROJECT=ON -DBUILD_HOMME_THETA_KOKKOS=ON."
  exit 1
fi

# Cap ranks by element count (HOMME requires >= 1 element per rank).
#   ne=2 -> 24 elements (smoke).
#   ne=8 -> 384 elements (prod).
NCPU=$(( SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE ))
if [[ "$MODE" == "smoke" && "$NCPU" -gt 24  ]]; then NCPU=24;  fi
if [[ "$MODE" == "prod"  && "$NCPU" -gt 384 ]]; then NCPU=384; fi

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=16M

mkdir -p movies
cp -f "$NAMELIST" input.nl

echo "== eulerian-ttype10: mode=$MODE  namelist=$NAMELIST  NCPU=$NCPU  nodes=$SLURM_JOB_NUM_NODES =="
date
srun --kill-on-bad-exit --cpu-bind=cores -n "$NCPU" "$EXEC" < input.nl
date

echo "== output ($MODE) =="
ls -lh movies/hommexx-eulerian-ttype10-*.nc || true
