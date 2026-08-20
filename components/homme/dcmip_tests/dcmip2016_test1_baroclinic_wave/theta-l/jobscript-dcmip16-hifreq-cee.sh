#!/bin/bash
#
# DCMIP-2016 test 1 (moist BW) — high-frequency training-data run for pyhommexx D8.
# Emits prognostic state + ground-truth physics tendencies (FM/FT/FQ) every
# physics step on the native GLL grid.
#
# Usage:
#   sbatch jobscript-dcmip16-hifreq-cee.sh smoke   # ne2, 2 physics steps, ~seconds
#   sbatch jobscript-dcmip16-hifreq-cee.sh prod    # ne30, ndays=5, hours
#
#SBATCH --job-name dcmip16-hifreq
#SBATCH --account=condo
#SBATCH -p acme-medium
#SBATCH -N 4
#SBATCH --time=24:00:00
#
# Time is sized for prod (ne30/ndays=60 — a couple of days at 4 nodes, comfortable
# 24h wallclock with margin). Smoke run finishes in seconds — override with
# `sbatch --time=00:05:00 --job-name dcmip16-smoke jobscript-dcmip16-hifreq-cee.sh smoke`
# if you don't want to sit in the medium queue for a trivial job.

set -euo pipefail

MODE="${1:-smoke}"

case "$MODE" in
  smoke) NAMELIST=namelist-dcmip16-ne2-hifreq.nl  ;;
  prod)  NAMELIST=namelist-dcmip16-ne30-hifreq.nl ;;
  *) echo "usage: $0 {smoke|prod}"; exit 2 ;;
esac

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=16M
export MV2_ENABLE_AFFINITY=0

NCPU=8
if [ -n "${SLURM_NNODES:-}" ]; then
  patt='([[:digit:]]+)'
  if [[ ${SLURM_TASKS_PER_NODE:-} =~ $patt ]]; then
    PER_NODE=${BASH_REMATCH[1]}
  else
    PER_NODE=16
  fi
  NCPU=$(( SLURM_NNODES * PER_NODE ))
fi

# nlev=30 native-output executable. `-native` = USE_PIO=TRUE = prim_movie_mod,
# which supports interp_type=0 (native GLL output) — required by our namelist.
# The plain `theta-l-nlev30` target uses interp_movie_mod (interpolated-only) and
# would silently ignore interp_type=0.
# Build with HOMME_AMDIS_PROJECT=ON so the FM/FT/FQ output_varnames1 dispatch is
# compiled in (E10). The `-native` target's QSIZE_D was bumped 3 -> 6 to match
# DCMIP-2016 test 1's qsize=6 (Kessler + toy chemistry indices consumed unconditionally
# by dcmip16_wrapper.F90:494-497).
EXEC=../../../test_execs/theta-l-nlev30-native/theta-l-nlev30-native
if [ ! -x "$EXEC" ]; then
  echo "ERROR: $EXEC not found. Build theta-l-nlev30-native with HOMME_AMDIS_PROJECT=ON."
  exit 1
fi

mkdir -p movies
cp -f "$NAMELIST" input.nl

echo "== dcmip16 hifreq: mode=$MODE  namelist=$NAMELIST  NCPU=$NCPU =="
date
srun -K -c 1 -n "$NCPU" -N "${SLURM_NNODES:-1}" "$EXEC" < input.nl
date

echo "== output ($MODE) =="
ls -lh movies/dcmip16-t1-*hifreq*.nc || true
