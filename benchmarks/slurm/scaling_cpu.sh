#!/bin/bash
# ==============================================================================
# scaling_cpu.sh — CPU core scaling and Amdahl fit for period finding
#
# Fetches a real ZTF region once, caches it, then re-runs period finding on
# the same light curves at 1, 2, 4, 8, 16, 32 and 64 cores.  Each core count
# runs in its own subprocess because periodfind's rayon thread pool is sized
# once at load time from RAYON_NUM_THREADS.
#
# Requests 64 cores so the highest core count is not competing with other
# jobs on the same node — contention inflates the slow points and makes the
# fitted serial fraction look worse than it is.
#
# --max-sources 2000 is set below because the single-core point processes the
# whole region serially and otherwise dominates the walltime.  Scaling
# behaviour is unchanged as long as the count stays well above the batch
# size; pass --max-sources 0 to use the full region.
#
# The single-core point processes the entire region serially and dominates
# the job's wall time.  Use --max-sources to cap the workload if the job
# would otherwise exceed the time limit; scaling behaviour is unchanged by
# the absolute source count as long as it stays well above the batch size.
#
# Usage (from ml4em repo root on MSI login node):
#   mkdir -p logs
#   sbatch benchmarks/slurm/scaling_cpu.sh
#
#   # Shorter run:
#   sbatch benchmarks/slurm/scaling_cpu.sh --max-sources 2000 --trials 3
# ==============================================================================
#SBATCH --job-name=ml4em_scaling_cpu
#SBATCH --output=logs/scaling_cpu_%j.out
#SBATCH --error=logs/scaling_cpu_%j.err
#SBATCH -p msismall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -A cough052
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jin00404@umn.edu

ML4EM_GPU=0
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/benchmarks/slurm/_common.sh"

ml4em_run benchmarks/scaling_cpu.py \
    --config "${ML4EM_CONFIG}" \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --core-counts 1 2 4 8 16 32 64 \
    --trials 3 \
    --max-sources 2000 \
    "$@"

# matplotlib ships in both environments via the [plots] extra, so a failure
# here is a real error rather than a missing dependency -- do not mask it.
ml4em_run benchmarks/plot_scaling.py
