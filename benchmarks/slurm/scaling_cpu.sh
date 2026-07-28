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

SIF=/scratch.global/$USER/ml4em_gpu.sif
DATA_DIR=/scratch.global/$USER/ml4em_data
# sbatch copies the submitted script to /var/spool on the compute node, so
# BASH_SOURCE points there and not at the checkout.  SLURM_SUBMIT_DIR is the
# directory sbatch was called from; the BASH_SOURCE form is the fallback for
# running this script directly with bash.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

module purge
module load apptainer

if [[ -f "${DATA_DIR}/.env" ]]; then
    set -a; source "${DATA_DIR}/.env"; set +a
fi

# Both are prerequisites, and failing here names the missing one instead of
# surfacing as a pydantic or Kowalski error several minutes in.
if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: ${SIF} not found." >&2
    echo "       Run: sbatch slurm/pull_image.sh" >&2
    exit 1
fi
if [[ ! -f "${DATA_DIR}/config_msi.yaml" ]]; then
    echo "ERROR: ${DATA_DIR}/config_msi.yaml not found." >&2
    echo "       cp config.example.yaml ${DATA_DIR}/config_msi.yaml and edit storage.*" >&2
    exit 1
fi
if [[ -z "${ML4EM_ZTF_TOKEN:-}" ]]; then
    echo "ERROR: ML4EM_ZTF_TOKEN is not set (expected in ${DATA_DIR}/.env)." >&2
    echo "       Run: python scripts/get_credentials.py" >&2
    exit 1
fi

cd "${REPO_DIR}"

apptainer run \
    --bind "${REPO_DIR}:/app/ml4em" \
    --bind "${DATA_DIR}:/data" \
    --env-file "${DATA_DIR}/.env" \
    --env PYTHONPATH=/data/pyshim \
    "${SIF}" \
    python benchmarks/scaling_cpu.py \
        --config /data/config_msi.yaml \
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
        --core-counts 1 2 4 8 16 32 64 \
        --trials 3 \
        --max-sources 2000 \
        "$@"

# matplotlib ships in the image via the [plots] extra, so a failure here
# is a real error rather than a missing dependency -- do not mask it.
apptainer run \
    --bind "${REPO_DIR}:/app/ml4em" \
    --bind "${DATA_DIR}:/data" \
    --env-file "${DATA_DIR}/.env" \
    --env PYTHONPATH=/data/pyshim \
    "${SIF}" \
    python benchmarks/plot_scaling.py
