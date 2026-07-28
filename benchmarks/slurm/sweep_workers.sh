#!/bin/bash
# ==============================================================================
# sweep_workers.sh — Sweep n_workers to find the Kowalski LC fetch ceiling
#
# Runs the find query (round trip 2) repeatedly with increasing n_workers
# values on the same set of source IDs.  Prints throughput at each value
# so you can find the knee of the curve — the point where adding more workers
# stops improving throughput (Kowalski server or bandwidth becomes bottleneck).
#
# The recommended n_workers from this sweep should be set permanently in
# config_msi.yaml under sources.ztf.n_workers and features.catalog.n_workers.
#
# cpus-per-task is set to 32 to ensure the OS has enough threads available
# for the largest worker count tested.  The actual parallelism is controlled
# by --workers, not by the CPU allocation.
#
# Usage (from ml4em repo root on MSI login node):
#   mkdir -p logs
#   sbatch benchmarks/slurm/sweep_workers.sh
#
#   # Custom worker values:
#   sbatch benchmarks/slurm/sweep_workers.sh --workers 1 2 4 8 16 32
# ==============================================================================
#SBATCH --job-name=ml4em_sweep_workers
#SBATCH --output=logs/sweep_workers_%j.out
#SBATCH --error=logs/sweep_workers_%j.err
#SBATCH -p msismall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -A cough052
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jin00404@umn.edu

DATA_DIR=/scratch.global/$USER/ml4em_data
# sbatch copies the submitted script to /var/spool on the compute node, so
# BASH_SOURCE points there and not at the checkout.  SLURM_SUBMIT_DIR is the
# directory sbatch was called from; the BASH_SOURCE form is the fallback for
# running this script directly with bash.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

module purge
module load conda

if [[ -f "${DATA_DIR}/.env" ]]; then
    set -a; source "${DATA_DIR}/.env"; set +a
fi

# Both are prerequisites, and failing here names the missing one instead of
# surfacing as a pydantic or Kowalski error several minutes in.
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

conda run --no-capture-output -n ml4em-gpu \
    python "${REPO_DIR}/benchmarks/sweep_workers.py" \
        --config "${DATA_DIR}/config_msi.yaml" \
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
        "$@"
