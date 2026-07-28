#!/bin/bash
# ==============================================================================
# single_latency.sh — Single-source latency benchmark on MSI A100
#
# Times each pipeline stage for one ZTF source: Kowalski fetch, statistics,
# period finding, dm/dt, Gaia xmatch.  Useful for CPU vs GPU comparison.
#
# NOT a throughput benchmark — single-source latency does not predict batch
# throughput.  Use batch_throughput.sh for compute hour estimation.
#
# Usage (from ml4em repo root on MSI login node):
#   mkdir -p logs
#   sbatch benchmarks/slurm/single_latency.sh
#   sbatch benchmarks/slurm/single_latency.sh --device cpu   # CPU comparison
#   sbatch benchmarks/slurm/single_latency.sh --device cuda  # GPU timing
# ==============================================================================
#SBATCH --job-name=ml4em_bench_single
#SBATCH --output=logs/bench_single_%j.out
#SBATCH --error=logs/bench_single_%j.err
#SBATCH -p msigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
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
module load cuda/11.8.0

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
    python "${REPO_DIR}/benchmarks/single_latency.py" \
        --config "${DATA_DIR}/config_msi.yaml" \
        "$@"
