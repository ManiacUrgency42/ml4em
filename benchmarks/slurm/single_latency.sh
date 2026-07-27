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
#SBATCH -p a100
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
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

module purge
module load conda
module load cuda/11.8.0

if [[ -f "${DATA_DIR}/.env" ]]; then
    set -a; source "${DATA_DIR}/.env"; set +a
fi

conda run -n ml4em-gpu \
    python "${REPO_DIR}/benchmarks/single_latency.py" \
        --config "${DATA_DIR}/config_msi.yaml" \
        "$@"
