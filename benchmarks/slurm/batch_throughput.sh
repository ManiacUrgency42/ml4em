#!/bin/bash
# ==============================================================================
# batch_throughput.sh — End-to-end batch throughput benchmark on MSI A100
#
# Runs the full production pipeline on a real ZTF sky region and reports
# per-stage timing.  This is the primary benchmark for estimating compute
# hours before production runs.
#
# Stages timed:
#   Round trip 1  near query    Kowalski spatial index → source IDs
#   Round trip 2  find query    fetch_batch() → full light curves
#                 Statistics    periodfind BasicStats
#                 Period        CE/AOV/LS/MHF (GPU batched)
#                 dm/dt         periodfind DmDt histogram
#   Round trip 3  Gaia xmatch  CatalogExtractor → Gaia EDR3 features
#
# Run this FIRST before any sweep benchmarks to get a baseline.
# Then run sweep_workers.sh and sweep_batch_size.sh to tune config.
#
# Usage (from ml4em repo root on MSI login node):
#   mkdir -p logs
#   sbatch benchmarks/slurm/batch_throughput.sh
#
#   # Override workers or device without editing config:
#   sbatch benchmarks/slurm/batch_throughput.sh --n-workers 8
#   sbatch benchmarks/slurm/batch_throughput.sh --n-workers 8 --device cuda
# ==============================================================================
#SBATCH --job-name=ml4em_bench_batch
#SBATCH --output=logs/bench_batch_%j.out
#SBATCH --error=logs/bench_batch_%j.err
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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
    python "${REPO_DIR}/benchmarks/batch_throughput.py" \
        --config "${DATA_DIR}/config_msi.yaml" \
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
        --device cuda \
        --warmup \
        "$@"
