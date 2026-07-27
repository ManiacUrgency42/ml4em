#!/bin/bash
# ==============================================================================
# sweep_batch_size.sh — Sweep feature_batch_size to find the GPU ceiling
#
# Fetches a real ZTF region once, then runs period finding repeatedly with
# increasing feature_batch_size values on the same light curves.
#
# Larger batches improve GPU lane utilization but consume more VRAM.
# The sweep stops automatically at OOM.  The last successful batch size
# before OOM (or plateau) is your recommended feature_batch_size.
#
# A100 nodes have 40GB VRAM.  Expect OOM somewhere between 2000–8000
# depending on the n_obs distribution of the fetched region.
#
# The recommended feature_batch_size should be set in config_msi.yaml
# under features.feature_batch_size.
#
# Usage (from ml4em repo root on MSI login node):
#   mkdir -p logs
#   sbatch benchmarks/slurm/sweep_batch_size.sh
#
#   # Custom batch sizes:
#   sbatch benchmarks/slurm/sweep_batch_size.sh --batch-sizes 500 1000 2000 4000 8000
# ==============================================================================
#SBATCH --job-name=ml4em_sweep_batch
#SBATCH --output=logs/sweep_batch_size_%j.out
#SBATCH --error=logs/sweep_batch_size_%j.err
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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
    python "${REPO_DIR}/benchmarks/sweep_batch_size.py" \
        --config "${DATA_DIR}/config_msi.yaml" \
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
        --device cuda \
        "$@"
