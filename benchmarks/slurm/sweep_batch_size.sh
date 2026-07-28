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
#SBATCH -p msigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

apptainer run --nv \
    --bind "${REPO_DIR}:/app/ml4em" \
    --bind "${DATA_DIR}:/data" \
    --env-file "${DATA_DIR}/.env" \
    --env PYTHONPATH=/data/pyshim \
    "${SIF}" \
    python benchmarks/sweep_batch_size.py \
        --config /data/config_msi.yaml \
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
        --device cuda \
        "$@"
