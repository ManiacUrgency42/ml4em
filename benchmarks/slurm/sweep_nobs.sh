#!/bin/bash
# ==============================================================================
# sweep_nobs.sh — n_obs vs period finding time on real ZTF light curves
#
# Fetches a real ZTF region, groups light curves into observation-count
# buckets, then times period finding per bucket.  Outputs a breakdown of
# ms/source by n_obs range and a projected GPU-hour estimate for your
# total source count.
#
# This is the benchmark to run when estimating compute hours for a cluster
# allocation request.  The ms/source column by n_obs bucket lets you project
# total GPU time given your target field's n_obs distribution.
#
# Run after sweep_batch_size.sh so you're timing with the optimal batch size.
#
# Usage (from ml4em repo root on MSI login node):
#   mkdir -p logs
#   sbatch benchmarks/slurm/sweep_nobs.sh
#
#   # Custom bucket boundaries:
#   sbatch benchmarks/slurm/sweep_nobs.sh --buckets 50 100 300 500 1000 2000
# ==============================================================================
#SBATCH --job-name=ml4em_sweep_nobs
#SBATCH --output=logs/sweep_nobs_%j.out
#SBATCH --error=logs/sweep_nobs_%j.err
#SBATCH -p msigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
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
    python benchmarks/sweep_nobs.py \
        --config /data/config_msi.yaml \
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
        --device cuda \
        "$@"
