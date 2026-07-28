#!/bin/bash
# ==============================================================================
# scaling_gpu.sh — Multi-GPU strong scaling for period finding
#
# Fetches a real ZTF region once, caches it, then measures aggregate
# period-finding throughput at 1, 2 and 4 GPUs.  Each measurement is
# repeated so the plot can show run-to-run spread rather than one number.
#
# Requests 4 A100s on a single node.  A single node is required because
# the benchmark shards the workload across local devices via
# CUDA_VISIBLE_DEVICES — it does not use MPI and cannot span nodes.
#
# Run sweep_batch_size.sh first so --batch-size below reflects the
# per-GPU optimum; scaling measured at a too-small batch understates
# what the hardware can do.
#
# Usage (from ml4em repo root on MSI login node):
#   mkdir -p logs
#   sbatch benchmarks/slurm/scaling_gpu.sh
#
#   # Override GPU counts or trial count:
#   sbatch benchmarks/slurm/scaling_gpu.sh --gpu-counts 1 2 4 --trials 7
# ==============================================================================
#SBATCH --job-name=ml4em_scaling_gpu
#SBATCH --output=logs/scaling_gpu_%j.out
#SBATCH --error=logs/scaling_gpu_%j.err
#SBATCH -p msigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
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

apptainer run --nv \
    --bind "${REPO_DIR}:/app/ml4em" \
    --bind "${DATA_DIR}:/data" \
    --env-file "${DATA_DIR}/.env" \
    --env PYTHONPATH=/data/pyshim \
    "${SIF}" \
    python benchmarks/scaling_gpu.py \
        --config /data/config_msi.yaml \
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
        --gpu-counts 1 2 4 \
        --trials 5 \
        "$@"

# matplotlib ships in the image via the [plots] extra, so a failure here
# is a real error rather than a missing dependency -- do not mask it.
apptainer run --nv \
    --bind "${REPO_DIR}:/app/ml4em" \
    --bind "${DATA_DIR}:/data" \
    --env-file "${DATA_DIR}/.env" \
    --env PYTHONPATH=/data/pyshim \
    "${SIF}" \
    python benchmarks/plot_scaling.py
