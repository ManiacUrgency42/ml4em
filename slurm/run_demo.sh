#!/bin/bash
# ==============================================================================
# run_demo.sh — Run the ml4em end-to-end demo on MSI (GPU node)
#
# Runs scripts/run_demo.py inside the ml4em:gpu Apptainer container.
# Bind-mounts your live git checkout so no image rebuild is needed for
# code changes (only rebuild when periodfind / CUDA / pyproject.toml change).
#
# Prerequisites
# -------------
# 1. ml4em_gpu.sif exists at /scratch.global/$USER/ml4em_gpu.sif
#    (run: sbatch slurm/pull_image.sh to pull it)
# 2. wdb_sources.csv is at /scratch.global/$USER/ml4em_data/wdb_sources.csv
# 3. config_msi.yaml is at /scratch.global/$USER/ml4em_data/config_msi.yaml
# 4. .env file with ML4EM_ZTF_TOKEN is at /scratch.global/$USER/ml4em_data/.env
#
# Usage (from ml4em repo root on MSI login node):
#   mkdir -p logs
#   sbatch slurm/run_demo.sh
#
# Output directories (created automatically by the demo):
#   /scratch.global/$USER/ml4em_data/features/demo.parquet
#   /scratch.global/$USER/ml4em_data/models/logistic_demo/
# ==============================================================================

#SBATCH --job-name=ml4em_demo
#SBATCH --output=logs/ml4em_demo_%j.out
#SBATCH --error=logs/ml4em_demo_%j.err
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH -A cough052
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jin00404@umn.edu

# ── Resolve paths ─────────────────────────────────────────────────────────────
SIF=/scratch.global/$USER/ml4em_gpu.sif
DATA_DIR=/scratch.global/$USER/ml4em_data

# Repo root (where this script lives under slurm/)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load apptainer

# ── Apptainer env ─────────────────────────────────────────────────────────────
export APPTAINER_CACHEDIR=/scratch.global/$USER/apptainer_cache
export TMPDIR=/scratch.global/$USER/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$TMPDIR"

# ── Run demo ──────────────────────────────────────────────────────────────────
# --nv            : pass through NVIDIA GPU drivers to the container
# --bind REPO:..  : live git checkout → code changes without image rebuild
# --bind DATA:..  : scratch data dir → catalog, .env, output features/models
# --env-file      : injects ML4EM_ZTF_TOKEN from .env into the container

apptainer run --nv \
    --bind "${REPO_DIR}:/app/ml4em" \
    --bind "${DATA_DIR}:/data" \
    --env-file "${DATA_DIR}/.env" \
    "${SIF}" \
    python scripts/run_demo.py --config /data/config_msi.yaml
