#!/bin/bash
# ==============================================================================
# setup_conda.sh — One-time conda environment setup on MSI (GPU)
#
# Creates the ml4em-gpu conda environment, compiles periodfind with CUDA
# extensions, and installs ml4em in editable mode. This is the conda
# equivalent of slurm/pull_image.sh (Apptainer path).
#
# The Rust build can exceed the login node's 15-minute CPU limit, so this
# runs as a SLURM job on a compute node. No GPU is needed to compile — a
# CPU node suffices because nvcc is available via the cuda module.
#
# Usage (from your ml4em repo root on MSI):
#   mkdir -p logs
#   sbatch slurm/setup_conda.sh
#
# After the job completes, activate with:
#   module load conda
#   conda activate ml4em-gpu
#
# To re-create a stale environment:
#   conda env remove -n ml4em-gpu --yes
#   sbatch slurm/setup_conda.sh
# ==============================================================================
#SBATCH --job-name=ml4em_conda_setup
#SBATCH --output=logs/ml4em_conda_setup_%j.out
#SBATCH --error=logs/ml4em_conda_setup_%j.err
#SBATCH -p amdsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH -A cough052
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jin00404@umn.edu

module purge
module load conda
module load cuda/11.8.0   # provides nvcc for periodfind CUDA extensions

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda env create -f "${REPO_DIR}/environment-gpu.yml"

conda run -n ml4em-gpu bash "${REPO_DIR}/scripts/setup_conda.sh" gpu
