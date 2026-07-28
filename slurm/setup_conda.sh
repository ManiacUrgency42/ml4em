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
#SBATCH -p msismall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH -A cough052
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jin00404@umn.edu

module purge
module load conda
module load cuda/11.8.0   # provides nvcc for periodfind CUDA extensions

# sbatch copies the submitted script to /var/spool on the compute node, so
# BASH_SOURCE points there and not at the checkout.  SLURM_SUBMIT_DIR is the
# directory sbatch was called from; the BASH_SOURCE form is the fallback for
# running this script directly with bash.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

conda env create -f "${REPO_DIR}/environment-gpu.yml"

conda run --no-capture-output -n ml4em-gpu bash "${REPO_DIR}/scripts/setup_conda.sh" gpu
