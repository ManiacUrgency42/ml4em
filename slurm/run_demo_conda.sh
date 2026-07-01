#!/bin/bash
# ==============================================================================
# run_demo_conda.sh — Run the ml4em demo on MSI using the conda environment
#
# Conda alternative to slurm/run_demo.sh (Apptainer path). Uses the ml4em-gpu
# conda environment created by slurm/setup_conda.sh.
#
# Prerequisites
# -------------
# 1. ml4em-gpu conda environment exists (run: sbatch slurm/setup_conda.sh)
# 2. wdb_sources.csv at /scratch.global/$USER/ml4em_data/wdb_sources.csv
# 3. config_msi.yaml at /scratch.global/$USER/ml4em_data/config_msi.yaml
#    (see docs/conda-deployment.md § 4c for the MSI config template)
# 4. .env with ML4EM_ZTF_TOKEN at /scratch.global/$USER/ml4em_data/.env
#
# Usage (from your ml4em repo root on MSI):
#   mkdir -p logs
#   sbatch slurm/run_demo_conda.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/ml4em_demo_conda_<JOBID>.out
#
# Output:
#   /scratch.global/$USER/ml4em_data/features/demo.parquet
#   /scratch.global/$USER/ml4em_data/models/logistic_demo/
# ==============================================================================
#SBATCH --job-name=ml4em_demo_conda
#SBATCH --output=logs/ml4em_demo_conda_%j.out
#SBATCH --error=logs/ml4em_demo_conda_%j.err
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

DATA_DIR=/scratch.global/$USER/ml4em_data
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module purge
module load conda
module load cuda/11.8.0

# Inject ZTF token from .env into the environment so run_demo.py can read it
if [[ -f "${DATA_DIR}/.env" ]]; then
    set -a
    source "${DATA_DIR}/.env"
    set +a
fi

conda run -n ml4em-gpu \
    python "${REPO_DIR}/scripts/run_demo.py" \
        --config "${DATA_DIR}/config_msi.yaml"
