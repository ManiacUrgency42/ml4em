#!/bin/bash
# ==============================================================================
# benchmark.sh — Single-source throughput benchmark on MSI GPU
#
# Runs scripts/benchmark_single.py inside the Apptainer container on an A100
# node. Times each pipeline stage independently: Kowalski fetch, statistics,
# period finding, dm/dt, and Gaia catalog cross-match.
#
# Run twice to compare CPU vs GPU:
#   sbatch slurm/benchmark.sh              # uses device from config_msi.yaml
#   sbatch slurm/benchmark.sh --device cpu
#   sbatch slurm/benchmark.sh --device cuda
#
# Usage:
#   mkdir -p logs
#   sbatch slurm/benchmark.sh [--ra RA] [--dec DEC] [--device cpu|cuda]
# ==============================================================================
#SBATCH --job-name=ml4em_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -A cough052
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jin00404@umn.edu

module purge
module load apptainer

DATA=/scratch.global/$USER/ml4em_data
SIF=/scratch.global/$USER/ml4em_gpu.sif

apptainer exec --nv \
    --bind $HOME/ml4em:/app/ml4em \
    --bind $DATA:/data \
    --env-file $DATA/.env \
    $SIF \
    python scripts/benchmark_single.py \
        --config /data/config_msi.yaml \
        "$@"
