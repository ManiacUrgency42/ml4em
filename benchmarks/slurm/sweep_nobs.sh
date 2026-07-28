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

ML4EM_GPU=1
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/benchmarks/slurm/_common.sh"

ml4em_run benchmarks/sweep_nobs.py \
    --config "${ML4EM_CONFIG}" \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda \
    "$@"
