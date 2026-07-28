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

ML4EM_GPU=1
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/benchmarks/slurm/_common.sh"

ml4em_run benchmarks/scaling_gpu.py \
    --config "${ML4EM_CONFIG}" \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --gpu-counts 1 2 4 \
    --trials 5 \
    "$@"

# matplotlib ships in both environments via the [plots] extra, so a failure
# here is a real error rather than a missing dependency -- do not mask it.
ml4em_run benchmarks/plot_scaling.py
