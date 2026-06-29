#!/bin/bash
# ==============================================================================
# pull_image.sh — Pull ml4em GPU container from GHCR and convert to Apptainer
#
# Pulls the ml4em:gpu Docker image from GitHub Container Registry and converts
# it to a .sif (Singularity Image Format) file for use with Apptainer on MSI.
#
# This job is expected to take 20-40 minutes. The image is ~5-7 GB compressed
# and Apptainer must download every layer, decompress them, merge into a single
# filesystem, and recompress as squashfs. This is normal — do not assume the
# job is stalled.
#
# xattr warnings in the .err log are harmless. MSI's scratch filesystem does
# not support extended attributes; Apptainer warns once and continues.
#
# Output: /scratch.global/$USER/ml4em_gpu.sif
#
# Usage:
#   mkdir -p logs
#   sbatch slurm/pull_image.sh
# ==============================================================================
#SBATCH --job-name=pull_ml4em.job
#SBATCH --output=logs/pull_ml4em_%j.out
#SBATCH --error=logs/pull_ml4em_%j.err
#SBATCH -p amdsmall
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --mem 16G
#SBATCH --time=02:00:00
#SBATCH -A cough052
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jin00404@umn.edu

module purge
module load apptainer

export APPTAINER_CACHEDIR=/scratch.global/$USER/apptainer_cache
export TMPDIR=/scratch.global/$USER/tmp
mkdir -p $APPTAINER_CACHEDIR $TMPDIR

apptainer pull \
    /scratch.global/$USER/ml4em_gpu.sif \
    docker://ghcr.io/maniacurgency42/ml4em:gpu
