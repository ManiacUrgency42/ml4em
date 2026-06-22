# ==============================================================================
# ml4em — multi-stage Dockerfile
#
# Two named build targets:
#   cpu  — Ubuntu 22.04, no CUDA. For local testing and CI.
#   gpu  — CUDA 11.8 base. For MSI/HPC production via Apptainer.
#
# Build:
#   docker build --target cpu -t ml4em:cpu .
#   docker build --target gpu -t ml4em:gpu .
#
# Push to GHCR (then researchers pull via Apptainer on MSI):
#   docker push ghcr.io/<org>/ml4em:cpu
#   docker push ghcr.io/<org>/ml4em:gpu
#
# MSI usage (Apptainer):
#   # One-time pull (only needed when image changes):
#   apptainer pull ml4em_gpu.sif docker://ghcr.io/<org>/ml4em:gpu
#
#   # Daily workflow — bind-mount your live git checkout, no rebuild needed:
#   apptainer run --nv \
#       --bind /users/7/jin00404/ml4em:/app/ml4em \
#       --bind /scratch.global/$USER/data:/data \
#       ml4em_gpu.sif \
#       python -m ml4em.run --config /data/config.yaml
#
#   # To pick up code changes: just git pull on MSI, no image rebuild.
#   # Only rebuild the image when periodfind, CUDA, or pyproject.toml changes.
#
# Store .sif files in /scratch (not $HOME) — they are 5–8 GB.
# ==============================================================================


# ==============================================================================
# CPU image — local development / CI testing
# ==============================================================================
FROM ubuntu:22.04 AS cpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3-pip \
        curl \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain — required to build periodfind_cpu
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip3 install --no-cache-dir maturin

# ── Build periodfind_cpu (Rust, no CUDA) ─────────────────────────────────────
COPY external/periodfind /build/periodfind
RUN cd /build/periodfind/rust \
    && maturin build --release \
    && pip3 install --no-cache-dir target/wheels/*.whl

# ── Build periodfind (Cython — no nvcc found, CPU-only extensions) ────────────
# setup.py checks for nvcc; skips CUDA extensions automatically if absent.
RUN pip3 install --no-cache-dir /build/periodfind

# ── Install ml4em (editable) ──────────────────────────────────────────────────
# -e means Python imports directly from /app/ml4em/src rather than copying
# files into site-packages. At runtime, bind-mount your live git checkout
# over /app/ml4em and code changes are picked up without rebuilding the image.
COPY . /app/ml4em
RUN pip3 install --no-cache-dir -e /app/ml4em

WORKDIR /app/ml4em


# ==============================================================================
# GPU image — MSI / HPC production
# ==============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS gpu

# Check MSI driver version before choosing a different CUDA base:
#   squeue --partition=gpu; ssh <node>; nvidia-smi
# The CUDA runtime in this image must be <= the driver version on MSI GPU nodes.
# CUDA 11.8 requires driver >= 520.x (most MSI GPU nodes satisfy this).

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3-pip \
        curl \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip3 install --no-cache-dir maturin

# ── Build periodfind_cpu (Rust) ───────────────────────────────────────────────
COPY external/periodfind /build/periodfind
RUN cd /build/periodfind/rust \
    && maturin build --release \
    && pip3 install --no-cache-dir target/wheels/*.whl

# ── Build periodfind (Cython + CUDA — nvcc IS present in this base image) ─────
RUN pip3 install --no-cache-dir /build/periodfind

# ── Install ml4em (editable) ──────────────────────────────────────────────────
# -e means Python imports directly from /app/ml4em/src rather than copying
# files into site-packages. At runtime, bind-mount your live git checkout
# over /app/ml4em and code changes are picked up without rebuilding the image.
COPY . /app/ml4em
RUN pip3 install --no-cache-dir -e /app/ml4em

WORKDIR /app/ml4em
