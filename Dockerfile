# ==============================================================================
# ml4em — multi-stage Dockerfile
#
# Two named build targets:
#   cpu  — python:3.11-slim-bookworm, no CUDA. For local testing and CI.
#   gpu  — CUDA 11.8 / Ubuntu 22.04 base. For MSI/HPC production via Apptainer.
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
# python:3.11-bookworm (buildpack-deps base) ships with build-essential, git,
# curl, and python3.11-dev pre-installed — no apt installs needed for toolchain.
# ==============================================================================
FROM python:3.11-bookworm AS cpu

# Upgrade build tools to satisfy pyproject.toml: requires = ["setuptools>=77"]
RUN pip install --upgrade pip setuptools wheel

# Rust toolchain — required to build periodfind_cpu
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3.11 -m pip install --no-cache-dir maturin

# ── Build periodfind_cpu (Rust, no CUDA) ─────────────────────────────────────
COPY external/periodfind /build/periodfind
RUN cd /build/periodfind/rust \
    && maturin build --release --interpreter python3.11 \
    && python3.11 -m pip install --no-cache-dir target/wheels/*.whl

# ── Build periodfind (Cython — no nvcc found, CPU-only extensions) ────────────
# setup.py checks for nvcc; skips CUDA extensions automatically if absent.
RUN python3.11 -m pip install --no-cache-dir /build/periodfind

# ── Install ml4em with test + ZTF extras (editable) ──────────────────────────
# -e means Python imports directly from /app/ml4em/src rather than copying
# files into site-packages. At runtime, bind-mount your live git checkout
# over /app/ml4em and code changes are picked up without rebuilding the image.
# [ztf,dev] installs penquins (ZTF data access) and pytest/ruff (test tooling).
COPY . /app/ml4em
RUN python3.11 -m pip install --no-cache-dir -e /app/ml4em[ztf,dev]

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

RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/80-retries \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        curl \
        ca-certificates \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.11 and upgrade build tools
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3.11 -m pip install --upgrade pip setuptools wheel

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3.11 -m pip install --no-cache-dir maturin

# ── Build periodfind_cpu (Rust) ───────────────────────────────────────────────
COPY external/periodfind /build/periodfind
RUN cd /build/periodfind/rust \
    && maturin build --release --interpreter python3.11 \
    && python3.11 -m pip install --no-cache-dir target/wheels/*.whl

# ── Build periodfind (Cython + CUDA — nvcc IS present in this base image) ─────
RUN python3.11 -m pip install --no-cache-dir /build/periodfind

# ── Install ml4em with training extras (editable) ────────────────────────────
# [training] adds torch, pandas, pyarrow — everything needed to train the
# logistic model and persist/reload feature vectors.
# The torch wheel from PyPI is CPU-only; periodfind uses CUDA directly via
# its own GPU kernel — torch does not need CUDA bindings for this demo model.
COPY . /app/ml4em
RUN python3.11 -m pip install --no-cache-dir -e "/app/ml4em[training]"

WORKDIR /app/ml4em
