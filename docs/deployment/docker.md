# Docker Builds

ml4em bundles `periodfind` as a git submodule at `external/periodfind` and bakes the
entire build stack into Docker images.

## Why Docker, not conda

`periodfind` has a two-layer build:

1. **`periodfind_cpu`** — a Rust extension built with `maturin` (requires the Rust toolchain)
2. **`periodfind`** — a Cython extension; CUDA GPU extensions compile automatically
   when `nvcc` is present, skipped silently when absent

Managing Rust + maturin + optional CUDA in a conda environment on HPC clusters caused
repeated dependency conflicts. Docker bakes the full toolchain once and ships a
reproducible binary image.

## Build targets

The `Dockerfile` defines two named targets:

| Target | Base image | GPU extensions | Use for |
|--------|-----------|---------------|---------|
| `cpu` | `ubuntu:22.04` | No (`nvcc` absent) | Local dev, CI, CPU-only runs |
| `gpu` | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` | Yes (`nvcc` present) | HPC GPU nodes |

Both targets execute the same build sequence:

1. Install Python 3.11, build tools, Rust toolchain via `rustup`
2. Build `periodfind_cpu` from `external/periodfind/rust` with `maturin build --release`
3. Install `periodfind` from `external/periodfind` (`setup.py` auto-detects `nvcc`)
4. Install `ml4em` (core dependencies only)

## Build commands

Run from the `ml4em/` directory (where `Dockerfile` lives):

```bash
# CPU image
docker build --platform linux/amd64 --target cpu -t ghcr.io/<org>/ml4em:cpu .

# GPU image
docker build --platform linux/amd64 --target gpu -t ghcr.io/<org>/ml4em:gpu .
```

!!! note "Apple Silicon"
    The `--platform linux/amd64` flag is required when building on Apple Silicon (M1/M2/M3)
    to produce images compatible with MSI's x86_64 nodes. Without it, Docker builds
    an arm64 image that won't run on x86 servers.

## Submodule setup

When cloning ml4em for the first time, initialize the submodule:

```bash
git clone --recurse-submodules <ml4em-repo-url>
# or, after a plain clone:
git submodule update --init
```

## Updating periodfind

The submodule is pinned to a specific commit. To advance it:

```bash
cd external/periodfind
git fetch origin
git checkout <new-commit-or-tag>
cd ../..
git add external/periodfind
git commit -m "chore: bump periodfind to <new-version>"
```

Rebuild and push a new Docker image after updating.
