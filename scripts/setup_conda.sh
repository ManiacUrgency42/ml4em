#!/usr/bin/env bash
# ==============================================================================
# setup_conda.sh — Build periodfind and install ml4em in editable mode
#
# Run this once after creating the conda environment:
#
#   conda activate ml4em-cpu && bash scripts/setup_conda.sh cpu
#   conda activate ml4em-gpu && bash scripts/setup_conda.sh gpu
#
# Or via conda run (no manual activation needed):
#
#   conda run -n ml4em-cpu bash scripts/setup_conda.sh cpu
#   conda run -n ml4em-gpu bash scripts/setup_conda.sh gpu
#
# GPU mode requires nvcc on PATH for periodfind's CUDA extensions. On MSI:
#   module load cuda/11.8.0
#
# Usage: bash scripts/setup_conda.sh [cpu|gpu]
# ==============================================================================

set -euo pipefail

MODE=${1:-cpu}

if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
    echo "Usage: $0 [cpu|gpu]" >&2
    exit 1
fi

# Resolve repo root regardless of where the script is invoked from
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── 1. torch ──────────────────────────────────────────────────────────────────
# Install torch first so that pip satisfies [training,inference] extras later
# without pulling in a second (CPU-only) wheel on top.
echo "==> Installing torch (${MODE})..."
if [[ "$MODE" == "gpu" ]]; then
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cu118
else
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi

# ── 2. periodfind Rust extension ──────────────────────────────────────────────
echo "==> Building periodfind Rust extension..."
cd external/periodfind/rust
maturin build --release --interpreter python
pip install --quiet target/wheels/*.whl
cd "$REPO_DIR"

# ── 3. periodfind Cython extensions ──────────────────────────────────────────
echo "==> Building periodfind Cython extensions..."
if [[ "$MODE" == "gpu" ]]; then
    if ! command -v nvcc &>/dev/null; then
        echo "" >&2
        echo "ERROR: nvcc not found. Load the CUDA module before running this script:" >&2
        echo "  module load cuda/11.8.0" >&2
        echo "" >&2
        exit 1
    fi
    echo "    nvcc: $(command -v nvcc) — CUDA extensions will be built"
else
    echo "    nvcc not required for cpu mode — CPU-only Cython extensions"
fi
pip install --quiet external/periodfind

# ── 4. ml4em editable install ─────────────────────────────────────────────────
# torch is already installed above; pip sees it as satisfied and will not pull
# in a second wheel when resolving [training,inference] extras.
echo "==> Installing ml4em in editable mode..."
pip install --quiet -e ".[ztf,catalog,training,dev]"

echo ""
echo "Done. Verify the install:"
echo "  python -c 'import ml4em; import periodfind; print(\"OK\")'"
