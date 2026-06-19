# Installation

## Docker (recommended)

ml4em bundles `periodfind` (a Rust + optional CUDA library) as a git submodule and
bakes the entire build stack into Docker images. This is the recommended path because
`periodfind` cannot be installed via `pip` on Python ≥ 3.11 — it must be compiled
from source.

See [Docker Builds](deployment/docker.md) for full build instructions, and
[GitHub Container Registry](deployment/ghcr.md) for pulling pre-built images.

```bash
docker pull ghcr.io/<org>/ml4em:cpu   # CPU-only (local dev)
docker pull ghcr.io/<org>/ml4em:gpu   # GPU-enabled (HPC)
```

## From source (advanced)

!!! warning "periodfind must be compiled separately"
    The `pip install` below will succeed but the `periodfind` import will fail unless
    you have already built and installed `periodfind` from `external/periodfind/`.
    This requires Rust + maturin. Use Docker unless you have a specific reason not to.

```bash
git clone --recurse-submodules <ml4em-repo-url>
cd ml4em
pip install -e ".[dev]"
```

## Optional extras

Core install (`pip install ml4em`) includes only `numpy`, `pydantic`, `pyyaml`,
`python-dotenv`, and `periodfind`. Everything else is optional:

| Extra | What it installs | When you need it |
|-------|-----------------|-----------------|
| `ztf` | `penquins` | Fetching ZTF data via Kowalski |
| `rubin` | `pyvo`, `pyarrow` | Fetching Rubin DP1 data via TAP |
| `catalog` | `astropy` | Gaia EDR3 cross-match |
| `training` | `torch`, `scikit-learn` | Training a model |
| `inference` | `torch` | Running inference with a PyTorch model |
| `dev` | `pytest`, `ruff` | Development and testing |

```bash
pip install "ml4em[ztf,training,dev]"
```

## API tokens { #api-tokens }

ml4em never stores tokens in `config.yaml`. Tokens are read from environment variables
or a `.env` file at runtime.

### ZTF token

ZTF data is accessed via **Kowalski**, ZTF's database query service. You need a
Kowalski account to get a token.

```bash
# .env
ML4EM_ZTF_TOKEN=your_kowalski_token_here
```

### Rubin token

Rubin data is accessed via the **Rubin Science Platform (RSP)** TAP service. You need
a Rubin RSP account (currently requires an approved proposal or institutional access).

```bash
# .env
ML4EM_RUBIN_TOKEN=your_rsp_token_here
```

Tokens are loaded with `config.get_ztf_token()` / `config.get_rubin_token()`, which
read the environment variable or `.env` file. The `.env` file is never committed — it
is in `.gitignore`.

## Python version

Python 3.11 or later is required. This is a hard requirement imposed by `periodfind`'s
build system and by the use of newer typing features.
