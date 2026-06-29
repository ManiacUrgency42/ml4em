# Quick Start

!!! warning "Work in progress"
    The training loop is not yet implemented. This page will be filled in once
    end-to-end execution is working. The steps below are correct for cloning and
    configuring; the run commands are placeholders.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- A ZTF or Rubin API token — set in a `.env` file (see [Deployment](deployment.md))

## 1. Clone

```bash
git clone --recurse-submodules <ml4em-repo-url>
cd ml4em
```

The `--recurse-submodules` flag is required because `periodfind` (the GPU period-finding
library) lives at `external/periodfind` as a git submodule.

## 2. Configure

```bash
cp config.example.yaml config.yaml
```

Open `config.yaml` and fill in the ZTF or Rubin connection details. API tokens go in a
`.env` file (not in `config.yaml`):

```bash
# .env
ML4EM_ZTF_TOKEN=your_token_here
ML4EM_RUBIN_TOKEN=your_token_here
```

See [Deployment](deployment.md) for where to get these.

## 3. Pull the Docker image

```bash
# CPU image (for local development)
docker pull ghcr.io/<org>/ml4em:cpu

# GPU image (for HPC / feature extraction at scale)
docker pull ghcr.io/<org>/ml4em:gpu
```

## 4. Run feature extraction

!!! note "TODO"
    Command placeholder — fill in once `ml4em.run` entry point is implemented.

```bash
# docker run --rm \
#   -v $(pwd)/config.yaml:/config.yaml \
#   -v $(pwd)/data:/data \
#   ghcr.io/<org>/ml4em:cpu \
#   python -m ml4em.run --config /config.yaml extract
```

## 5. Run inference

!!! note "TODO"
    Command placeholder — fill in once the training loop and predictor are implemented.

```bash
# docker run --rm \
#   -v $(pwd)/config.yaml:/config.yaml \
#   -v $(pwd)/data:/data \
#   ghcr.io/<org>/ml4em:cpu \
#   python -m ml4em.run --config /config.yaml predict
```

## 6. Output

The inference layer produces a `candidates.parquet` file with one row per source:

| Column | Description |
|--------|-------------|
| `source_id` | Survey-native identifier |
| `ra`, `dec` | Sky position in decimal degrees |
| `probability` | P(positive class) in [0, 1] |
| `confidence` | `"high"` / `"medium"` / `"low"` |
| `period` | Dominant period in days (from feature layer) |
| `period_algorithm` | Which algorithm(s) found the period |

Confidence tiers are set by `inference.confidence_thresholds` in `config.yaml`.
