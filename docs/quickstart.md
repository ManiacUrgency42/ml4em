# Quick Start

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) — for local CPU testing
- A Kowalski ZTF token — set in a `.env` file (see below)

## 1. Clone

```bash
git clone --recurse-submodules <ml4em-repo-url>
cd ml4em
```

`--recurse-submodules` is required because `periodfind` (the GPU period-finding
library) lives at `external/periodfind` as a git submodule.

## 2. Configure

```bash
cp config.example.yaml config.yaml
```

Open `config.yaml` and verify the ZTF connection settings. Your Kowalski token goes
in a `.env` file, not in `config.yaml`:

```bash
# .env  (gitignored — never commit this)
ML4EM_ZTF_TOKEN=your_kowalski_token_here
```

## 3. Add your catalog

Place your WDB source catalog at the path set by `storage.catalog_path`
(default: `data/wdb_sources.csv`). The file must have at minimum `ra` and `dec`
columns in decimal degrees:

```
obj_id,ra,dec
WDB_001,123.456,-45.678
...
```

## 4. Pull the CPU image (local testing)

```bash
docker pull ghcr.io/maniacurgency42/ml4em:cpu
```

## 5. Run the demo

```bash
docker run --rm \
    -v $(pwd):/app/ml4em \
    -v $(pwd)/data:/data \
    --env-file .env \
    ghcr.io/maniacurgency42/ml4em:cpu \
    python scripts/run_demo.py --config config.yaml
```

This runs the full end-to-end pipeline:

1. Reads `wdb_sources.csv` — the WDB catalog of target sky positions
2. Queries ZTF via Kowalski cone search — positives within 2 arcsec, negatives at 2–30 arcsec
3. Extracts features (statistics, periods, dm/dt histograms)
4. Trains a `LogisticExampleClassifier` on the labeled sources
5. Saves the model and runs inference on the test set
6. Prints results grouped by confidence tier (high / medium / low)

## 6. Output

```
features/demo.parquet        — extracted feature vectors (reusable)
models/logistic_demo/        — saved model weights and manifest
```

Inference results are printed to stdout. Each source shows:

| Field | Description |
|-------|-------------|
| `source_id` | ZTF integer source ID |
| `probability` | P(WDB) in [0, 1] |
| `confidence` | `"high"` / `"medium"` / `"low"` |
| `period` | Dominant period in days |
| `period_algorithm` | Which algorithm(s) agreed on the period |

Confidence thresholds are set by `inference.confidence_thresholds` in `config.yaml`.

## Running on MSI (GPU)

For production runs on MSI with GPU-accelerated feature extraction, see [Deployment](deployment.md).
