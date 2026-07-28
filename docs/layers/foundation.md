# Foundation Layer

Provides the shared vocabulary for the entire pipeline — the four data contracts, shared constants, and configuration schema. Nothing flows *through* this layer; every other layer imports *from* it.

```
src/ml4em/
  types.py        Data contracts (LightCurve, FeatureVector, LabeledSample, Candidate)
  constants.py    Survey constants, dm/dt bin parameters
  config/
    schema.py     Pydantic models — PipelineConfig and sub-configs
    loader.py     YAML loader and env-var secret accessors
```

## Contents

- [types.py — Data contracts](#types)
- [constants.py — Shared constants](#constants)
- [config/ — Pipeline configuration](#config)

---

## `types.py` — Data contracts { #types }

Defines the four dataclasses that cross layer boundaries. See [Data Contracts](../data-contracts.md) for full field tables.

| Type | Produced by | Consumed by |
|------|-------------|-------------|
| `LightCurve` | Data layer | Feature layer |
| `FeatureVector` | Feature layer | Models, Training, Inference |
| `LabeledSample` | Training dataset | Training layer |
| `Candidate` | Inference layer | Caller |

---

## `constants.py` — Shared constants { #constants }

### dm/dt histogram parameters

Default bin edges for `DmdtExtractor`. Override via `features.dmdt` in `config.yaml`.

| Constant | Value | Description |
|----------|-------|-------------|
| `DMDT_DT_EDGES` | 27 edges, 0 → 2000 days | Δt bin edges (time axis) |
| `DMDT_DM_EDGES` | 27 edges, −8 → +8 mag | Δmag bin edges (magnitude axis) |
| `N_DT_BINS` | 26 | `len(DMDT_DT_EDGES) - 1` |
| `N_DM_BINS` | 26 | `len(DMDT_DM_EDGES) - 1` |

Both axes are **non-uniform by design** — these are the hand-tuned edges from the
reference WDB production run, and the exact values are part of the feature
definition. A histogram computed on different edges is not comparable to one
computed on these, even at the same 26×26 shape. Two properties are lost under
uniform spacing:

- Δmag spans ±8 mag so deep eclipses land in a real bin instead of saturating the
  outermost one, while the edges tighten to 0.05 mag either side of zero to resolve
  low-amplitude variability.
- Δt starts at exactly 0 and places four edges inside the first 0.12 d, resolving
  same-night pairs instead of collapsing them into one bin.

### Survey parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `ZTF_BANDS` | `("g", "r", "i")` | ZTF photometric bands |
| `ZTF_SIDEREAL_DAY` | 0.99727 days | Sidereal day length, in solar days |
| `SIDEREAL_FREQ_PER_DAY` | 24/23.9345 ≈ 1.00274 | Sidereal frequency, cycles per solar day. Used to group alias peaks into families |
| `ZTF_MIN_CADENCE_DAYS` | 5/1440 ≈ 0.003472 | Intra-night duplicate threshold (5 min) |
| `ZTF_DR16_MAX_HJD` | 2,459,951.5 | Maximum HJD in ZTF Data Release 16 |
| `XMATCH_RADIUS_ARCSEC` | 2.0 | Gaia cross-match search radius |
| `GAIA_RUWE_CLEAN` | 1.4 | RUWE threshold for a clean astrometric solution |

`ZTF_MIN_CADENCE_DAYS` is 5 minutes rather than the half hour a general-purpose
variability pipeline would use. A 30-minute threshold discards every pair of epochs
closer together than half an hour, which is precisely the regime short-period
binaries live in, and it contradicts a period search that reaches down to
`min_period_days = 0.003472` d (5 minutes). Five minutes still removes the
back-to-back exposures that would otherwise imprint the nightly cadence on the
periodogram, without deleting the signal being looked for.

---

## `config/` — Pipeline configuration { #config }

### `PipelineConfig`

Each section of `PipelineConfig` maps directly to a layer:

| Config section | Controls |
|---------------|---------|
| `sources.ztf` | `ZTFSource` — connection + data quality |
| `features` | `FeaturePipeline` and all extractors |
| `features.period` | `PeriodExtractor` — algorithm selection, period grid |
| `features.dmdt` | `DmdtExtractor` — bin parameters |
| `features.catalog` | `CatalogExtractor` — search radius |
| `storage` | `StorageConfig` — input file paths and output directories |
| `training` | `StandardTrainer` — loop parameters |
| `inference` | `StandardPredictor` — batch size, confidence thresholds |

### Assignment is validated

Every nested config model — `ZTFConfig`, `FeatureConfig`, `PeriodConfig`,
`DmdtConfig`, `CatalogConfig`, `StorageConfig`, `TrainingConfig`,
`InferenceConfig` — sets `model_config = {"validate_assignment": True}`.

Pydantic runs field validators when a model is *constructed*, but by default it
skips them when a field is *assigned* afterwards. Overriding a field after load is
the normal way scripts and benchmarks reconfigure a run:

```python
cfg = load_config("config.yaml")
cfg.features.device = "cuda"          # normalised to "gpu" by the field validator
cfg.features.feature_batch_size = 0   # raises ValidationError here, not mid-run
```

Without `validate_assignment` the first line would leave the un-normalised string
`"cuda"` in place until `periodfind.set_device()` rejected it hours later, and the
second would produce a pipeline that silently returned nothing. With it, an invalid
assignment raises at the assignment site, where the offending line is still visible.

### `StorageConfig`

All file paths used by the pipeline. Relative paths resolve from wherever you run the process. On MSI, override with absolute scratch paths in `config.yaml` — no environment variables, no magic.

**Input files** (must exist before running):

| Field | Default | Description |
|-------|---------|-------------|
| `catalog_path` | `data/wdb_sources.csv` | CSV of known target sources with `ra`, `dec` columns. Resolved to ZTF source IDs via `ZTFSource.fetch_by_position()` or `fetch_by_region()` cone search. |
| `labels_path` | `data/labels.csv` | CSV you supply; ml4em never generates labels. Columns: `source_id`, `label` (non-negative class index; 0/1 for the binary case). Read by `FeatureDataset._load_labels()`. See [Label preparation](../guides/label-preparation.md). |

**Output directories** (created automatically):

| Field | Default | Written by |
|-------|---------|------------|
| `features_dir` | `features/` | Feature layer |
| `models_dir` | `models/` | Training layer |
| `predictions_dir` | `predictions/` | Inference layer |

On MSI, override the two input paths in `config.yaml`:

```yaml
storage:
  catalog_path: /scratch.global/jin00404/ml4em/data/wdb_sources.csv
  labels_path:  /scratch.global/jin00404/ml4em/data/labels.csv
```

The `data/` directory, `config.yaml`, and all output directories are gitignored and never committed.

Model architecture hyperparameters (tree depth, estimators, dropout) are **not** in `PipelineConfig` — they live in per-model config dataclasses set in code. See [Design Principles](../architecture/design-principles.md#2-code-controls-architecture-config-controls-parameters).

### Loading config

```python
from ml4em.config import load_config, load_default_config

cfg = load_config("config.yaml")   # from file
cfg = load_config()                # looks for config.yaml in cwd
cfg = load_default_config()        # programmatic defaults, no file needed
```

### API tokens

Tokens are never stored in `config.yaml`. Set them as environment variables or in a `.env` file:

```bash
# .env  (never commit this file)
ML4EM_ZTF_TOKEN=your_kowalski_token
```

```python
from ml4em.config import get_ztf_token

token = get_ztf_token()    # reads ML4EM_ZTF_TOKEN from env or .env
```

Raises a clear error if the token is not found.

---

[Data layer →](data.md){ .md-button .md-button--primary }
