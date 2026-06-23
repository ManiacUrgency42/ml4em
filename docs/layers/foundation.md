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
| `N_DT_BINS` | 26 | Number of Δt bins (time axis) |
| `N_DM_BINS` | 26 | Number of Δmag bins (magnitude axis) |
| `DMDT_DT_MIN` | 1×10⁻³ days | Minimum time separation (~1.4 min) |
| `DMDT_DT_MAX` | 1×10³ days | Maximum time separation (~2.7 years) |
| `DMDT_DM_MIN` | −3.0 mag | Minimum magnitude difference |
| `DMDT_DM_MAX` | +3.0 mag | Maximum magnitude difference |

### Survey parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `ZTF_BANDS` | `("g", "r", "i")` | ZTF photometric bands |
| `ZTF_SIDEREAL_DAY` | 0.99727 days | Sidereal day length |
| `ZTF_MIN_CADENCE_DAYS` | 30/1440 | Intra-night duplicate threshold (30 min) |
| `ZTF_DR16_MAX_HJD` | 2,459,951.5 | Maximum HJD in ZTF Data Release 16 |
| `RUBIN_BANDS` | `("u", "g", "r", "i", "z", "y")` | Rubin photometric bands |
| `XMATCH_RADIUS_ARCSEC` | 2.0 | Gaia cross-match search radius |
| `GAIA_RUWE_CLEAN` | 1.4 | RUWE threshold for a clean astrometric solution |

---

## `config/` — Pipeline configuration { #config }

### `PipelineConfig`

Each section of `PipelineConfig` maps directly to a layer:

| Config section | Controls |
|---------------|---------|
| `sources.ztf` | `ZTFSource` — connection + data quality |
| `sources.rubin` | `RubinSource` — TAP endpoint + table names |
| `features` | `FeaturePipeline` and all extractors |
| `features.period` | `PeriodExtractor` — algorithm selection, period grid |
| `features.dmdt` | `DmdtExtractor` — bin parameters |
| `features.catalog` | `CatalogExtractor` — search radius |
| `storage` | File paths used by all layers |
| `training` | `StandardTrainer` — loop parameters |
| `inference` | `StandardPredictor` — batch size, confidence thresholds |

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
ML4EM_RUBIN_TOKEN=your_rsp_token
```

```python
from ml4em.config import get_ztf_token, get_rubin_token

token = get_ztf_token()    # reads ML4EM_ZTF_TOKEN from env or .env
```

Raises a clear error if the token is not found.

---

[Data layer →](data.md){ .md-button .md-button--primary }
