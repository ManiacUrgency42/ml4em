# Foundation Layer

!!! abstract "Layer at a glance"
    **Role:** Shared vocabulary — no data flows *through* this layer; every other layer imports from it.
    **Files:** `types.py` · `constants.py` · `config/schema.py` · `config/loader.py`
    **Provides:** `LightCurve`, `FeatureVector`, `LabeledSample`, `Candidate`, `PipelineConfig`, survey constants
    **Background:** [Light Curves](../background/light-curves.md) · [Surveys](../background/surveys.md)

The foundation layer is not a "layer" in the data-flow sense — nothing passes through
it. It provides the shared vocabulary that all other layers use.

```
src/ml4em/
  types.py        Data contracts (LightCurve, FeatureVector, LabeledSample, Candidate)
  constants.py    Survey constants, dm/dt bin parameters, physical constants
  config/
    schema.py     Pydantic models — PipelineConfig and all sub-configs
    loader.py     YAML loader + env-var secret accessors
```

---

## `types.py` — Data contracts

Defines the four shared dataclasses. See [Data Contracts](../data-contracts.md) for the
full field tables and explanations.

---

## `constants.py` — Shared constants { #constants-py }

### Physical constants

Used by `SimulatedSource` for orbital mechanics calculations:

| Constant | Value | Description |
|----------|-------|-------------|
| `G` | 6.674×10⁻¹¹ m³ kg⁻¹ s⁻² | Gravitational constant |
| `C` | 2.998×10⁸ m s⁻¹ | Speed of light |
| `MSUN` | 1.988×10³⁰ kg | Solar mass |
| `RSUN` | 6.957×10⁸ m | Solar radius |
| `MTSUN_SI` | G × MSUN / C³ | Solar mass in seconds (used in GW calculations) |

### dm/dt histogram parameters

These define the default bin edges for the `DmdtExtractor`. They can be overridden via
`config.yaml` under `features.dmdt`.

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
| `ZTF_SIDEREAL_DAY` | 0.99727 days | Earth's rotation period relative to stars |
| `ZTF_MIN_CADENCE_DAYS` | 30/1440 (30 min) | Intra-night duplicate threshold |
| `ZTF_DR16_MAX_HJD` | 2,459,951.5 | Maximum HJD in ZTF Data Release 16 |
| `RUBIN_BANDS` | `("u", "g", "r", "i", "z", "y")` | Rubin photometric bands |

### Cross-match parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `XMATCH_RADIUS_ARCSEC` | 2.0 arcsec | Gaia search radius |
| `GAIA_RUWE_CLEAN` | 1.4 | RUWE threshold for "clean" astrometric solution |

---

## `config/` — Pipeline configuration

### `PipelineConfig` — the pipeline as configuration

`PipelineConfig` maps one-to-one onto the pipeline layers:

| Config section | Controls |
|---------------|---------|
| `PipelineConfig.sources.ztf` | `ZTFSource` — connection + data quality |
| `PipelineConfig.sources.rubin` | `RubinSource` — TAP endpoint + table names |
| `PipelineConfig.features` | `FeaturePipeline` and all extractors |
| `PipelineConfig.features.period` | `PeriodExtractor` — algorithm selection, period grid |
| `PipelineConfig.features.dmdt` | `DmdtExtractor` — bin parameters |
| `PipelineConfig.features.catalog` | `CatalogExtractor` — search radius |
| `PipelineConfig.storage` | File paths used by all layers |
| `PipelineConfig.training` | `StandardTrainer` — loop parameters only |
| `PipelineConfig.inference` | `StandardPredictor` — batch size, confidence thresholds |

**What is NOT in `PipelineConfig`:** model architecture hyperparameters (tree depth,
number of estimators, dropout). Those live in per-model config dataclasses
(`XGBoostConfig`, etc.) and are set in code. See
[Design Principles](../architecture/design-principles.md#2-code-controls-architecture-config-controls-parameters).

### `loader.py` — loading config

```python
from ml4em.config import load_config

cfg = load_config("config.yaml")              # load from file
cfg = load_config()                           # looks for "config.yaml" in cwd
cfg = load_default_config()                  # programmatic defaults, no file needed
```

### Secrets — API tokens

Tokens are **never stored in `config.yaml`**. They are read from environment variables
or a `.env` file:

```bash
# .env  (never commit this file)
ML4EM_ZTF_TOKEN=your_kowalski_token
ML4EM_RUBIN_TOKEN=your_rsp_token
```

```python
from ml4em.config import get_ztf_token, get_rubin_token

token = get_ztf_token()    # reads ML4EM_ZTF_TOKEN from env or .env
```

If the token is not found, these functions raise a clear error rather than silently
passing an empty string.

---

[Data layer →](data.md){ .md-button .md-button--primary }
