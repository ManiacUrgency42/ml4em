# Feature Layer

Converts raw light curves into fixed-length numerical representations (`FeatureVector` objects) for the model. All computationally intensive work is delegated to **periodfind**, a GPU-accelerated Rust/CUDA library.

!!! tip "periodfind"
    periodfind is an external compiled library that powers all period-finding and statistics computation in this layer. For a technical deep-dive — how Rust, CUDA C++, and Cython fit together — see [Architecture → periodfind](../architecture/periodfind.md).

**Consumes:** `list[list[LightCurve]]` — outer list is sources, inner list is bands per source

**Emits:** `list[FeatureVector]` — one per source, with 43 scalar fields + 26×26 dm/dt image

```
src/ml4em/features/
  base.py         FeatureExtractor Protocol
  statistics.py   StatisticsExtractor   [implemented]
  period.py       PeriodExtractor       [implemented]
  dmdt.py         DmdtExtractor         [implemented]
  catalog.py      CatalogExtractor      [stub]
  pipeline.py     FeaturePipeline       [implemented]
```

## Contents

- [FeatureExtractor Protocol](#featureextractor)
- [FeaturePipeline](#featurepipeline)
- [StatisticsExtractor](#statisticsextractor)
- [PeriodExtractor](#periodextractor)
- [DmdtExtractor](#dmdtextractor)
- [CatalogExtractor (stub)](#catalogextractor)

---

## `FeatureExtractor` Protocol { #featureextractor }

The contract every extractor must satisfy. Extractors are called by `FeaturePipeline` — never directly.

**Consumes:** `list[list[LightCurve]]` — one list of bands per source

**Emits:** `list[dict[str, Any]]` — one dict per source mapping `FeatureVector` field names to values

```python
class FeatureExtractor(Protocol):
    def extract(self, sources: list[list[LightCurve]]) -> list[dict[str, Any]]: ...
```

`extract` must **never raise**. On failure, return an empty dict `{}` for that source; the pipeline fills those fields with `np.nan`.

---

## `FeaturePipeline` { #featurepipeline }

Composes extractors in order and assembles the resulting dicts into `FeatureVector` objects. This is the entry point for the feature layer.

**Consumes:** `list[list[LightCurve]]` — sources grouped by band

**Emits:** `list[FeatureVector]` — one per source; sources below `min_observations` return an all-NaN vector

```python
from ml4em.features import FeaturePipeline
from ml4em.config import load_config

pipeline = FeaturePipeline.default(load_config().features)
feature_vectors = pipeline.run_batch(grouped_lcs)
```

For a custom extractor set:

```python
pipeline = FeaturePipeline(
    extractors=[stats, period],
    min_observations=50,
    compute_dmdt=False,
    device="auto",
    batch_size=1000,
)
```

### Device and batching

```yaml
features:
  device: auto             # "cpu" | "gpu" | "auto"
  feature_batch_size: 1000
```

`device` controls whether periodfind uses CPU (Rust) or GPU (CUDA). `auto` uses GPU if available and falls back to CPU. `feature_batch_size` controls memory usage per periodfind call.

### Minimum observations

Sources with fewer than `min_observations` observations in their primary band (default: 50) skip all extractors and return an all-NaN `FeatureVector`.

---

## `StatisticsExtractor` { #statisticsextractor }

Computes 22 scalar light curve variability statistics using `periodfind.BasicStats`.

**Consumes:** Primary band light curve (the band with the most observations) per source

**Emits:** 22 scalar fields in `FeatureVector` — see [Variability Statistics](../background/variability-statistics.md) for definitions

```python
from ml4em.features.statistics import StatisticsExtractor

extractor = StatisticsExtractor()
results = extractor.extract(grouped_lcs)   # list[dict] — 22 keys per source
```

Casts time/mag/error arrays to float32, then calls `periodfind.BasicStats().calc(times, mags, errs)` in a single batched call over all N sources. Column names are remapped from `periodfind.BasicStats.STAT_NAMES` to `FeatureVector` field names via `_STAT_NAME_MAP`.

---

## `PeriodExtractor` { #periodextractor }

Finds the dominant period using multiple algorithms and computes 14 Fourier decomposition features.

**Consumes:** Primary band light curve per source

**Emits:** `period`, `period_algorithm`, and 14 Fourier fields (`f1_power`, `f1_bic`, `f1_a`, `f1_b`, `f1_amp`, `f1_phi0`, `f1_relamp1–4`, `f1_relphi1–4`)

```python
from ml4em.features.period import PeriodExtractor
from ml4em.config import load_config

extractor = PeriodExtractor(load_config().features.period)
results = extractor.extract(grouped_lcs)   # list[dict] — 15 keys per source
```

Configure via `config.yaml`:

```yaml
features:
  period:
    algorithms: [CE, AOV, LS, MHF]
    min_period_days: 0.003
    max_period_days: 30.0
    n_freq_grid: 10000
```

### Supported algorithms

| Key | periodfind class | Default parameters |
|-----|------------------|--------------------|
| `CE` | `ConditionalEntropy` | `n_phase=20, n_mag=10` |
| `AOV` | `AOV` | `n_phase=20` |
| `LS` | `LombScargle` | — |
| `MHF` | `MultiHarmonicFourier` | `max_harmonics=5` |
| `FPW` | `FPW` | `n_bins=10` |
| `BLS` | `BoxLeastSquares` | `n_bins=50` |

The default production set (CE, AOV, LS, MHF) matches the upstream scope-ml pipeline.

### Agreement scoring

Each algorithm runs in one batched call and returns its top period candidates. `_agree()` then:

1. Checks all algorithm pairs for period agreement (within 2% fractional tolerance)
2. Reports the period confirmed by the most algorithms
3. Falls back to the highest-significance single result if no two algorithms agree

### Fourier decomposition

After period finding, `periodfind.FourierDecomposition().calc()` runs on sources with a valid period and returns 14 features per source:

```
[power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
```

---

## `DmdtExtractor` { #dmdtextractor }

Computes a 26×26 Δmag/Δt pairwise histogram using `periodfind.DmDt`.

**Consumes:** Primary band light curve per source

**Emits:** `dmdt` field in `FeatureVector` — shape `(26, 26)` float32 array, L2-normalized per source

```python
from ml4em.features.dmdt import DmdtExtractor
from ml4em.config import load_config

extractor = DmdtExtractor(load_config().features.dmdt)
results = extractor.extract(grouped_lcs)   # list[dict] — one "dmdt" key per source
```

Δt bin edges (log-spaced) and Δmag bin edges (linear) are built once at construction and reused. To skip this extractor:

```yaml
features:
  compute_dmdt: false
```

See [The dm/dt Histogram](../background/dmdt.md) for a full explanation.

---

## `CatalogExtractor` *(stub)* { #catalogextractor }

Will cross-match each source against Gaia EDR3 within 2 arcseconds and return 4 astrometric features.

**Consumes:** `(ra, dec)` from each source's `LightCurve`

**Emits:** `gaia_parallax`, `gaia_parallax_error`, `gaia_bp_rp`, `gaia_ruwe`

Planned backends: astroquery TAP+ or Kowalski Gaia cone search.

> **Status:** returns empty dicts for all sources — all 4 Gaia fields remain `np.nan`.

---

[← Data](data.md){ .md-button } [Models →](models.md){ .md-button .md-button--primary }
