# Data Contracts

Four dataclasses are defined in `src/ml4em/types.py`. They are the **only objects
that cross layer boundaries**. All inter-layer communication goes through these types —
no raw dicts, tuples, or numpy arrays with implicit structure.

---

## LightCurve — Data → Features

Single-band photometric time series for one source.

```python
@dataclass
class LightCurve:
    source_id : str
    time      : np.ndarray   # shape (N,)
    mag       : np.ndarray   # shape (N,)
    mag_err   : np.ndarray   # shape (N,)
    band      : Band         # "u"|"g"|"r"|"i"|"z"|"y"
    survey    : Survey       # "ztf"|"rubin"|"simulated"
    ra        : float
    dec       : float
```

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `str` | Survey-native identifier |
| `time` | `ndarray (N,)` | Observation timestamps (MJD or HJD) |
| `mag` | `ndarray (N,)` | Apparent magnitude — **smaller = brighter** |
| `mag_err` | `ndarray (N,)` | 1-sigma uncertainty in magnitude |
| `band` | `Band` | Photometric filter: `u g r i z y` |
| `survey` | `Survey` | `"ztf"` \| `"rubin"` \| `"simulated"` |
| `ra` | `float` | Right ascension, decimal degrees (J2000) |
| `dec` | `float` | Declination, decimal degrees (J2000) |

**Validation rules** (enforced in `__post_init__`):
- `time`, `mag`, `mag_err` must all have the same length (N)
- All arrays must be 1-dimensional

**One object per band:** a source observed in ZTF g, r, and i produces three
`LightCurve` objects. The feature layer receives `list[list[LightCurve]]` — the outer
list is one entry per source, the inner list contains all bands for that source.

See [Light Curves](background/light-curves.md) for a full explanation of magnitude, MJD,
and bands.

---

## FeatureVector — Features → Training / Inference

Fully extracted feature set for one source. All float fields default to `np.nan`.

```python
@dataclass
class FeatureVector:
    source_id        : str
    survey           : Survey
    ra               : float
    dec              : float
    # ... 46 more fields
```

### Field groups

**Sky position (2 fields)**

| Field | Description |
|-------|-------------|
| `ra`, `dec` | Sky position in decimal degrees |

**Light curve statistics (22 fields)** — computed by `StatisticsExtractor`

| Field | Description |
|-------|-------------|
| `n_obs` | Number of observations |
| `median` | Median apparent magnitude |
| `wmean` | Error-weighted mean magnitude |
| `chi2red` | Reduced chi-squared |
| `roms` | Ratio of median scatter to sigma |
| `wstd` | Error-weighted standard deviation |
| `norm_peak_to_peak_amp` | Normalized peak-to-peak amplitude |
| `norm_excess_var` | Normalized excess variance |
| `median_abs_dev` | Median absolute deviation |
| `iqr` | Interquartile range |
| `i60r` `i70r` `i80r` `i90r` | Percentile range ratios |
| `skew` | Skewness |
| `small_kurt` | Small-sample kurtosis |
| `inv_von_neumann` | Inverse Von Neumann ratio |
| `stetson_i` `stetson_j` `stetson_k` | Stetson variability indices |
| `anderson_darling` | Anderson-Darling normality statistic |
| `shapiro_wilk` | Shapiro-Wilk W statistic |

See [Variability Statistics](background/variability-statistics.md) for definitions.

**Period features (3 fields)** — computed by `PeriodExtractor`

| Field | Description |
|-------|-------------|
| `period` | Best period in days |
| `period_significance` | Algorithm confidence score |
| `period_algorithm` | Which algorithm(s) found the period (`str`, not a scalar) |

**Fourier features (14 fields)** — computed by `PeriodExtractor`

| Field | Description |
|-------|-------------|
| `f1_power` | Power of Fourier model at best period |
| `f1_bic` | Bayesian Information Criterion (lower = better fit) |
| `f1_a` | Cosine coefficient of 1st harmonic |
| `f1_b` | Sine coefficient of 1st harmonic |
| `f1_amp` | Amplitude of 1st harmonic |
| `f1_phi0` | Phase offset of 1st harmonic |
| `f1_relamp1–4` | Amplitudes of harmonics 2–5 relative to 1st |
| `f1_relphi1–4` | Phases of harmonics 2–5 relative to 1st |

See [Period Finding](background/period-finding.md) for definitions.

**dm/dt image (1 field)**

| Field | Type | Description |
|-------|------|-------------|
| `dmdt` | `ndarray (26, 26)` | L2-normalized Δmag/Δt pairwise histogram |

See [The dm/dt Histogram](background/dmdt.md) for a full explanation.

**Gaia catalog features (4 fields)** — computed by `CatalogExtractor` (stub)

| Field | Description |
|-------|-------------|
| `gaia_parallax` | Parallax in milliarcseconds (distance proxy) |
| `gaia_parallax_error` | Parallax uncertainty in milliarcseconds |
| `gaia_bp_rp` | BP–RP colour (temperature proxy) |
| `gaia_ruwe` | Astrometric quality — RUWE > 1.4 suggests binary |

See [Gaia & Stellar Catalogs](background/gaia.md) for definitions.

### SCALAR_FIELDS

`models.SCALAR_FIELDS` is an ordered list of **43 float field names** — everything
in `FeatureVector` except `source_id`, `survey`, `period_algorithm`, and `dmdt`. This
is the input to any scalar-based model.

!!! warning "Field order is stable"
    The ordering of `SCALAR_FIELDS` is fixed. Changing it invalidates any previously
    saved model that was trained on that ordering. Never reorder `SCALAR_FIELDS` without
    retraining all models.

---

## LabeledSample — Label preparation → Training

```python
@dataclass
class LabeledSample:
    feature : FeatureVector
    label   : int   # 1 = positive class, 0 = background
```

Wraps a `FeatureVector` with its ground-truth label. Labels are never generated by
ml4em — they must be supplied by the researcher (e.g. from a catalog cross-match).

Labels must be binary: `1` = positive class (whatever you're looking for), `0` =
background. See [Preparing Labels](guides/label-preparation.md) for how to create them.

---

## Candidate — Inference → Output

Immutable inference result for one source (`frozen=True` on the dataclass — no field
can be changed after creation).

```python
@dataclass(frozen=True)
class Candidate:
    source_id        : str
    ra               : float
    dec              : float
    survey           : Survey
    probability      : float
    period           : float
    period_algorithm : str
    confidence       : Confidence   # "high" | "medium" | "low"
```

| Field | Type | Description |
|-------|------|-------------|
| `source_id` `ra` `dec` `survey` | — | Source identity (copied from `FeatureVector`) |
| `probability` | `float` | P(positive class) ∈ [0, 1] |
| `period` | `float` | Best period in days (from feature layer) |
| `period_algorithm` | `str` | Which algorithm found the period |
| `confidence` | `"high"` \| `"medium"` \| `"low"` | Derived from probability thresholds |

### Confidence tiers

Confidence is assigned by `inference.postprocess.probabilities_to_candidates` using
thresholds from `InferenceConfig.confidence_thresholds`:

```yaml
inference:
  confidence_thresholds:
    high: 0.9
    medium: 0.5
```

| Probability | Confidence |
|------------|------------|
| ≥ 0.9 | `"high"` |
| 0.5 – 0.9 | `"medium"` |
| < 0.5 | `"low"` |

Thresholds are configurable. There is no science-specific meaning baked into the tiers —
set them to match the purity/completeness trade-off you need.
