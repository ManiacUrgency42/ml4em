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
    source_id   : str
    time        : np.ndarray   # shape (N,), float64
    mag         : np.ndarray   # shape (N,), float64
    mag_err     : np.ndarray   # shape (N,), float64
    band        : Band         # "u"|"g"|"r"|"i"|"z"|"y"
    survey      : Survey       # "ztf"|"simulated"
    ra          : float
    dec         : float
    time_system : str = "hjd"  # "hjd"|"mjd"|"bjd"|"jd"|"relative"
```

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `str` | Survey-native identifier |
| `time` | `ndarray (N,) float64` | Observation timestamps in days, in the system named by `time_system` |
| `mag` | `ndarray (N,) float64` | Apparent magnitude — **smaller = brighter** |
| `mag_err` | `ndarray (N,) float64` | 1-sigma uncertainty in magnitude |
| `band` | `Band` | Photometric filter: `u g r i z y` |
| `survey` | `Survey` | `"ztf"` \| `"simulated"` |
| `ra` | `float` | Right ascension, decimal degrees (J2000) |
| `dec` | `float` | Declination, decimal degrees (J2000) |
| `time_system` | `str` | Which time scale `time` is on: `hjd`, `mjd`, `bjd`, `jd`, or `relative` |

**Validation rules** (enforced in `__post_init__`, so every consumer may rely on them):

- `time`, `mag`, `mag_err` must all have the same length (N), and all must be 1-dimensional.
- `time_system` must be one of the five recognised values. Nothing downstream can tell
  the systems apart from the values alone, and every feature here depends only on time
  *differences*, so an unlabelled mix would produce plausible-looking nonsense rather
  than an error.
- **`time` must not be float32.** A float32 array is rejected with a `ValueError`
  naming the offending `source_id`; it is not silently upcast. At HJD ~2.46e6 one
  float32 ULP is 0.25 days, so by the time such an array arrives here every pair of
  epochs less than six hours apart has already collapsed to a separation of exactly
  zero. Upcasting would satisfy the float64 guarantee while carrying that damage
  forward invisibly. Rejecting it at the boundary keeps the blame on the loader that
  produced it. Loaders must build `time` in float64.
- `mag` and `mag_err` *are* coerced to float64, since they carry no large zero point
  and lose nothing meaningful in a narrower type.
- `time` is sorted ascending, with `mag` and `mag_err` permuted to match.
  Consecutive-difference statistics (von Neumann ratio, Stetson J and K) and the dm/dt
  histogram all read neighbouring elements and are silently wrong on unsorted input
  rather than failing, so the sort happens once here instead of being trusted to every
  loader and every extractor.

Non-finite values are **not** removed. Dropping epochs is a survey loader's decision,
and doing it here would hide it from the caller.

**One object per band:** a source observed in ZTF g, r, and i produces three
`LightCurve` objects. The feature layer receives `list[list[LightCurve]]` — the outer
list is one entry per source, the inner list contains all bands for that source.

See [Light Curves](background/light-curves.md) for a full explanation of magnitude, MJD,
and bands.

---

## FeatureVector — Features → Training / Inference

Fully extracted feature set for one **(source, band)** pair. All float fields default to
`np.nan`, so a partially extracted feature set is explicit rather than absent.

```python
@dataclass
class FeatureVector:
    source_id        : str
    survey           : Survey
    band             : str = ""
    ra               : float = np.nan
    dec              : float = np.nan
    # ... 62 more fields (67 dataclass fields in total)
```

**One row per (source, band).** The feature pipeline expands each source into one entry
per photometric band, so a source observed in g and r yields two `FeatureVector` rows
sharing a `source_id` and distinguished by `band`. Independent per-band periods are
themselves a signal: a real variable repeats across bands, an artefact usually does not.

The authoritative field list is `dataclasses.fields(FeatureVector)`; the scalar subset a
model consumes is `ml4em.models.base.SCALAR_FIELDS`.

### Field groups

**Identity and sky position (5 fields)**

| Field | Description |
|-------|-------------|
| `source_id` | Survey-native identifier, shared across the bands of one source |
| `survey` | Originating survey |
| `band` | Photometric band these features were computed from |
| `ra`, `dec` | Sky position in decimal degrees, copied from the primary `LightCurve` |

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

**Period features** — computed by `PeriodExtractor`

The adopted period, plus everything needed to judge how much to trust it. Only the
first two are scalars a model consumes; the rest are diagnostics and structured
candidate lists.

| Field | Description |
|-------|-------------|
| `period` | Adopted period in days |
| `period_significance` | Algorithm-specific confidence score for that period |
| `period_algorithm` | Algorithm that found it (`str`, not a scalar) |
| `period_top` | `dict[algorithm, list[float]]` — top-N candidate periods per algorithm, best first |
| `significance_top` | `dict[algorithm, list[float]]` — matching significance values |
| `period_n_agree_pairs` `period_n_total_pairs` | Agreeing (algorithm, algorithm) pairs, and pairs compared |
| `period_agree_score` | `n_agree_pairs / n_total_pairs` |
| `period_agree_strict` | Same, counting only exact 1:1 matches |
| `period_agree_weighted` | Agreement weighted by peak rank |
| `period_best_agree` | Period from the best agreeing pair, days |
| `period_best_consensus` | Period agreed on by the most algorithms, days |
| `period_family_n_algos` | Distinct algorithms in the winning sidereal-alias family |
| `period_family_rank_score` | Rank-weighted strength of that family |
| `period_family_n_members` `period_family_n_total` | Peaks in the winning family, and families found |
| `period_family_best` | Representative period of the winning family, days |
| `period_family_algorithm` | Algorithm contributing that representative |

Two peaks "agree" when they match within tolerance at a 1:1, 1:2, 2:1, 1:3 or 3:1
harmonic ratio. The `period_family_*` group exists because ground-based cadence makes a
true frequency and its sidereal aliases indistinguishable: all top-N peaks from all
algorithms are treated as nodes, peaks linked by an alias relation are merged into
families, and the family spanning the most distinct algorithms wins. That survives the
common case where every algorithm found the same star but landed on a different alias
of its period.

`period_top` and `significance_top` are kept as dicts because the number of algorithms
is a config choice. The training layer flattens them into `period_{rank}_{ALGO}` and
`significance_{rank}_{ALGO}` parquet columns on save and reverses that on load.

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
| `dmdt` | `Optional[ndarray (26, 26)]` | L2-normalized Δmag/Δt pairwise histogram; `None` when not computed |

Shape is `(N_DM_BINS, N_DT_BINS)`. It is `None` when `features.compute_dmdt` is off, or
when the band had too few observations to form a pair.

See [The dm/dt Histogram](background/dmdt.md) for a full explanation.

**Gaia catalog features (7 fields)** — computed by `CatalogExtractor`

Nearest Gaia EDR3 source within `XMATCH_RADIUS_ARCSEC`. Every field is
`Optional[float]` and is `None` — not NaN — when no counterpart was found or when no
Kowalski client was supplied to the pipeline.

| Field | Description |
|-------|-------------|
| `gaia_parallax` | Parallax in milliarcseconds (distance proxy) |
| `gaia_parallax_error` | Parallax uncertainty in milliarcseconds |
| `gaia_g_mean_mag` | G apparent magnitude |
| `gaia_bp_mean_mag` | BP apparent magnitude |
| `gaia_rp_mean_mag` | RP apparent magnitude |
| `gaia_bp_rp` | BP−RP colour (temperature proxy) |
| `gaia_astrometric_excess_noise` | Astrometric residual noise; lower means a cleaner single-source fit |

The raw G/BP/RP magnitudes are kept alongside the derived colour. They cost nothing
extra — the cross-match already returns them — and a colour alone cannot be turned back
into them, so dropping them would decide on the model's behalf that apparent brightness
is irrelevant. G together with parallax gives absolute magnitude, which is what places a
source on the HR diagram, the standard way to separate white dwarfs from main sequence
stars of the same colour.

See [Gaia & Stellar Catalogs](background/gaia.md) for definitions.

### SCALAR_FIELDS

`models.SCALAR_FIELDS` is an ordered list of **45 field names** whose values are plain
floats, or ints castable to float. `N_SCALAR_FEATURES` is its length.

It excludes the string and structured fields: `source_id`, `survey`, `band`,
`period_algorithm`, `period_family_algorithm`, the `period_top` /
`significance_top` dicts, and the `dmdt` image. It also excludes `ra` and `dec`, which
identify a source rather than describe its variability. `features_to_array()` turns a
`list[FeatureVector]` into a `(N, 45)` float32 array in this order, mapping `None` Gaia
values to NaN, because an unmatched source is missing data rather than a zero.

A name in `SCALAR_FIELDS` that `FeatureVector` does not declare would be invisible at
runtime — `features_to_array()` would fall back to NaN and every model would train on a
dead column. That mismatch is checked once at import, so a typo fails loudly.

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
    label   : int   # non-negative class index
```

Wraps a `FeatureVector` with its ground-truth label. Labels are never generated by
ml4em — they must be supplied by the researcher (e.g. from a catalog cross-match).

`label` is a **non-negative class index**, not a strict 0/1 flag. Binary use is the
common case (`0` = background, `1` = the positive class you are looking for), but the
feature store is not specific to binary classification and a multi-class head is an
equally valid consumer of it. `FeatureDataset._load_labels()` accepts any non-negative
integer and rejects negatives, since a negative label is almost always a parse error
rather than a real class.

`FeatureDataset.positive_fraction()` counts `label == 1` specifically and is therefore
only meaningful for the binary case; `class_counts()` is the general form.

See [Preparing Labels](guides/label-preparation.md) for how to create the labels CSV.

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
    high:   0.9   # default
    medium: 0.7   # default
```

| Probability | Confidence |
|------------|------------|
| ≥ 0.9 | `"high"` |
| 0.7 – 0.9 | `"medium"` |
| < 0.7 | `"low"` |

Both keys are required, each must lie in [0, 1], and `high` must be strictly greater
than `medium`; `InferenceConfig` rejects any other combination at load time.

Thresholds are configurable. There is no science-specific meaning baked into the tiers —
set them to match the purity/completeness trade-off you need.
