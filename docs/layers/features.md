# Feature Layer

!!! abstract "Layer at a glance"
    **Receives:** `list[list[LightCurve]]` — outer list is sources, inner list is bands per source
    **Produces:** `list[FeatureVector]` — one per source, with 43 scalar fields + 26×26 dm/dt image
    **Protocol:** `FeatureExtractor` → `extract(sources)`; composed by `FeaturePipeline`
    **Files:** `features/base.py` · `features/statistics.py` · `features/period.py` · `features/dmdt.py` · `features/catalog.py` · `features/pipeline.py`
    **Background:** [Variability Statistics](../background/variability-statistics.md) · [Period Finding](../background/period-finding.md) · [The dm/dt Histogram](../background/dmdt.md) · [Gaia](../background/gaia.md)

The feature layer converts raw light curves into `FeatureVector` objects — the fixed-
length numerical representations that the model operates on.

```
src/ml4em/features/
  base.py         FeatureExtractor Protocol
  statistics.py   StatisticsExtractor  — 22 scalar LC statistics    [implemented]
  period.py       PeriodExtractor      — period finding + 14 Fourier [implemented]
  dmdt.py         DmdtExtractor        — 26×26 pairwise histogram    [implemented]
  catalog.py      CatalogExtractor     — 4 Gaia EDR3 features        [stub]
  pipeline.py     FeaturePipeline      — composer                    [implemented]
```

All computationally intensive work is delegated to **periodfind**, a GPU-accelerated
Rust/CUDA library. The Python code sets up parameters and reshapes inputs/outputs;
the actual number crunching happens in compiled code.

---

## How the pieces connect

```text
FeaturePipeline.run_batch(sources)              sources: list[list[LightCurve]]
  │
  ├─ [< min_observations] ───────────────────→ all-NaN FeatureVector  (skipped)
  │
  ├─→ StatisticsExtractor.extract(sources)    → list[dict]  22 scalar fields
  ├─→ PeriodExtractor.extract(sources)        → list[dict]  15 fields (period + Fourier)
  ├─→ DmdtExtractor.extract(sources)          → list[dict]  dmdt: 26×26 array
  └─→ CatalogExtractor.extract(sources)       → list[dict]  4 Gaia fields
        │
        └─ merges dicts → FeatureVector per source
```

**Entry point:** `FeaturePipeline.run_batch` — the extractors are called by it in order, never directly.

If an extractor returns `{}` for a source (on error), the pipeline fills those fields with NaN.

---

## Protocol — `FeatureExtractor`

```python
class FeatureExtractor(Protocol):
    def extract(self, sources: list[list[LightCurve]]) -> list[dict[str, Any]]: ...
```

**Batch-first interface:** the input is a list of sources, where each source is a list
of `LightCurve` objects (one per band). The output is one dict per source mapping
field names to values.

**Must never raise.** If an extractor fails for a source (network error, algorithm
divergence, too few points), it returns an empty dict `{}` for that source. The pipeline
fills in NaN for the missing fields.

---

## `StatisticsExtractor`

Computes the 22 scalar light curve variability statistics.

For each source, selects the **primary band** (the band with the most observations),
casts the time/mag/error arrays to float32, and delegates to
`periodfind.BasicStats().calc(times, mags, errs)` — a Rust-backed batched
implementation that processes all N sources in one call.

Returns an `(N, 22)` array; column names come from `periodfind.BasicStats.STAT_NAMES`
and are remapped to `FeatureVector` field names via `_STAT_NAME_MAP` in `statistics.py`.

!!! note "No sigma-clipping"
    `StatisticsExtractor` does not sigma-clip outliers before computing statistics.
    This is intentional — consistent with the upstream scope-ml pipeline's approach.
    The statistics themselves (median, MAD, Stetson indices) are chosen to be robust
    to outliers.

See [Variability Statistics](../background/variability-statistics.md) for a plain-English
explanation of all 22 statistics.

---

## `PeriodExtractor`

Finds the dominant period and computes 14 Fourier decomposition features.

### Algorithm objects

Algorithms are built once at construction time and reused across all `extract()` calls:

| Config key | periodfind class | Default parameters |
|------------|----------------|--------------------|
| `CE` | `ConditionalEntropy` | `n_phase=20, n_mag=10` |
| `AOV` | `AOV` | `n_phase=20` |
| `LS` | `LombScargle` | — |
| `MHF` | `MultiHarmonicFourier` | `max_harmonics=5` |
| `FPW` | `FPW` | `n_bins=10` |
| `BLS` | `BoxLeastSquares` | `n_bins=50` |

The default production set (from scope-ml) is CE, AOV, LS, MHF. Configure via:

```yaml
features:
  period:
    algorithms: [CE, AOV, LS, MHF]
    min_period_days: 0.003
    max_period_days: 30.0
    n_freq_grid: 10000
```

### Agreement scoring

Each algorithm runs over all N sources in one batched call and returns its top period
candidates. Then `_agree()` runs across algorithms per source:

1. Check all pairs of algorithms for period agreement (within 2% fractional tolerance)
2. Report the period confirmed by the most algorithms
3. If no two algorithms agree, fall back to highest-significance single result

### Fourier decomposition

After period finding, `periodfind.FourierDecomposition().calc()` is run on the subset
of sources with a valid period. Returns 14 features per source:

```
[power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
```

Mapped to `FeatureVector` fields: `f1_power`, `f1_bic`, `f1_a`, `f1_b`, `f1_amp`,
`f1_phi0`, `f1_relamp1–4`, `f1_relphi1–4`.

See [Period Finding](../background/period-finding.md) for a complete explanation of all
algorithms, agreement scoring, and Fourier features.

---

## `DmdtExtractor`

Computes the 26×26 Δmag/Δt pairwise histogram.

Δt bin edges (log-spaced, float32) and Δmag bin edges (linear, float32) are built
once in `__init__` using the config parameters and reused. Delegates to
`periodfind.DmDt().calc(times, mags, dt_edges, dm_edges)`, returns an
`(N, 26, 26)` float32 array, L2-normalized per source.

To skip this extractor:

```yaml
features:
  compute_dmdt: false
```

See [The dm/dt Histogram](../background/dmdt.md) for a full explanation.

---

## `CatalogExtractor` *(stub)*

Will query Gaia EDR3 for the nearest counterpart within 2 arcseconds of each source's
(ra, dec). Returns `gaia_parallax`, `gaia_parallax_error`, `gaia_bp_rp`, `gaia_ruwe`.

Two planned backends: astroquery TAP+ or Kowalski Gaia cone search.

Status: returns empty dicts for all sources. All 4 Gaia fields remain NaN.

See [Gaia & Stellar Catalogs](../background/gaia.md) for a full explanation.

---

## `FeaturePipeline`

Composes extractors in order and builds `FeatureVector` objects.

### Construction

```python
# Standard ordering (statistics → period → dmdt → catalog)
pipeline = FeaturePipeline.default(cfg.features)

# Custom extractor list
pipeline = FeaturePipeline(
    extractors=[stats, period],
    min_observations=50,
    compute_dmdt=False,
    device="auto",
    batch_size=1000,
)
```

### Running

```python
fvs = pipeline.run_batch(grouped_lcs)    # list[list[LightCurve]] → list[FeatureVector]
fv  = pipeline.run_batch([lcs])[0]       # single source (batch of one)
```

### Batching and device selection

`run_batch` calls `periodfind.set_device(device)` once before processing, then
processes `grouped_lcs` in chunks of `feature_batch_size` (default 1000).

```yaml
features:
  device: auto          # "cpu" | "gpu" | "auto"
  feature_batch_size: 1000
```

`device` controls whether periodfind uses CPU (Rust) or GPU (CUDA). `auto` uses GPU
if available and falls back to CPU. This is orthogonal to `feature_batch_size` — batch
size controls memory usage; device controls which hardware processes each batch.

### Minimum observations

Sources with fewer than `min_observations` (default 50) observations in their primary
band return an all-NaN `FeatureVector` immediately, without running any extractor.

### The all-NaN vector

When a source is skipped (too few observations) or an extractor fails, the corresponding
fields in `FeatureVector` are `np.nan`. This is intentional — see
[Design Principles → Partial execution is safe](../architecture/design-principles.md#4-partial-execution-is-safe).

---

## Adding a new extractor

1. Create `src/ml4em/features/my_extractor.py`
2. Implement `extract(sources)` — batch-first, return `list[dict]`, never raise
3. Keys in the returned dicts must match `FeatureVector` field names exactly
4. Pass to `FeaturePipeline` constructor or add to `FeaturePipeline.default()`

```python
class MyExtractor:
    def extract(self, sources: list[list[LightCurve]]) -> list[dict[str, Any]]:
        results = []
        for lcs in sources:
            try:
                value = compute_something(lcs)
                results.append({"my_feature": value})
            except Exception:
                results.append({})   # never raise — return empty dict
        return results
```

See [Guide: Add an Extractor](../guides/add-extractor.md) for step-by-step instructions.

---

[← Data](data.md){ .md-button } [Models →](models.md){ .md-button .md-button--primary }
