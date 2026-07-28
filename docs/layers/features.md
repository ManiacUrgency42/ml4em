# Feature Layer

Converts raw light curves into fixed-length numerical representations (`FeatureVector` objects) for the model. All computationally intensive work is delegated to **periodfind**, a GPU-accelerated Rust/CUDA library.

!!! tip "periodfind"
    periodfind is an external compiled library that powers all period-finding and statistics computation in this layer. For a technical deep-dive — how Rust, CUDA C++, and Cython fit together — see [Background → periodfind](../background/periodfind.md).

**Consumes:** `list[list[LightCurve]]` — outer list is sources, inner list is bands per source

**Emits:** `list[FeatureVector]` — one per **(source, band)**, with 45 scalar fields + an optional 26×26 dm/dt image

```
src/ml4em/features/
  base.py         FeatureExtractor Protocol, to_float32_time()
  statistics.py   StatisticsExtractor
  period.py       PeriodExtractor
  dmdt.py         DmdtExtractor
  catalog.py      CatalogExtractor
  pipeline.py     FeaturePipeline
```

## Contents

- [FeatureExtractor Protocol](#featureextractor)
- [Time precision — `to_float32_time`](#time-precision)
- [FeaturePipeline](#featurepipeline)
- [StatisticsExtractor](#statisticsextractor)
- [PeriodExtractor](#periodextractor)
- [DmdtExtractor](#dmdtextractor)
- [CatalogExtractor](#catalogextractor)

---

## `FeatureExtractor` Protocol { #featureextractor }

The contract every extractor must satisfy. Extractors are called by `FeaturePipeline` — never directly.

**Consumes:** `list[list[LightCurve]]` — one list of bands per source

**Emits:** `list[dict[str, Any]]` — one dict per source mapping `FeatureVector` field names to values

```python
@runtime_checkable
class FeatureExtractor(Protocol):
    def extract(self, sources: list[list[LightCurve]]) -> list[dict[str, Any]]: ...
```

`extract()` is the **only required member**. Keys absent from a returned dict leave the
corresponding `FeatureVector` field at its default: `np.nan` for floats, `None` for
optionals, `""` for strings.

`extract` must **never raise**. On failure, return a list of empty dicts — one per input
source — rather than propagating the exception. The pipeline wraps every call in a
`try` anyway, since that contract is a request rather than a guarantee, and losing one
extractor should cost only its own features rather than the whole chunk. The pipeline
also checks that the returned list has exactly one entry per input source and discards
the extractor's output for that chunk if it does not, because a short list would
misalign features onto the wrong sources.

### `prepare()` is optional and is not on the Protocol

An extractor may define:

```python
def prepare(self, sources: list[list[LightCurve]]) -> None: ...
```

The pipeline calls it once with the **complete** per-band source list, before chunking,
so an extractor can derive any quantity that must be identical across chunks.
`PeriodExtractor` uses it to build the frequency grid from the full-field baseline;
without it the grid would be rebuilt per chunk and the same star would get a different
period depending on which sources happened to share its batch.

`prepare()` is deliberately **not declared on the Protocol**. `FeatureExtractor` is
`@runtime_checkable`, and a `runtime_checkable` Protocol's `isinstance()` check tests
for the presence of every declared member. Declaring an optional method there would make
every extractor that does not need it — including `StatisticsExtractor` and
`DmdtExtractor`, both shipped in this package — fail an `isinstance()` check against the
protocol they satisfy. The pipeline instead probes for it:

```python
prepare = getattr(extractor, "prepare", None)
if prepare is not None:
    prepare(per_band)
```

so `isinstance()` agrees with what the pipeline actually demands.

---

## Time precision — `to_float32_time` { #time-precision }

periodfind consumes float32 arrays. `LightCurve.time` is float64 and, for ZTF, holds
absolute HJD around 2.46e6. Casting that directly to float32 is silently destructive:
one float32 ULP at that magnitude is 0.25 days, so *every* pair of observations less
than six hours apart collapses to a separation of exactly zero. Intra-night structure —
the entire short-period regime — disappears with no error raised.

`features.base.to_float32_time()` subtracts the earliest finite epoch **in float64
first**, moving the values into `[0, baseline]` where the ULP scales with the baseline
rather than with the zero point:

| Baseline | float32 resolution |
|----------|--------------------|
| 1000 days | ~5 s |
| 1757 days (full ZTF DR16) | ~10.5 s |

That is not exact. Ten seconds is enough to move an intra-night pair between adjacent
Δt bins at the short end of the dm/dt grid. It is nonetheless the right trade: absolute
epoch is not something any feature here depends on — only differences matter — and the
alternative destroys the sub-day structure outright rather than nudging it.

`nanmin` is used for the offset because a plain `min()` propagates a single NaN across
the whole array, and `LightCurve` deliberately does not drop non-finite values, so one
bad epoch would otherwise blank every epoch. The offset is applied unconditionally: MJD
around 6e4 is less severe than HJD but still loses precision.

`PeriodExtractor._prepare()` performs the same float64-subtract-then-cast for its own
preprocessing, since it also normalises magnitudes in the same pass.

---

## `FeaturePipeline` { #featurepipeline }

Composes extractors in order and assembles the resulting dicts into `FeatureVector` objects. This is the entry point for the feature layer.

**Consumes:** `list[list[LightCurve]]` — sources grouped by band

**Emits:** `list[FeatureVector]` — one per (source, band); bands below `min_observations` return an all-NaN vector

```python
from ml4em.features import FeaturePipeline
from ml4em.config import load_config

pipeline = FeaturePipeline.default(load_config().features)
feature_vectors = pipeline.run_batch(grouped_lcs)
```

`FeaturePipeline.default()` builds the standard extractor chain in the order
statistics → period → dmdt → catalog, dropping `DmdtExtractor` when
`features.compute_dmdt` is false. It takes an optional `kowalski_client`; pass
`ztf_source.client` to enable live Gaia cross-matching. Without it every `gaia_*` field
stays `None`.

For a custom extractor set:

```python
pipeline = FeaturePipeline(
    extractors=[stats, period],
    min_observations=50,
    compute_dmdt=False,
    device="auto",
    batch_size=1000,
    checkpoint_dir=None,
)
```

### One row per (source, band)

Before anything else, `run_batch()` expands each source group into one single-band group
per `LightCurve`. A source with g and r light curves becomes two entries and yields two
`FeatureVector` rows sharing a `source_id`.

The alternative — searching only the longest band — discards roughly 40% of a typical
ZTF source's epochs and, more importantly, throws away the cross-band consistency check.
A period that reproduces in both g and r is far more likely to be astrophysical than one
that appears in a single band, where it may just be that band's cadence pattern.

The expansion happens before `prepare()` and before chunking, so the frequency grid and
the checkpoint source count both refer to the same expanded list.

### Device and batching

```yaml
features:
  device: auto             # "auto" | "cpu" | "gpu"  ("cuda" is accepted as an alias for "gpu")
  feature_batch_size: 1000
```

`device` controls whether periodfind uses CPU (Rust) or GPU (CUDA). `auto` delegates to
periodfind's own detection — a CUDA-extension import plus an `nvidia-smi` probe — and
falls back to CPU. `periodfind.set_device()` itself accepts only the literals `cpu` and
`gpu`, so `auto` is resolved and `cuda` normalised before the call. The device is
resolved once and set before the first batch.

`feature_batch_size` controls memory per periodfind call. Lower it if the GPU runs out
of memory on long light curves. It must be at least 1: chunking uses
`range(0, n, batch_size)`, so zero raises mid-run and a negative value produces no
chunks at all, returning nothing without reporting an error.

### Minimum observations

Bands with fewer than `min_observations` epochs (default: **50**) skip all extractors and
receive a default `FeatureVector` carrying only `source_id`, `survey`, `band`, `ra`,
`dec` and the real `n_obs`.

Individual extractors apply their own, lower floors on top of this — see
[StatisticsExtractor](#statisticsextractor) and [DmdtExtractor](#dmdtextractor). Those
floors are what the kernels require to return anything at all; `min_observations` is the
science threshold and is normally far above them.

### Checkpoint and resume

Set `features.checkpoint_dir` to a unique per-run path on scratch storage and the
pipeline writes a checkpoint after every chunk, restores completed `FeatureVector`s on
the next invocation with the same directory, and resumes from the next chunk. The file is
deleted on successful completion.

Writes go to a temporary file and are then renamed, which is atomic on POSIX, so a crash
during the write cannot leave a corrupt checkpoint.

A checkpoint is discarded and the run starts fresh if either the source count or the
**`FeatureVector` schema fingerprint** — the tuple of its dataclass field names — differs
from the current run. The schema check matters because unpickling a `FeatureVector`
saved before a field was added does not fail: the missing attribute resolves to the
dataclass default, so restored rows would silently carry `None` where later rows carry
real values.

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

Times go through [`to_float32_time`](#time-precision); magnitudes and errors are cast
directly to float32. `periodfind.BasicStats().calc(times, mags, errs)` is then called
once for the whole batch and returns an `(M, 22)` array. Column names are remapped from
`periodfind.BasicStats.STAT_NAMES` to `FeatureVector` field names via `_STAT_NAME_MAP`.

### Minimum of four observations

periodfind's Rust kernel (`rust/src/basicstats.rs`) returns an **all-NaN row for fewer
than four points**. The extractor mirrors that guard and skips any source whose primary
band has `n_obs < 4`.

The reason it cannot simply pass the row through is that `n_obs` is an `int` field on
`FeatureVector`, and `int(nan)` raises. Without the guard a single three-point light
curve would take down the statistics for its entire chunk.

For the same reason, a row that comes back non-finite despite passing the length check —
non-finite input magnitudes, for instance — is also skipped, detected by testing the `N`
column for finiteness before anything is cast.

Skipped sources get an empty dict, which is the documented signal that this one source
has no statistics; every statistics field stays at `np.nan`. If the kernel itself raises,
every source in the call gets an empty dict, because the call is batched and there is no
way to attribute the failure to one entry.

!!! note "Four here, two in DmdtExtractor"
    The two floors differ on purpose. They are not a shared policy — each one is the
    minimum its own kernel needs. `DmdtExtractor` needs two points because two points
    make one pair; `BasicStats` needs four before its higher-moment statistics are
    defined.

---

## `PeriodExtractor` { #periodextractor }

Runs several period-finding algorithms, scores their agreement, and Fourier-decomposes
at the period that survives.

**Consumes:** Primary band light curve per entry (the band with the most epochs, though
the pipeline normally passes single-band groups)

**Emits:** `period`, `period_significance`, `period_algorithm`, the `period_top` /
`significance_top` candidate dicts, the agreement and sidereal-family scores, and 14
Fourier fields (`f1_power`, `f1_bic`, `f1_a`, `f1_b`, `f1_amp`, `f1_phi0`,
`f1_relamp1–4`, `f1_relphi1–4`). See [Data Contracts](../data-contracts.md) for the full
field list.

```python
from ml4em.features.period import PeriodExtractor
from ml4em.config import load_config

extractor = PeriodExtractor(load_config().features.period)
extractor.prepare(all_grouped_lcs)         # builds the frequency grid — see below
results = extractor.extract(grouped_lcs)
```

Configure via `config.yaml`:

```yaml
features:
  period:
    algorithms: [CE, AOV, LS, MHF, FPW]   # default
    min_period_days: 0.003472             # 5 min — matches --max-freq 288
    max_period_days: 10.0
    top_n_periods: 10
    min_agreement: 2
    samples_per_peak: 10.0
    n_chunks_multiplier: 3
```

`min_period_days` must be strictly below `max_period_days`; an inverted range would give
`f_min > f_max` and an empty frequency grid, surfacing far downstream as "no periods
found" rather than as a config error, so `PeriodConfig` rejects it at load time.

### Supported algorithms

| Key | periodfind class | Parameters used | Direction |
|-----|------------------|-----------------|-----------|
| `CE` | `ConditionalEntropy` | `n_phase=20, n_mag=10` | minimised |
| `AOV` | `AOV` | `n_phase=20` | maximised |
| `LS` | `LombScargle` | — | maximised |
| `MHF` | `MultiHarmonicFourier` | `max_harmonics=3` | maximised |
| `FPW` | `FPW` | `n_bins=20` | maximised |
| `BLS` | `BoxLeastSquares` | `n_bins=50, qmin=0.01, qmax=0.5` | maximised |

The default set is `[CE, AOV, LS, MHF, FPW]`. `BLS` is available but off by default; it
is the right choice for flat-bottomed eclipses.

The minimise/maximise direction is read off each `Periodogram` object rather than
hardcoded, so adding an algorithm does not require editing the scoring code. It matters
for the significance score: scoring a minimised statistic with `|value − mean|` would
rate a strongly *bad* trial exactly as highly as a strongly good one, so the z-score is
sign-flipped for minimised statistics. That also makes `significance` comparable between
algorithms, which the consensus scorer relies on when it falls back to the
highest-significance peak.

A legacy `E`-prefixed spelling (`ECE` for `CE`) is stripped by the config validator.

### The frequency grid is built once, in `prepare()`

```
f_min = max(2 / baseline, 1 / max_period_days)
f_max = 1 / min_period_days
df    = 1 / (samples_per_peak * baseline)
```

`df` is the intrinsic frequency resolution `1/baseline` oversampled by
`samples_per_peak`, so a real peak cannot fall between grid points.

The `2 / baseline` floor requires at least two full cycles inside the observing window;
one cycle is not a detection, it is a trend. On a multi-year ZTF baseline that floor
alone would admit periods of many hundreds of days, so `max_period_days` is applied on
top of it — without that the config knob would have no effect at all.

`baseline` is the **longest span in the whole field**, computed by `prepare()` over the
complete source list before chunking. A per-chunk baseline would give the same star a
different `df`, and therefore a different period, depending on which sources happened to
share its batch. That breaks reproducibility and makes checkpoint/resume
non-deterministic.

If `extract()` is called without `prepare()` — which happens for single-batch callers
such as tests and `benchmarks/single_latency.py` — the grid is built from that batch's
baseline and a warning is logged saying periods will not be comparable across batches.

### Peak extraction

A real peak occupies many adjacent grid points, so taking the `top_n_periods` best bins
would return `top_n_periods` samples of the *same* peak. The periodogram is instead split
into `n_chunks_multiplier * top_n_periods` contiguous chunks, the best point in each chunk
is taken, and those winners are ranked.

The grid runs to roughly 10⁶ points, so one full periodogram is several MB per source and
a 1000-source batch would return multiple GB to host memory. Sources are fed to
periodfind in sub-batches sized to stay under a 2 GiB budget; each sub-batch is reduced to
its top-N peaks and discarded before the next is requested.

### Why a consensus step exists

A periodogram's tallest peak is frequently not the true period, for two reasons that no
better algorithm fixes:

**Harmonics.** An eclipsing binary with two similar eclipses per orbit folds equally
cleanly at P and at P/2, so noise decides the ranking between them.

**Sidereal aliases.** Ground-based sampling repeats on the sidereal day, so a true
frequency `f` is indistinguishable from `f ± n·f_sidereal`
(`SIDEREAL_FREQ_PER_DAY ≈ 1.00274` cycles/day). The periodogram shows a comb of near-equal
peaks.

The information to break these ties is not present in a single periodogram. What does
help is asking several algorithms with different statistics which *family* of aliases
they collectively favour. The resulting scores are themselves useful ML features, which
is why they are stored rather than discarded after picking a period.

### Agreement and family scoring

Two peaks agree when they match within a **5% fractional tolerance** at one of the
harmonic ratios 1:1, 1:2, 2:1, 1:3 or 3:1. Pairwise agreement across algorithms produces
`period_agree_score`, `period_agree_strict` (1:1 only), `period_agree_weighted` (weighted
by peak rank), `period_best_agree` and `period_best_consensus`.

Sidereal-family grouping treats all top-N peaks from all algorithms as nodes, links peaks
related by an alias `f ± n·f_sidereal` (n up to 15, 3% tolerance on the integer residual),
and merges linked peaks into families. The family spanning the most distinct algorithms
wins, giving `period_family_best` and the surrounding `period_family_*` fields. Family
grouping uses the narrower harmonic set 1:1, 1:2, 2:1, because the alias search already
covers a wide frequency span and admitting 1:3 and 3:1 as well would merge families that
are not physically related.

Periods shorter than ~10 minutes are treated as spurious during consensus scoring.

### Fourier decomposition

`periodfind.FourierDecomposition().calc()` runs twice: once at each algorithm's top
period, to give the family scorer an `f1_power` to choose between algorithms with, and
once more at the finally chosen period, whose 14 outputs are unpacked into the
`f1_*` fields:

```
[power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
```

Entries with fewer than four epochs are skipped and keep the all-NaN period result.

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

Δt bin edges (log-spaced) and Δmag bin edges (linear) are built once at construction and
reused. Times go through [`to_float32_time`](#time-precision).

A source needs **at least 2 observations**, since two points make one pair. Shorter
entries are skipped and get an empty dict, leaving `dmdt` at `None`. This floor is lower
than `StatisticsExtractor`'s four on purpose — each extractor's floor is the minimum its
own kernel needs, not a shared policy.

`DmdtConfig` holds the two axes as explicit edge lists, `dt_edges` and `dm_edges`,
rather than a min/max pair — the reference binning is non-uniform and no spacing law
reproduces it. It rejects edges that are not strictly increasing (duplicated or
descending edges give zero- or negative-width bins, which `periodfind.DmDt` does not
itself reject) and rejects a negative first `dt_edge`, since Δt is never negative.

To skip this extractor:

```yaml
features:
  compute_dmdt: false
```

That also drops `DmdtExtractor` from `FeaturePipeline.default()` entirely, avoiding the
O(N²) pairwise computation.

See [The dm/dt Histogram](../background/dmdt.md) for a full explanation.

---

## `CatalogExtractor` { #catalogextractor }

Cross-matches each source against Gaia EDR3 within `XMATCH_RADIUS_ARCSEC` (2 arcsec) and
returns 7 astrometric and photometric features.

**Consumes:** `(ra, dec)` from each source's `LightCurve`

**Emits:** `gaia_parallax`, `gaia_parallax_error`, `gaia_g_mean_mag`,
`gaia_bp_mean_mag`, `gaia_rp_mean_mag`, `gaia_bp_rp`, `gaia_astrometric_excess_noise`

These help distinguish white dwarf systems: high parallax means nearby, a blue BP−RP
means a hot atmosphere, and low `astrometric_excess_noise` means a clean single-source
astrometric fit. `bp_rp` is derived as `phot_bp_mean_mag − phot_rp_mean_mag`; Gaia does
not publish it as a field.

**Backend.** A live authenticated penquins Kowalski client, passed at construction time.
`FeaturePipeline.default()` accepts an optional `kowalski_client` and forwards it here:

```python
pipeline = FeaturePipeline.default(cfg.features, kowalski_client=ztf_source.client)
```

With no client, or with `catalog.include_gaia: false`, every Gaia field is left at its
`FeatureVector` default of `None`. Note that `None` is what "no counterpart" means as
well — the extractor does not distinguish "not looked up" from "looked up and not found",
and `features_to_array()` maps both to NaN, which is what missing data means to a model.

All positions in a batch go out in one multi-position cone search, so a batch costs one
network round trip. When several Gaia sources fall inside the radius, the nearest by
angular separation wins.

```yaml
features:
  catalog:
    xmatch_radius_arcsec: 2.0
    include_gaia: true
    n_workers: 1        # 8-16 on MSI; 1 for local/single-source runs
```

See [Gaia & Stellar Catalogs](../background/gaia.md) for what the fields mean.

---

[← Data](data.md){ .md-button } [Models →](models.md){ .md-button .md-button--primary }
