# The dm/dt Histogram

The dm/dt histogram is a **2D image feature** that captures how a source's brightness
changes across different timescales. Unlike the scalar statistics (which collapse the
entire light curve into single numbers), the dm/dt histogram preserves the relationship
between time separation and brightness change.

---

## The basic idea

For a light curve with N observations, consider every possible **pair** of observations
(i, j) where j > i. There are N×(N−1)/2 such pairs.

For each pair, compute:

```
Δt   = |t_j - t_i|        (time separation, in days)
Δmag = mag_j - mag_i       (magnitude change)
```

This gives you a cloud of (Δt, Δmag) points in a 2D space.

**Example for 5 observations:** that's 10 pairs → 10 points in the (Δt, Δmag) plane.

For a real light curve with 200 observations, that's ~20,000 pairs.

---

## The 26×26 histogram

Bin the (Δt, Δmag) point cloud into a 2D histogram:

```
         Δt axis (columns): 26 log-spaced bins from 0.001 to 1000 days
         Δmag axis (rows):  26 linearly-spaced bins from -3.0 to +3.0 mag
```

Each cell in the resulting 26×26 grid counts how many observation pairs fall in that
(Δt, Δmag) bin.

### Why log-spaced for Δt?

Astronomical timescales span many orders of magnitude:

- Intra-night observations: Δt ~ 0.01 days
- Night-to-night cadence: Δt ~ 1–10 days
- Seasonal: Δt ~ 100 days
- Multi-year: Δt ~ 1000 days

A linear axis would give almost all bins to the long-timescale regime and almost none
to the short-timescale regime. A log axis distributes resolution evenly across all
timescales.

### Why linear for Δmag?

Magnitude differences are bounded. Light curves in these surveys typically have total
amplitudes under 3 magnitudes. A linear axis from –3 to +3 mag covers essentially all
physical cases.

---

## L2 normalization

After building the histogram, divide every cell by the histogram's L2 norm:

```
histogram_normalized = histogram / sqrt(sum(histogram²))
```

This makes light curves with different numbers of observations (and therefore different
numbers of pairs) produce comparable histograms. Without normalization, a source with
1000 observations would have a much larger histogram than one with 100 observations.

---

## What the image encodes

Different types of sources produce recognizable patterns in the dm/dt histogram:

**Constant source:**
The cloud is concentrated near Δmag ≈ 0 at all timescales. All pairs have nearly
identical brightness at both observations.

```
Δmag ^
 +3  |  . . . . . . . . . . . . . . . . . . . . . . . . . .
 +2  |  . . . . . . . . . . . . . . . . . . . . . . . . . .
 +1  |  . . . . . . . . . . . . . . . . . . . . . . . . . .
  0  |  ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■  ← all weight here
 -1  |  . . . . . . . . . . . . . . . . . . . . . . . . . .
     +--------------------------------------------------->
       0.001                                          1000  Δt (days)
```

**Eclipsing binary with period P:**
At timescales near P/2 (half the period), one observation catches the source at
baseline and the other catches it in eclipse — producing large |Δmag|. At timescales
near P, both observations are at the same phase → Δmag ≈ 0 again.

This creates a distinctive "stripe" pattern at Δt ≈ P/2.

**Long-period variable:**
Weight concentrated at large Δt values (the source changes significantly only over
many days).

**Stochastic AGN:**
Weight spread broadly across all timescales and |Δmag| values — no clear structure.

---

## How the model uses it

The `dmdt` field in `FeatureVector` is the 26×26 ndarray (dtype float32).

A **convolutional neural network (CNN)** treats the histogram as an image and learns
to recognize spatial patterns associated with each source class. The spatial structure
(where in the Δt × Δmag plane the weight is concentrated) encodes information about
both the amplitude and the timescale of variability.

The reference model (`XGBoostClassifier`) **ignores** the `dmdt` field — it is a
gradient-boosted tree that operates only on the 43 scalar fields listed in
`models.SCALAR_FIELDS`. A future CNN or hybrid model can use the image.

To skip the dm/dt computation entirely (e.g. for scalar-only experiments):

```yaml
features:
  compute_dmdt: false
```

This saves significant computation time when the model doesn't use it.

---

## Code connection

- Configuration: `config.yaml` → `features.dmdt.*` → `DmdtConfig` in
  `src/ml4em/config/schema.py`
- Implementation: `src/ml4em/features/dmdt.py:DmdtExtractor`
- Backend: `periodfind.DmDt` (Rust implementation, same library as everything else)
- Bin edges (Δt and Δmag) are pre-built once in `DmdtExtractor.__init__()` using the
  config parameters and reused for all `extract()` calls
- Constants: `N_DT_BINS = 26`, `N_DM_BINS = 26`, `DMDT_DT_MIN/MAX`,
  `DMDT_DM_MIN/MAX` in `src/ml4em/constants.py`
