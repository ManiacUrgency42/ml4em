# Period Finding

Many astronomical sources repeat the same brightness pattern on a regular cycle —
an eclipse every 3.7 hours, a pulsation every 12 days. Finding that period is one of
the most computationally intensive steps in the ml4em pipeline.

---

## What is a "period"?

A **period** is the length of one brightness cycle, measured in days.

| Period | Physical cause |
|--------|---------------|
| 0.05 days (72 min) | Short-period white dwarf binary |
| 0.1–0.5 days | Typical eclipsing binary |
| 0.5–1 day | Rapidly rotating star |
| 1–100 days | Long-period pulsating star (Cepheid, Mira) |

White dwarf binaries — the primary science case for this library — typically have
periods between a few minutes and a few hours (0.003–0.5 days).

---

## Phase folding — how you reveal a period

If you know the period P, you can **phase-fold** the light curve: instead of plotting
magnitude vs. time, plot magnitude vs. `time mod P` (the remainder after dividing by P).

If P is the true period, all observations at the same phase will have similar
magnitudes — the repeating pattern collapses into a clear shape. Random noise stays
scattered.

Phase folding is the standard visual check that confirms a period candidate is real.

---

## The challenge: unevenly sampled, noisy data

Finding periods in real astronomical data is hard:

1. Observations are at irregular times (not a clean grid)
2. Each measurement has noise (mag_err is not zero)
3. You don't know P in advance — you have to search a grid of candidates
4. The "true" period might not be the strongest signal (aliases can dominate)

The approach: for each candidate period in a **frequency grid**, evaluate a score
function that measures how periodic the data looks at that period. Plot all scores vs.
period — this is the **periodogram**. Peaks in the periodogram are period candidates.

---

## The frequency grid

Before searching, ml4em builds a grid of candidate frequencies (1/period values):

- `min_period_days` → `max_period_days`: the search range
- `n_freq_grid`: how many candidate frequencies to test

These are set in `config.yaml` under `features.period`:

```yaml
features:
  period:
    min_period_days: 0.003    # ~4 minutes
    max_period_days: 30.0
    n_freq_grid: 10000
```

More grid points → more sensitive to narrow period peaks, but slower.

---

## The six algorithms

ml4em supports six period-finding algorithms, all implemented in `periodfind` (a
GPU-accelerated Rust/CUDA library). Each algorithm independently scores every candidate
period in the grid and returns its top candidates.

### Lomb-Scargle (LS)

The most widely used period-finding algorithm in astronomy.

**How it works:** For each candidate frequency, fits a sine wave to the data and
measures how much of the total variance is explained by that fit. The ratio (power)
peaks at the true period.

**Strengths:** Fast, analytically understood, excellent for smooth sinusoidal signals.

**Weaknesses:** Assumes the signal is sinusoidal. Less sensitive to sharp features like
eclipses.

**When to use:** First-pass search; any approximately sinusoidal variability.

### Conditional Entropy (CE)

**How it works:** Phase-folds the data at each candidate period and bins the result
into a 2D grid (phase × magnitude bins). Measures the **Shannon entropy** of this grid
— a true period produces an ordered (low-entropy) pattern; random data stays
high-entropy.

`ConditionalEntropy(n_phase=20, n_mag=10)` uses 20 phase bins × 10 magnitude bins.

**Strengths:** Does not assume the shape of the signal — works for any periodic
waveform.

**Weaknesses:** Sensitivity depends on bin count; can miss very narrow features.

### Analysis of Variance (AOV)

**How it works:** Phase-folds the data at each candidate period and performs an
ANOVA F-test: asks whether the variance *between* phase bins is significantly larger
than the variance *within* phase bins. A true period produces large between-bin
variance (all points at the same phase cluster together).

`AOV(n_phase=20)` uses 20 phase bins.

**Strengths:** More robust than Lomb-Scargle for non-sinusoidal signals; well-studied
statistical properties.

### Multi-Harmonic Fourier (MHF)

**How it works:** Instead of fitting a single sine wave (like LS), fits a sum of
harmonics — the fundamental frequency plus its integer multiples (overtones).

`MultiHarmonicFourier(max_harmonics=5)` tries fits with 1–5 harmonics and picks the
best using BIC (see [BIC](#bic) below).

**Strengths:** Handles non-sinusoidal periodic signals (sharp eclipses, asymmetric
light curves).

### Box Least Squares (BLS)

**How it works:** Searches for a periodic "box" shape — a flat baseline with a
short, periodic dip. Fits a step function to the phase-folded light curve.

`BoxLeastSquares(n_bins=50)` uses 50 phase bins.

**Strengths:** Highly sensitive to transit/eclipse signals (short dip, long baseline).

**Weaknesses:** Less useful for other variability types.

### FPW — Fast Phase-Folding Weighted

**How it works:** A weighted phase-folding algorithm optimized for speed.

`FPW(n_bins=10)` uses 10 bins.

**Strengths:** Fast; useful for initial screening.

---

## Agreement scoring

Each algorithm independently finds its top period candidate. ml4em then asks: do
multiple algorithms agree?

Two algorithms "agree" if their periods are within **2% of each other**
(`_AGREE_TOL = 0.02` in `period.py`):

```
|P_A - P_B| / P_A < 0.02
```

The period confirmed by the most algorithms is reported in `FeatureVector.period`.
`FeatureVector.period_algorithm` records which algorithm(s) agreed.

If no two algorithms agree, the single algorithm with the highest significance wins.

**Why this matters:** Any one algorithm can be fooled by noise or aliases. Agreement
across algorithms with different mathematical assumptions is much stronger evidence of
a real period.

---

## Period significance

`FeatureVector.period_significance` records the algorithm's confidence in the period.
Higher is more confident, but the exact definition varies by algorithm:

- LS: the Lomb-Scargle power at the peak (normalized)
- CE: inverse entropy (lower entropy → higher significance)
- AOV: the F-test statistic

---

## Fourier decomposition — the 14 Fourier features

Once the best period is found, ml4em fits a Fourier model to the phase-folded light
curve at that period. This encodes the *shape* of the periodic signal (not just that
a period exists).

**The model:**

```
mag(t) = offset + slope·t + A₁cos(ωt) + B₁sin(ωt)
                           + A₂cos(2ωt) + B₂sin(2ωt)
                           + A₃cos(3ωt) + B₃sin(3ωt)
                           + A₄cos(4ωt) + B₄sin(4ωt)
                           + A₅cos(5ωt) + B₅sin(5ωt)
```

where `ω = 2π/P` (angular frequency).

### BIC — Bayesian Information Criterion { #bic }

MHF tries 0–5 harmonics and picks the number that minimizes BIC:

```
BIC = k·ln(N) - 2·ln(L)
```

where `k` = number of parameters, `N` = number of observations, `L` = likelihood.
BIC penalizes extra parameters, so it picks the simplest model that fits well.

### The 14 output features

| Field | Description |
|-------|-------------|
| `f1_power` | Power of the best-fit Fourier model |
| `f1_bic` | BIC score (lower = better fit per parameter) |
| `f1_a` | Cosine coefficient of the 1st harmonic |
| `f1_b` | Sine coefficient of the 1st harmonic |
| `f1_amp` | Amplitude of the 1st harmonic = √(A₁² + B₁²) |
| `f1_phi0` | Phase offset of the 1st harmonic |
| `f1_relamp1` | Amplitude of 2nd harmonic / amplitude of 1st |
| `f1_relphi1` | Phase of 2nd harmonic relative to 1st |
| `f1_relamp2` | Amplitude of 3rd harmonic / amplitude of 1st |
| `f1_relphi2` | Phase of 3rd harmonic relative to 1st |
| `f1_relamp3` | Amplitude of 4th harmonic / amplitude of 1st |
| `f1_relphi3` | Phase of 4th harmonic relative to 1st |
| `f1_relamp4` | Amplitude of 5th harmonic / amplitude of 1st |
| `f1_relphi4` | Phase of 5th harmonic relative to 1st |

**Relative amplitudes** (`f1_relamp1–4`) encode whether higher harmonics are important.
A pure sine wave has `f1_relamp1–4 ≈ 0`. A sharp eclipse has large higher-harmonic
content. These features give the model information about light curve shape.

---

## Aliases — false periods from cadence

A telescope with a regular observing cadence produces a spurious peak in the periodogram
at the cadence period and its harmonics. These are called **aliases**.

ZTF observes roughly every night, producing peaks near:

- 1.0 day (nightly cadence)
- 0.9973 days (sidereal day — the true rotation period of Earth relative to stars,
  stored as `ZTF_SIDEREAL_DAY` in `constants.py`)
- 0.5 day, 0.3333 day, ... (harmonics)

The `min_cadence_days` filter in `ZTFSource` (default 30 minutes) reduces intra-night
aliases but does not eliminate the ~1-day alias. The feature vector contains the raw
period without alias correction — alias rejection is left to the trained model.

---

## Code connection

- Configuration: `config.yaml` → `features.period.*` → `PeriodConfig` in
  `src/ml4em/config/schema.py`
- Implementation: `src/ml4em/features/period.py:PeriodExtractor`
- Backed by `periodfind` (GPU-accelerated, Rust/CUDA)
- Output fields in `FeatureVector`: `period`, `period_significance`, `period_algorithm`,
  and `f1_power` through `f1_relphi4`
