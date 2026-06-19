# Variability Statistics

The `StatisticsExtractor` computes 22 scalar numbers from each light curve. These
numbers compress the entire brightness history of a source into a fixed-length feature
vector. Together, they tell the model whether a source is variable (and how), without
requiring explicit period information.

All 22 statistics are computed by the Rust-backed `periodfind.BasicStats` library in
a single batched call. They are computed on the **primary band** — the band with the
most observations for that source.

!!! info "Units"
    All statistics that have magnitude units are in the same units as `LightCurve.mag`
    (apparent magnitude). Remember: smaller = brighter.

---

## The 22 statistics

### n_obs — number of observations

How many data points are in the light curve.

Sources with fewer than `min_observations` (default 50, set in `config.yaml`) receive
an all-NaN feature vector and are skipped.

---

### median — median apparent magnitude

The 50th-percentile brightness. More robust than the mean because a single very bright
or very faint outlier (e.g. a cosmic ray that slipped through) doesn't shift it much.

In magnitude units: smaller = brighter.

---

### wmean — error-weighted mean magnitude

The mean brightness, weighted so that observations with smaller `mag_err` count more:

```
wmean = Σ(mag_i / mag_err_i²) / Σ(1 / mag_err_i²)
```

A single high-uncertainty observation barely shifts this, while a precise observation
counts for more. This is the standard way to combine measurements of different quality.

---

### chi2red — reduced chi-squared (χ²/dof)

Measures how much the light curve varies **relative to what measurement noise alone
would predict**.

```
chi2red = [ Σ((mag_i - wmean)² / mag_err_i²) ] / (n_obs - 1)
```

| chi2red | Interpretation |
|---------|---------------|
| ≈ 1 | The source is constant — variation is consistent with noise |
| >> 1 | The source is genuinely variable |
| < 1 | The error bars may be overestimated |

This is one of the most important variability discriminators. A constant star has
chi2red ≈ 1; an eclipsing binary might have chi2red of 50 or more.

---

### roms — ratio of median scatter to sigma

```
roms = median( |mag_i - wmean| / mag_err_i )
```

A robust alternative to chi2red. Where chi2red squares the deviations (making it
sensitive to outliers), roms uses the median of absolute deviations. A few bad
observations don't dominate the result.

---

### wstd — error-weighted standard deviation

Standard deviation of the magnitudes, weighted by measurement precision:

```
wstd = √[ Σ(mag_i - wmean)² / mag_err_i² / Σ(1/mag_err_i²) ]
```

Measures the typical spread of the light curve around the weighted mean.

---

### norm_peak_to_peak_amp — normalized peak-to-peak amplitude

```
norm_peak_to_peak_amp = (max(mag) - min(mag)) / median(mag_err)
```

How large is the total brightness swing, measured in units of the typical measurement
noise? A source that varies by 0.5 mag with typical errors of 0.05 mag has
`norm_peak_to_peak_amp = 10`.

Large values indicate high-amplitude variability.

!!! warning "Sensitivity to outliers"
    Because this uses the maximum and minimum, a single bad observation can inflate it.
    Consider `norm_excess_var` or `iqr` for more robust amplitude estimates.

---

### norm_excess_var — normalized excess variance

```
norm_excess_var = (var(mag) - mean(mag_err²)) / wmean²
```

The variance in excess of what noise alone predicts, normalized by the mean brightness
squared. Subtracting `mean(mag_err²)` removes the noise floor contribution.

- Negative or near-zero → no excess variance; the source is consistent with being
  constant
- Large positive → genuine variability beyond noise

---

### median_abs_dev — median absolute deviation (MAD)

```
median_abs_dev = median( |mag_i - median(mag)| )
```

The typical deviation of an observation from the median brightness. Unlike standard
deviation, MAD is highly robust — a single extreme outlier changes it very little.

A constant source has `median_abs_dev` close to 0.674 × σ (for Gaussian noise, MAD
relates to standard deviation by this factor). Larger values indicate variability.

---

### iqr — interquartile range

```
iqr = 75th percentile(mag) - 25th percentile(mag)
```

The spread of the "middle 50%" of brightness values. Another robust spread measure.
For Gaussian noise, `iqr ≈ 1.35 × σ`.

---

### i60r, i70r, i80r, i90r — percentile range ratios

These four statistics measure what fraction of the total amplitude is captured by the
central 60%, 70%, 80%, and 90% of observations:

```
i90r = (90th percentile(mag) - 10th percentile(mag)) / (max(mag) - min(mag))
i80r = (80th percentile(mag) - 20th percentile(mag)) / (max(mag) - min(mag))
...
```

**Intuition for eclipsing sources:** An eclipsing binary spends most of its time at
baseline brightness, with only brief dips during eclipse. Therefore:

- Most observations cluster near the baseline → small range for the central 90%
- A few observations dip deeply during eclipse → the total range (max - min) is large
- Result: `i90r` is **small** for eclipsing sources

A source that varies continuously (like a pulsating star) has a more uniform
distribution of magnitudes → `i90r` closer to 1.0.

---

### skew — skewness

Skewness measures the asymmetry of the magnitude distribution.

- `skew < 0` (negative): the tail extends toward **bright** values (smaller magnitude).
  The star is usually faint, with occasional bright excursions.
- `skew > 0` (positive): the tail extends toward **faint** values (larger magnitude).
  The star is usually bright, with occasional dim excursions (e.g., eclipses).
- `skew ≈ 0`: symmetric distribution (e.g., a sine wave or Gaussian noise).

Eclipsing sources typically have `skew > 0` because most observations are at the bright
baseline and the eclipses pull the tail toward larger magnitudes.

---

### small_kurt — small-sample kurtosis

Kurtosis measures how "heavy-tailed" the brightness distribution is compared to a
Gaussian. The "small" qualifier refers to a bias correction for small sample sizes.

- `small_kurt ≈ 0` (excess kurtosis): the distribution has Gaussian-like tails
- `small_kurt > 0` (leptokurtic): heavier tails than a Gaussian — more extreme values
  than expected. Common in eclipsing sources (most values clustered near baseline,
  plus extreme dips during eclipses).
- `small_kurt < 0` (platykurtic): lighter tails — flatter, more uniform distribution.

---

### inv_von_neumann — inverse Von Neumann ratio (1/η)

The Von Neumann ratio η measures autocorrelation in the time series:

```
η = variance(successive differences) / variance(mag)
  = mean[ (mag_{i+1} - mag_i)² ] / variance(mag)
```

For truly random (white noise) data, successive differences have the same variance as
the data itself, so η ≈ 2.

For a periodically variable source, consecutive observations are correlated (if the
star is brightening now, it will likely still be bright at the next observation). This
reduces η below 2.

ml4em stores **1/η** (the inverse), so that:

- `inv_von_neumann ≈ 0.5` → random, constant source
- `inv_von_neumann >> 0.5` → strong temporal correlation, likely variable

This convention makes larger values correspond to *more* variability, which is more
intuitive for a classifier.

---

### stetson_i — Stetson I index

Named after Peter Stetson (1996). Measures correlated variability across pairs of
observations.

Positive values suggest the source is variable; values near zero suggest it is constant.
This index was originally designed for comparing simultaneous multi-band observations
but works on single-band time series as well.

---

### stetson_j — Stetson J index

A robust version of Stetson I that uses a sign-preserving weighting function to
down-weight extreme outliers. Less sensitive to individual bad observations.

Rule of thumb: `stetson_j > 0.5` is often used as a variability threshold in
literature.

---

### stetson_k — Stetson K index

Measures the kurtosis of the residual distribution (how the magnitudes deviate from
the weighted mean), in a way that is robust to outliers.

| stetson_k | Interpretation |
|-----------|---------------|
| ≈ 0.9 | Gaussian residuals → likely constant source |
| < 0.9 | Residuals more uniform than Gaussian (possibly periodic) |
| > 1.0 | Residuals more peaked than Gaussian |

---

### anderson_darling — Anderson-Darling normality test statistic

Tests whether the magnitude distribution is consistent with a Gaussian (normal)
distribution.

- Larger values → greater deviation from Gaussian → more likely to be variable
- A perfectly constant source with Gaussian noise has small Anderson-Darling values

The Anderson-Darling test gives extra weight to the tails of the distribution, making
it sensitive to extreme outliers (like eclipse dips).

---

### shapiro_wilk — Shapiro-Wilk normality test W statistic

Another test of whether the magnitude distribution is consistent with Gaussian noise.

- Values close to **1.0** → Gaussian-like → consistent with a constant source
- Values much less than **1.0** → non-Gaussian → likely variable

!!! note
    The field name is `shapiro_wilk`, but the stored value is the W test statistic
    (between 0 and 1), not the p-value. Lower W = less Gaussian = more likely variable.

---

## Code connection

- Implementation: `src/ml4em/features/statistics.py:StatisticsExtractor`
- Backend: `periodfind.BasicStats` (Rust, compiled into the Docker image)
- Field name mapping: `_STAT_NAME_MAP` in `statistics.py` maps periodfind column names
  (e.g. `"RoMS"`, `"WelchI"`) to `FeatureVector` field names (e.g. `"roms"`, `"stetson_i"`)
- All 22 fields are included in `models.SCALAR_FIELDS` and used by `XGBoostClassifier`
