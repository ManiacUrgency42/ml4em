# Light Curves

A light curve is the most fundamental data structure in ml4em. Everything — feature
extraction, period finding, training, inference — operates on light curves.

---

## What is a light curve?

A light curve is a **time series of how bright a star (or other source) appears**,
sampled at irregular intervals.

Think of it as a sensor log: each row is a timestamp and a brightness reading. Over
months or years, a telescope visits the same patch of sky repeatedly and records a new
measurement each time. The full collection of those measurements for one source is its
light curve.

```
time (days)   magnitude   uncertainty
──────────    ─────────   ───────────
59000.1       18.42       0.03
59002.8       18.39       0.04
59007.3       18.51       0.03
59010.0       17.95       0.05   ← brief brightening
59014.6       18.44       0.03
...
```

The x-axis is time. The y-axis is brightness — but brightness in astronomy is measured
in a unit called **magnitude**, which has some surprising properties.

---

## Magnitude — the backwards brightness scale

Astronomers measure brightness in **magnitudes** (mag). This is a logarithmic scale
that runs **backwards**: a higher magnitude number means a *dimmer* source.

The math:

```
m = -2.5 × log₁₀(flux) + constant
```

Practical consequences:

- A **magnitude 15** star is **100× brighter** than a magnitude 20 star
- The Sun is about magnitude −26.7. A faint galaxy might be magnitude 28.
- In ml4em: `LightCurve.mag` stores apparent magnitude. **Smaller values = brighter.**

!!! tip "Why does this matter for the code?"
    Several variability statistics (like `norm_peak_to_peak_amp` and `norm_excess_var`)
    are computed on the `mag` array. Because the scale is inverted, a *decrease* in `mag`
    corresponds to a *brightening* event (e.g., an eclipse ending).

**Apparent vs. absolute magnitude:**
Apparent magnitude is how bright a source looks from Earth — it depends on both the
source's intrinsic luminosity and its distance. Absolute magnitude is the intrinsic
brightness (standardized to a fixed distance of 10 parsecs). ml4em works exclusively
with apparent magnitudes — no distance correction is applied.

**Magnitude uncertainty:**
`LightCurve.mag_err` is the 1-sigma measurement uncertainty in the same magnitude units.
A value of 0.03 mag means the true brightness is expected to be within ±0.03 mag of
the reported value about 68% of the time.

---

## Time — MJD and HJD

Observations are timestamped using astronomical time systems. Both appear in the `time`
field of `LightCurve`.

### MJD — Modified Julian Date

MJD is simply a single decimal number counting **days since November 17, 1858** (noon).
There is nothing magical about this date — it was chosen by convention.

```
time[0] = 59000.0
```

This means the first observation was taken 59,000 days after the reference date —
roughly the year 2020. For most coding purposes, MJD is just a float timestamp with
"days" as the unit.

Rubin uses MJD (`expMidptMJD` in `dp1.Visit`).

### HJD — Heliocentric Julian Date

HJD is MJD corrected for Earth's position in its orbit around the Sun. As Earth moves
from one side of its orbit to the other (January to July), the light travel time to a
distant star changes by up to ±8 minutes. HJD removes that offset so that observations
from different times of year are comparable.

ZTF data arrives in HJD.

!!! note
    In ml4em, both MJD and HJD are stored in `LightCurve.time` — the conversion is
    handled transparently in the data layer (`ZTFSource`, `RubinSource`). The feature
    extractors don't know which system was used; they only see the numeric timestamps.

---

## Photometric bands (filters)

A **band** (also called a filter) is a wavelength window — only light within a specific
color range is allowed through. Taking a light curve "in the g band" means measuring
brightness only in green light.

Standard band letters (from ultraviolet to near-infrared):

| Band | Wavelength center | Color | Used by |
|------|------------------|-------|---------|
| `u`  | ~355 nm | Ultraviolet | Rubin |
| `g`  | ~475 nm | Green | ZTF, Rubin |
| `r`  | ~622 nm | Red | ZTF, Rubin |
| `i`  | ~763 nm | Near-infrared | ZTF, Rubin |
| `z`  | ~905 nm | Near-infrared | Rubin |
| `y`  | ~990 nm | Near-infrared | Rubin |

**In ml4em:** each band is a separate `LightCurve` object. One star observed in ZTF
g, r, and i produces **three** `LightCurve` objects. `LightCurve.band` records which
filter was used.

Why separate objects per band? A star's brightness varies differently in each band for
different physical reasons. An eclipse in which the hotter (bluer) star is covered is
much more pronounced in the `u` band than in the `i` band. Keeping bands separate
preserves this information.

When multiple bands are available, the feature layer selects the **primary band** (the
one with the most observations) to compute statistics from.

---

## What makes a source "variable"?

A **constant** source has approximately the same magnitude at every observation —
variation is just measurement noise.

A **variable** source has genuine brightness changes:

| Type | Physical cause | Period? |
|------|---------------|---------|
| Eclipsing binary | One star passes in front of the other | Yes |
| White dwarf binary | Gravitational interaction, reflection, eclipses | Yes |
| Pulsating star (RR Lyrae, Cepheid) | Star physically expands and contracts | Yes |
| AGN (quasar) | Accretion disk variability | No (stochastic) |
| Nova / supernova | Explosive brightening | No (one-time) |
| Rotating star (spots) | Cooler star spots rotate into view | Yes |

ml4em classifies sources by comparing a source's features to a labeled training set.
The library is agnostic about *which* class you are looking for.

---

## Code connection

`LightCurve` is defined in `src/ml4em/types.py`:

```python
@dataclass
class LightCurve:
    source_id : str            # Survey-native identifier
    time      : np.ndarray     # Observation timestamps (MJD or HJD), shape (N,)
    mag       : np.ndarray     # Apparent magnitude, shape (N,)
    mag_err   : np.ndarray     # 1-sigma uncertainty, shape (N,)
    band      : Band           # "u" | "g" | "r" | "i" | "z" | "y"
    survey    : Survey         # "ztf" | "rubin" | "simulated"
    ra        : float          # Right ascension, decimal degrees (J2000)
    dec       : float          # Declination, decimal degrees (J2000)
```

`ra` and `dec` are sky coordinates — see the [Gaia page](gaia.md) for what J2000
and decimal degrees mean.
