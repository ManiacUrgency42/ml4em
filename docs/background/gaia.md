# Gaia & Stellar Catalogs

**Gaia** is a space mission by the European Space Agency (ESA) that has precisely
measured the positions, distances, and brightnesses of nearly 2 billion stars. ml4em
uses Gaia to add 4 features to every source's `FeatureVector` — features that provide
distance and color information unavailable from the time-series data alone.

---

## What is Gaia?

Gaia operates by measuring each star's position on the sky with extraordinary precision,
then repeating that measurement as Earth orbits the Sun. As Earth moves from one side
of its orbit to the other, nearby stars appear to shift slightly relative to the
background of more distant stars — this shift is called **parallax**.

The data release used by ml4em is **Gaia EDR3** (Early Data Release 3, published 2020).

---

## Catalog cross-match

To get Gaia information for a source in our light curve data (ZTF or Rubin), we look
up the Gaia source at the same sky position. This is called a **catalog cross-match**.

The procedure:
1. Take the source's (ra, dec) from `LightCurve`
2. Search the Gaia EDR3 catalog for any Gaia source within 2 arcseconds of that position
   (`XMATCH_RADIUS_ARCSEC = 2.0` in `constants.py`)
3. If a match is found, return its 4 features. If not, return NaN.

**How small is 2 arcseconds?**
1 arcsecond = 1/3600 of a degree. 2 arcseconds is roughly the angular diameter of a
1 mm ball seen from 100 meters away. This search radius is small enough to be
unambiguous for most isolated stars, while large enough to account for small position
offsets between the Gaia catalog and the ZTF/Rubin positions.

!!! note "Status"
    `CatalogExtractor` is currently a stub. These 4 fields are all `NaN` until it is
    implemented. `XGBoostClassifier` will use them once they are populated — XGBoost
    handles NaN natively.

---

## The 4 Gaia features

### gaia_parallax — distance indicator

**What it is:** The apparent shift of a star's position in the sky, measured in
milliarcseconds (mas), caused by Earth's orbital motion around the Sun.

**The key formula:**
```
parallax [mas] = 1000 / distance [parsecs]
```

Larger parallax = closer star. A star at 100 parsecs (about 326 light-years) has a
parallax of 10 mas.

**Why it matters for WDB detection:** White dwarfs are stellar remnants — the collapsed
cores of dead stars. They are intrinsically faint objects. For us to detect them
photometrically (bright enough to be in ZTF or Rubin), they must be relatively nearby.
Nearby stars have large parallaxes. A large `gaia_parallax` is therefore a prior
indicator of a white dwarf candidate.

### gaia_parallax_error — reliability of the parallax

The 1-sigma measurement uncertainty on the parallax, in milliarcseconds.

A useful derived signal-to-noise ratio:
```
parallax SNR = gaia_parallax / gaia_parallax_error
```

Parallax measurements with SNR < 5 are generally unreliable. A high uncertainty doesn't
mean the star is far — it means Gaia couldn't measure its distance precisely.

### gaia_bp_rp — colour (temperature proxy)

**What it is:** The difference between Gaia's blue photometer (BP) and red photometer
(RP) magnitudes.

```
BP–RP = magnitude(blue band) - magnitude(red band)
```

Because this is a magnitude difference (not a flux ratio), a smaller (even negative)
BP–RP means the source is **bluer** relative to a larger (redder) BP–RP.

| BP–RP value | Color | Temperature (approx.) |
|-------------|-------|----------------------|
| < 0 | Very blue | > 10,000 K |
| 0 – 0.6 | Blue-white | 7,000 – 10,000 K (A/F type) |
| 0.6 – 1.5 | Yellow-white | 4,500 – 7,000 K (G/K type, like the Sun) |
| > 1.5 | Red | < 4,500 K (M type) |

**Why it matters for WDB detection:** White dwarfs are the exposed hot cores of dead
stars — they are typically blue (low BP–RP). A blue source with a large parallax is
a strong white dwarf candidate. The companion in a WDB system can be a red M-dwarf,
causing intermediate BP–RP values.

!!! tip "British spelling"
    "Colour" (with a 'u') is the standard spelling in the astronomical literature.
    You'll see it in Gaia documentation and papers. In ml4em the variable is named
    `gaia_bp_rp` for brevity.

### gaia_ruwe — astrometric quality flag

**RUWE** = Renormalized Unit Weight Error. This is a dimensionless quality metric for
Gaia's astrometric fit.

Gaia fits a **single-star astrometric model** to each source — assuming it is one
point of light moving predictably across the sky (due to parallax and proper motion).
RUWE measures how well this model fits the actual observations.

| RUWE value | Interpretation |
|------------|---------------|
| ≈ 1.0 | Good fit — source behaves like a single star |
| 1.0 – 1.4 | Acceptable |
| > 1.4 | Poor fit — the source doesn't move like a single star |

**What causes high RUWE?**

- **Unresolved binary:** Two stars too close to resolve. The combined light wobbles
  as they orbit each other, causing the apparent position to deviate from a single-star
  path. This is the most common cause in our science case.
- **Extended source:** A galaxy or nebula rather than a point star.
- **Crowded field:** Another star so close that its light contaminates the measurement.

**Why RUWE > 1.4 is useful for WDB detection:** White dwarf binaries have two objects
(the WD and its companion) orbiting each other. If their separation is small enough,
Gaia sees one blended source — and the orbital motion causes exactly the kind of
astrometric wobble that elevates RUWE. `GAIA_RUWE_CLEAN = 1.4` is stored in
`constants.py` as the accepted clean threshold.

---

## Sky coordinates — ra and dec

Sky positions are given in:
- **ra** — right ascension: the celestial equivalent of longitude, in decimal degrees.
  Ranges from 0 to 360.
- **dec** — declination: the celestial equivalent of latitude, in decimal degrees.
  Ranges from −90 (south pole) to +90 (north pole).
- **J2000**: the coordinate system is fixed to Earth's orientation on January 1, 2000.

These appear in both `LightCurve` and `FeatureVector`.

---

## Code connection

- Implementation: `src/ml4em/features/catalog.py:CatalogExtractor` (stub)
- Constants: `XMATCH_RADIUS_ARCSEC = 2.0`, `GAIA_RUWE_CLEAN = 1.4` in
  `src/ml4em/constants.py`
- Output fields: `FeatureVector.gaia_parallax`, `.gaia_parallax_error`,
  `.gaia_bp_rp`, `.gaia_ruwe`
- All four are in `models.SCALAR_FIELDS` — once populated, `XGBoostClassifier`
  will use them automatically
