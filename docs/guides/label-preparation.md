# Guide: Preparing Labels

ml4em does not generate labels. The library classifies sources using a trained model,
and that model requires labeled training data — you must supply it.

---

## The labels.csv format

```csv
source_id,label
686149073900013696,1
686149073900013697,0
686149073900013698,0
686149073900013699,1
```

- **`source_id`**: the survey-native identifier, same string as in `LightCurve.source_id`
- **`label`**: `1` = positive class (what you're looking for), `0` = background

Labels must be binary integers. Any source present in only one of (features file,
labels CSV) is silently skipped when building `FeatureDataset`.

---

## Where labels come from

Labels must come from an **external authoritative source** — typically a catalog
cross-match. You ask: "which of the sources in my light curve sample are *known* to be
members of my target class?"

**Common label sources for WDB detection:**

| Catalog | What it contains | How to access |
|---------|-----------------|---------------|
| [Gaia WD catalog](https://www.gaia-eso.eu) | ~360,000 high-confidence WD candidates from Gaia photometry | Public, downloadable |
| SDSS spectroscopic WD catalog | WDs confirmed by spectroscopy | Public, downloadable |
| Montreal WD database | Curated WD catalog | Public |

**For other science cases:**

- AGN: use spectroscopic AGN catalogs (SDSS, 2QZ, etc.)
- RR Lyrae: use Gaia SOS variable star catalog or PS1 RR Lyrae catalogs
- Eclipsing binaries: use OGLE, Kepler/K2, or ATLAS EB catalogs

---

## Building labels.csv from a catalog

The general workflow:

1. Download the target catalog (e.g. Gaia WD list as a CSV or FITS table)
2. Download your survey's source list (the sources you have light curves for)
3. Cross-match: for each source in your survey list, check if there is a matching
   catalog entry within some angular radius
4. Assign label `1` to matched sources, `0` to the rest

### Example with astropy

```python
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# Your survey sources with sky positions
survey = pd.read_csv("ztf_sources.csv")   # columns: source_id, ra, dec

# Your target catalog (e.g. Gaia WD candidates)
catalog = pd.read_csv("gaia_wd_catalog.csv")   # columns: ra, dec

# Build SkyCoord objects
survey_coords = SkyCoord(ra=survey["ra"].values * u.deg,
                         dec=survey["dec"].values * u.deg)
catalog_coords = SkyCoord(ra=catalog["ra"].values * u.deg,
                          dec=catalog["dec"].values * u.deg)

# Cross-match: find nearest catalog source for each survey source
idx, sep, _ = survey_coords.match_to_catalog_sky(catalog_coords)

# Label as positive if within 2 arcseconds
MATCH_RADIUS = 2.0 * u.arcsec
labels = (sep < MATCH_RADIUS).astype(int)

# Write the result
result = pd.DataFrame({
    "source_id": survey["source_id"].astype(str),
    "label": labels,
})
result.to_csv("labels.csv", index=False)
```

---

## Class imbalance

Rare astrophysical objects are genuinely rare. Expect roughly:

| Science case | Typical positive fraction |
|-------------|--------------------------|
| White dwarf binaries | ~0.1–2% of all sources |
| AGN | ~0.5–5% of sources (field-dependent) |
| RR Lyrae | ~0.01–0.1% |

Check your imbalance before training:

```python
dataset = FeatureDataset.from_storage(cfg.storage, labels_path="labels.csv")
print(dataset.class_counts())       # {0: 9850, 1: 150}
print(dataset.positive_fraction())  # 0.015  → 1.5% positive
```

A model trained on severely imbalanced data will tend to predict "background" for
almost everything (and achieve high accuracy while being useless). Mitigation strategies:

- **XGBoost:** `scale_pos_weight = n_negative / n_positive`
- **Neural nets:** weighted loss function (`pos_weight` in BCEWithLogitsLoss)
- **Oversampling:** duplicate positive samples in the training set
- **Undersampling:** randomly drop background samples

ml4em does not apply any of these automatically — the choice is left to the model
implementation.

---

## Validation

After preparing `labels.csv`, verify it before training:

```python
import pandas as pd

labels = pd.read_csv("labels.csv")
print(labels.dtypes)                         # source_id: object, label: int64
print(labels["label"].value_counts())        # 0: 9850, 1: 150
print(labels["source_id"].duplicated().any())  # False — no duplicate IDs
```
