# Guide: Add a New Data Source

Adding a new survey or data source requires **one new file**. No existing files need
to change (except optionally exporting from `data/__init__.py` for convenience).

---

## Step 1 — Create the file

```
src/ml4em/data/my_source.py
```

## Step 2 — Implement the two methods

Your class must have exactly two methods with these signatures:

```python
from ml4em.types import LightCurve

class MySource:
    def fetch(self, source_id: str) -> list[LightCurve]:
        """Return all bands for one source."""
        ...

    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        """Return all bands for all sources in one call."""
        ...
```

Both methods return a **flat** `list[LightCurve]` — not grouped by source. If one
source has 3 bands, the list has 3 entries. The feature layer groups them by
`source_id` internally.

## Step 3 — Build LightCurve objects

Each `LightCurve` must have:
- `time`: timestamps in **days** (MJD or HJD — pick one and be consistent)
- `mag`: apparent magnitude (smaller = brighter)
- `mag_err`: 1-sigma uncertainty, same units as `mag`
- `band`: one of `"u"`, `"g"`, `"r"`, `"i"`, `"z"`, `"y"`
- `survey`: a string identifying your survey (e.g. `"my_survey"`)
- `ra`, `dec`: sky position in decimal degrees

```python
import numpy as np
from ml4em.types import LightCurve

class MySource:
    def fetch(self, source_id: str) -> list[LightCurve]:
        raw = my_api.get(source_id)

        return [LightCurve(
            source_id=source_id,
            time=np.array(raw["mjd"], dtype=np.float64),
            mag=np.array(raw["mag"], dtype=np.float64),
            mag_err=np.array(raw["magerr"], dtype=np.float64),
            band=raw["band"],          # "g", "r", etc.
            survey="my_survey",
            ra=float(raw["ra"]),
            dec=float(raw["dec"]),
        )]

    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        # Make one batched API call instead of N individual calls
        results = my_api.batch_get(source_ids)
        lcs = []
        for row in results:
            lc = LightCurve(
                source_id=str(row["id"]),
                time=np.array(row["mjd"]),
                mag=np.array(row["mag"]),
                mag_err=np.array(row["magerr"]),
                band=row["band"],
                survey="my_survey",
                ra=float(row["ra"]),
                dec=float(row["dec"]),
            )
            lcs.append(lc)
        return lcs
```

## Step 4 — Optional: export for convenience

Add to `src/ml4em/data/__init__.py`:

```python
from .my_source import MySource
__all__ = [..., "MySource"]
```

## Step 5 — Use it

```python
from ml4em.data.my_source import MySource
from ml4em.features import FeaturePipeline
from ml4em.config import load_config

cfg = load_config()
source = MySource(...)
pipeline = FeaturePipeline.default(cfg.features)

lcs = source.fetch_batch(["id1", "id2", "id3"])
# group by source_id
from itertools import groupby
grouped = [[lc for lc in g] for _, g in groupby(lcs, key=lambda l: l.source_id)]
fvs = pipeline.run_batch(grouped)
```

---

## Common pitfalls

**Wrong array dtype:** `LightCurve.__post_init__` validates that `time`, `mag`, and
`mag_err` are all 1-dimensional and the same length. If your arrays are 2D (e.g. from
a pandas DataFrame without `.values.flatten()`), you'll get a validation error.

**Flux instead of magnitude:** Many surveys return flux in Janskies or nJy, not
magnitudes. Convert before building `LightCurve`:
```python
mag = -2.5 * np.log10(flux_njy / 3631e9)   # AB magnitude
```

**HJD vs MJD:** Both are fine as long as you're consistent. ZTF uses HJD, Rubin uses
MJD. The feature extractors treat `time` as just a float array of days.
