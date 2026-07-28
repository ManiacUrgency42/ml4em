# Guide: Add a New Data Source

Adding a new survey or data source requires **one new file**. No existing files need
to change (except optionally exporting from `data/__init__.py` for convenience).

---

## Step 1 — Create the file

```
src/ml4em/data/my_source.py
```

## Step 2 — Implement `fetch_batch()`

`LightCurveSource` requires exactly one method:

```python
from ml4em.types import LightCurve

class MySource:
    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        """Return all bands for all sources in one call."""
        ...
```

It returns a **flat** `list[LightCurve]` — not grouped by source. If one source has 3
bands, the list has 3 entries, and order is not guaranteed. Callers group by `source_id`
before handing the result to the feature layer.

There is no separate single-source method on the Protocol. For one source, pass a
one-element list: `fetch_batch([source_id])`. Adding your own convenience `fetch()` is
fine — `ZTFSource` has one — but nothing in the pipeline calls it.

Batch the underlying requests wherever the API supports it. The whole point of the
batch-first signature is that a run over 100k sources should not become 100k round trips.

## Step 3 — Build LightCurve objects

Each `LightCurve` must have:

- `time`: timestamps in **days**, **float64** — see the dtype note below
- `mag`: apparent magnitude (smaller = brighter)
- `mag_err`: 1-sigma uncertainty, same units as `mag`
- `band`: one of `"u"`, `"g"`, `"r"`, `"i"`, `"z"`, `"y"`
- `survey`: a string identifying your survey (e.g. `"my_survey"`)
- `ra`, `dec`: sky position in decimal degrees
- `time_system`: which time scale `time` is on — `"hjd"`, `"mjd"`, `"bjd"`, `"jd"`, or
  `"relative"`. Defaults to `"hjd"`, so set it explicitly if that is not what you are
  publishing.

!!! danger "`time` must be float64"
    `LightCurve.__post_init__` **rejects a float32 `time` array with a `ValueError`**.
    It is not upcast.

    At HJD ~2.46e6 one float32 ULP is 0.25 days, so by the time such an array reaches the
    constructor every pair of epochs less than six hours apart has already collapsed to a
    separation of exactly zero — the entire short-period signal is gone and nothing later
    can detect that it was ever there. Upcasting would satisfy the float64 guarantee while
    carrying that damage forward invisibly, so the constructor refuses instead, while the
    loader that produced the array is still identifiable.

    Build the array in float64 in your loader: `np.asarray(raw["mjd"], dtype=np.float64)`.
    `mag` and `mag_err` are coerced rather than rejected, since they carry no large zero
    point.

    Extractors do eventually hand times to periodfind as float32, but only after
    zero-offsetting in float64 — see
    [Feature layer → Time precision](../layers/features.md#time-precision).

```python
import numpy as np
from ml4em.types import LightCurve

class MySource:
    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        # Make one batched API call instead of N individual calls
        results = my_api.batch_get(source_ids)
        lcs = []
        for row in results:
            lc = LightCurve(
                source_id=str(row["id"]),
                time=np.asarray(row["mjd"], dtype=np.float64),   # float64, always
                mag=np.asarray(row["mag"], dtype=np.float64),
                mag_err=np.asarray(row["magerr"], dtype=np.float64),
                band=row["band"],
                survey="my_survey",
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                time_system="mjd",
            )
            lcs.append(lc)
        return lcs
```

The constructor sorts `time` ascending and permutes `mag` and `mag_err` to match, so you
do not need to sort. It does **not** drop non-finite values — whether to discard bad
epochs is your loader's decision, and silently doing it here would hide that from the
caller.

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

**float32 timestamps:** the single most common one. Many APIs and parquet files hand back
float32 columns. `LightCurve` raises rather than accepting them — cast to float64 at the
point you build the array, not afterwards. Casting a float32 array up to float64 does not
recover anything; the precision is already gone.

**Wrong array shape:** `LightCurve.__post_init__` validates that `time`, `mag`, and
`mag_err` are all 1-dimensional and the same length. If your arrays are 2D (e.g. from
a pandas DataFrame without `.values.flatten()`), you'll get a validation error.

**Flux instead of magnitude:** Many surveys return flux in Janskies or nJy, not
magnitudes. Convert before building `LightCurve`:
```python
mag = -2.5 * np.log10(flux_njy / 3631e9)   # AB magnitude
```

**HJD vs MJD:** Both are fine, but say which one you are using via `time_system`. Every
feature here depends only on time *differences*, so nothing downstream can tell the
systems apart from the values alone, and an unlabelled mix of HJD and MJD light curves
would produce plausible-looking nonsense rather than an error. Recording the system makes
a merge checkable instead of silent.
