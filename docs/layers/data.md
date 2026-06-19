# Data Layer

The data layer's job is simple: given a source identifier, return a list of
`LightCurve` objects. Everything about *how* that data is fetched — which API, which
table, which format — is hidden inside the source implementation.

```
src/ml4em/data/
  base.py         LightCurveSource Protocol
  ztf.py          ZTFSource   — Kowalski/penquins client      [implemented]
  rubin.py        RubinSource — Rubin DP1 via TAP             [stub]
  simulation.py   SimulatedSource — Lcurve wrapper            [stub]
```

---

## Protocol — `LightCurveSource`

```python
class LightCurveSource(Protocol):
    def fetch(self, source_id: str) -> list[LightCurve]: ...
    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]: ...
```

Any class with these two methods is a valid source. The feature layer accepts any
compliant object without importing the concrete source class.

**`fetch(source_id)`** → returns all bands for one source as a flat list of
`LightCurve` objects (one per band available).

**`fetch_batch(source_ids)`** → batch version. Implementations should make one network
request for all IDs rather than N separate requests.

---

## `ZTFSource`

Fetches ZTF light curves from the Kowalski database via the `penquins` Python client.

!!! info "What is Kowalski?"
    Kowalski is ZTF's data access service — a MongoDB database of ZTF observations
    with a query API. See [Surveys → ZTF](../background/surveys.md#ztf).

### Setup

```python
from ml4em.data import ZTFSource
from ml4em.config import load_config, get_ztf_token

cfg = load_config()
source = ZTFSource(cfg.sources.ztf, token=get_ztf_token())
```

### Batch fetching

`fetch_batch` sends a single batched Kowalski `find` query for all source IDs, which
is much faster than N individual requests:

```python
lcs = source.fetch_batch(["686149073900013696", "686149073900013697"])
# → list[LightCurve] — one per (source_id, band) combination
```

### Data cleaning

`ZTFSource` applies two cleaning steps before returning data:

1. **catflags filtering:** observations where `catflags != 0` are discarded. These are
   flagged as unreliable by the ZTF pipeline (bad seeing, cosmic rays, etc.).

2. **Intra-night duplicate removal:** observations within 30 minutes
   (`ZTF_MIN_CADENCE_DAYS`) of an earlier observation in the same night are removed.
   This prevents the nightly cadence from creating spurious short-period signals.

!!! info "What are catflags?"
    ZTF attaches a quality flag integer to each observation. `catflags != 0` means
    one or more quality conditions failed. See [Surveys → ZTF](../background/surveys.md#catflags).

### Band mapping

ZTF uses integer filter codes internally: `1 → g`, `2 → r`, `3 → i`. `ZTFSource`
converts these to the string band codes used in `LightCurve.band`.

---

## `RubinSource` *(stub)*

Will query Rubin DP1 via TAP using `pyvo`.

!!! info "What is TAP?"
    TAP (Table Access Protocol) is a standard SQL-like web API for astronomical data.
    See [Surveys → Rubin](../background/surveys.md#tap).

Planned implementation:
- JOIN `dp1.Object ⋈ dp1.ForcedSource ⋈ dp1.Visit` on `objectId`
- Filter by band using the visit metadata
- Convert `psfFlux` (nanojanskies) to magnitude
- Return up to 6 `LightCurve` objects per `objectId` (one per band: u, g, r, i, z, y)

Status: raises `NotImplementedError`. Pending DP1 schema confirmation.

---

## `SimulatedSource` *(stub)*

Will wrap Tom Marsh's **Lcurve** code to produce physics-based synthetic white dwarf
binary light curves.

- `source_id` will be a path to a `.mod` Lcurve parameter file or a grid index
- The simulation models orbital mechanics (eclipses, reflection, ellipsoidal variation)
  given parameters like masses, radii, inclination
- Gaussian photon noise is injected to simulate real measurement conditions
- Used to generate training data for WDB detection without needing labeled survey data

Status: raises `NotImplementedError`. Pending Lcurve integration.

---

## Adding a new source

1. Create `src/ml4em/data/my_source.py`
2. Implement `fetch(source_id)` and `fetch_batch(source_ids)`
3. Return `list[LightCurve]` — map your survey's time system to MJD
4. Optionally export from `data/__init__.py` for convenience

No other files need to change. The feature layer will accept your new source because
it only checks for the two methods at runtime.

```python
class MySource:
    def fetch(self, source_id: str) -> list[LightCurve]:
        raw = my_api.query(source_id)
        return [LightCurve(
            source_id=source_id,
            time=raw["mjd"],
            mag=raw["mag"],
            mag_err=raw["mag_err"],
            band=raw["band"],
            survey="my_survey",
            ra=raw["ra"],
            dec=raw["dec"],
        )]

    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        raw = my_api.batch_query(source_ids)
        return [...]
```

See [Guide: Add a Data Source](../guides/add-data-source.md) for step-by-step instructions.
