# Data Layer

Fetches raw photometric observations from survey databases and returns them as `LightCurve` objects. The feature layer is the sole consumer — everything about *how* data is fetched is hidden inside the source implementation.

**Consumes:** `source_id` strings — survey-native identifiers

**Emits:** `list[LightCurve]` — one object per (source, band)

```
src/ml4em/data/
  base.py         LightCurveSource Protocol
  ztf.py          ZTFSource          [implemented]
  simulation.py   SimulatedSource    [stub]
```

## Contents

- [LightCurveSource Protocol](#lightcurvesource)
- [ZTFSource](#ztfsource)
- [SimulatedSource (stub)](#simulatedsource)

---

## `LightCurveSource` Protocol { #lightcurvesource }

The contract every data source must satisfy. Any class with a compatible `fetch_batch` method qualifies — no base class or registration required.

**Consumes:** `list[str]` — source ID strings

**Emits:** `list[LightCurve]` — all light curves across all requested sources and bands

```python
@runtime_checkable
class LightCurveSource(Protocol):
    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]: ...
```

`fetch_batch()` is the only member. Pass a one-element list to fetch a single source:

```python
lcs = source.fetch_batch([single_id])
```

Every source returns one `LightCurve` per band per sky position. For ZTF each `_id`
encodes a single band, so a multi-band source is several ids; a survey that keys one
object across all its filters would instead return several `LightCurve`s for one id.
Order is not guaranteed.

Time arrays must be **float64** — `LightCurve` rejects float32. See
[Add a data source](../guides/add-data-source.md) for the reason and for the rest of the
loader contract.

---

## `ZTFSource` { #ztfsource }

Fetches ZTF photometric light curves from the Kowalski database via the `penquins` client. Issues a single batched query for all requested IDs.

**Consumes:** ZTF integer source IDs cast to `str`

**Emits:** `list[LightCurve]` — one per (source, band) that survives quality filtering

```python
from ml4em.data import ZTFSource
from ml4em.config import load_config, get_ztf_token

source = ZTFSource(load_config().sources.ztf, token=get_ztf_token())
lcs = source.fetch_batch(["686149073900013696", "686149073900013697"])
```

### `fetch_by_position(ra, dec, radius_arcsec)`

Cone search — resolves a sky coordinate to ZTF light curves directly, without knowing the source `_id` in advance. Use this when your input is a catalog of (ra, dec) positions (e.g. `data/wdb_sources.csv`) rather than ZTF IDs.

**Consumes:** `ra`, `dec` in decimal degrees (J2000); `radius_arcsec` (default 2.0)

**Emits:** `list[LightCurve]` — all matching sources within the cone, all bands, same cleaning as `fetch_batch`

```python
lcs = source.fetch_by_position(ra=256.123, dec=45.678, radius_arcsec=2.0)
```

Default radius is 2.0 arcsec — appropriate for isolated stars. Increase to 5–10 arcsec in crowded fields.

### `near_ids()` and `fetch_by_region(ra, dec, radius_arcsec)`

For production runs that start from a sky region rather than an ID list,
`fetch_by_region()` uses two Kowalski round trips instead of one cone search:
`near_ids()` hits the spatial index and returns only `_id`s (a tiny response), then
`fetch_batch()` pulls the light curves for those IDs and can be parallelised across
`n_workers` threads. It returns `(source_ids, light_curves)`.

`near_ids()` also applies `sources.ztf.min_nobs` server-side. Kowalski indexes an `nobs`
field per source, so filtering there rejects sparsely-sampled sources before any light
curve data crosses the network. Without it a region query returns every source in the
footprint and the pipeline pays the full transfer cost for rows it will immediately
discard against `features.min_observations`. Set it to 0 to disable.

A ZTF quad is roughly 3600 arcsec across, so `radius_arcsec=1800` gives a quad-sized
circular region.

### Data cleaning

Three filtering steps are applied before returning:

| Step | Condition | Effect |
|------|-----------|--------|
| catflags filter | `catflags != 0` | Drops observations flagged bad by the ZTF pipeline |
| programid filter | epoch `programid` not in `sources.ztf.program_ids` | Drops epochs from unwanted observing programs |
| Intra-night dedup | Δt < 5 min from the previous kept point | Drops near-simultaneous repeat observations |

The programid filter is applied **per epoch**, not only in the Mongo query. The query
filter on `data.programid` selects *documents* containing at least one matching epoch; it
does not remove the non-matching epochs from the returned array. Relying on the query
alone would silently return partnership and Caltech data to a run configured for public
data only. An epoch with no `programid` field passes, because some collections omit it
and dropping those epochs would return empty light curves rather than an error.

The cadence threshold is `sources.ztf.min_cadence_days`, defaulting to
`ZTF_MIN_CADENCE_DAYS = 5/1440` days (5 minutes). Set it to 0 to disable deduplication.
See [Surveys → Cadence](../background/surveys.md#ztf) for why 5 minutes rather than the
conventional half hour.

A source whose document survives none of these — no clean epochs, or nothing left after
cadence filtering — is dropped entirely rather than returned empty.

### Band mapping

ZTF stores filter codes as integers. `ZTFSource` converts them to the string band codes used in `LightCurve.band`:

| Integer | Band |
|---------|------|
| 1 | `g` |
| 2 | `r` |
| 3 | `i` |

---

## `SimulatedSource` *(stub)* { #simulatedsource }

Not yet implemented.

> **Status:** raises `NotImplementedError`.

---

[← Foundation](foundation.md){ .md-button } [Features →](features.md){ .md-button .md-button--primary }
