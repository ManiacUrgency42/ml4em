# Data Layer

Fetches raw photometric observations from survey databases and returns them as `LightCurve` objects. The feature layer is the sole consumer — everything about *how* data is fetched is hidden inside the source implementation.

**Consumes:** `source_id` strings — survey-native identifiers

**Emits:** `list[LightCurve]` — one object per (source, band)

```
src/ml4em/data/
  base.py         LightCurveSource Protocol
  ztf.py          ZTFSource          [implemented]
  rubin.py        RubinSource        [stub]
  simulation.py   SimulatedSource    [stub]
```

## Contents

- [LightCurveSource Protocol](#lightcurvesource)
- [ZTFSource](#ztfsource)
- [RubinSource (stub)](#rubinsource)
- [SimulatedSource (stub)](#simulatedsource)

---

## `LightCurveSource` Protocol { #lightcurvesource }

The contract every data source must satisfy. Any class with a compatible `fetch_batch` method qualifies — no base class or registration required.

**Consumes:** `list[str]` — source ID strings

**Emits:** `list[LightCurve]` — all light curves across all requested sources and bands

```python
class LightCurveSource(Protocol):
    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]: ...
```

Pass a one-element list to fetch a single source:

```python
lcs = source.fetch_batch([single_id])
```

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

### Data cleaning

Two filtering steps are applied before returning:

| Step | Condition | Effect |
|------|-----------|--------|
| catflags filter | `catflags != 0` | Drops observations flagged bad by the ZTF pipeline |
| Intra-night dedup | Δt < 30 min between same-night observations | Drops near-simultaneous repeat observations |

### Band mapping

ZTF stores filter codes as integers. `ZTFSource` converts them to the string band codes used in `LightCurve.band`:

| Integer | Band |
|---------|------|
| 1 | `g` |
| 2 | `r` |
| 3 | `i` |

---

## `RubinSource` *(stub)* { #rubinsource }

Will query Rubin DP1 via the TAP protocol using `pyvo`.

**Consumes:** Rubin `objectId` strings

**Emits:** `list[LightCurve]` — up to 6 per source (bands: u, g, r, i, z, y)

Planned query joins `dp1.Object`, `dp1.ForcedSource`, and `dp1.Visit` on `objectId`, converting `psfFlux` (nanojanskies) to AB magnitudes.

> **Status:** raises `NotImplementedError` — pending Rubin DP1 schema confirmation.

---

## `SimulatedSource` *(stub)* { #simulatedsource }

Not yet implemented.

> **Status:** raises `NotImplementedError`.

---

[← Foundation](foundation.md){ .md-button } [Features →](features.md){ .md-button .md-button--primary }
