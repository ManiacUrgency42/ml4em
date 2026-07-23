"""
Gaia EDR3 catalog cross-match extractor.

Enriches each source's FeatureVector with astrometric and photometric
properties from the nearest Gaia EDR3 counterpart within a configurable
cone radius.  These features help distinguish white dwarf systems:
- High parallax → nearby → WD candidate
- Blue BP-RP (< 0) → hot stellar atmosphere → WD nature
- Low astrometric_excess_noise → clean single-source astrometry → reliable fit

Physics alignment
-----------------
Matches scope-ml's external_xmatch.py exactly:
- Kowalski cone_search against the Gaia_EDR3 collection
- Batch multi-position query — all sources in one network round trip
- Multiple matches resolved by nearest angular separation
- astrometric_excess_noise (not RUWE) — matches scope-ml's Gaia projection

Backend
-------
Requires a live penquins Kowalski client, passed in at construction time.
The FeaturePipeline.default() factory accepts an optional kowalski_client
and forwards it here.  If no client is provided (or include_gaia=False),
all Gaia fields are left at their FeatureVector defaults (None).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ml4em.config.schema import CatalogConfig
from ml4em.types import LightCurve

# Kowalski collection name — same as scope-ml config.defaults.yaml
_GAIA_CATALOG = "Gaia_EDR3"

# Fields projected from Kowalski — matches scope-ml's catalog_info for Gaia_EDR3.
# coordinates.radec_geojson is required for nearest-match tie-breaking.
_GAIA_PROJECTION = {
    "_id"                       : 1,
    "parallax"                  : 1,
    "parallax_error"            : 1,
    "phot_bp_mean_mag"          : 1,
    "phot_rp_mean_mag"          : 1,
    "astrometric_excess_noise"  : 1,
    "coordinates.radec_geojson" : 1,
}


class CatalogExtractor:
    """Cross-match sources against Gaia EDR3 to add 4 astrometric features.

    Parameters
    ----------
    config:
        CatalogConfig from FeatureConfig.catalog.
    kowalski_client:
        Authenticated penquins Kowalski instance.  Pass the same client
        used by ZTFSource (ZTFSource.client).  If None, Gaia features
        are skipped and all fields remain at their FeatureVector defaults.
    """

    def __init__(self, config: CatalogConfig, kowalski_client=None) -> None:
        self._cfg = config
        self._client = kowalski_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_match(self, ra: float, dec: float, matches: list[dict]) -> dict:
        """Return the Gaia match closest to (ra, dec) by angular separation.

        Matches scope-ml's tie-breaking logic in external_xmatch.py.
        GeoJSON stores longitude in (-180, 180); add 180 to recover RA in (0, 360).
        """
        if len(matches) == 1:
            return matches[0]

        ra_rad  = math.radians(ra)
        dec_rad = math.radians(dec)
        best     = matches[0]
        best_sep = float("inf")

        for m in matches:
            try:
                coords = m["coordinates"]["radec_geojson"]["coordinates"]
                m_ra  = coords[0] + 180.0   # GeoJSON lon → RA
                m_dec = coords[1]
                # Haversine angular separation
                d_ra  = math.radians(m_ra  - ra)
                d_dec = math.radians(m_dec - dec)
                a = (
                    math.sin(d_dec / 2) ** 2
                    + math.cos(dec_rad)
                    * math.cos(math.radians(m_dec))
                    * math.sin(d_ra / 2) ** 2
                )
                sep = 2.0 * math.asin(math.sqrt(min(a, 1.0)))
                if sep < best_sep:
                    best_sep = sep
                    best = m
            except (KeyError, TypeError):
                continue

        return best

    def _parse_gaia_doc(self, doc: dict) -> dict[str, Any]:
        """Extract the 4 Gaia features from a matched Gaia EDR3 document.

        bp_rp = phot_bp_mean_mag - phot_rp_mean_mag  (not a direct field).
        Matches scope-ml's feature naming convention.
        """
        def _f(key: str):
            v = doc.get(key)
            return float(v) if v is not None else None

        bp = doc.get("phot_bp_mean_mag")
        rp = doc.get("phot_rp_mean_mag")
        bp_rp = (float(bp) - float(rp)) if (bp is not None and rp is not None) else None

        return {
            "gaia_parallax"                  : _f("parallax"),
            "gaia_parallax_error"            : _f("parallax_error"),
            "gaia_bp_rp"                     : bp_rp,
            "gaia_astrometric_excess_noise"  : _f("astrometric_excess_noise"),
        }

    @staticmethod
    def _no_match() -> dict[str, Any]:
        return {
            "gaia_parallax"                  : None,
            "gaia_parallax_error"            : None,
            "gaia_bp_rp"                     : None,
            "gaia_astrometric_excess_noise"  : None,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self, sources: list[list[LightCurve]]
    ) -> list[dict[str, Any]]:
        """Batch cross-match all sources against Gaia EDR3 in one Kowalski query.

        Parameters
        ----------
        sources:
            Each element is all bands for one source.  Coordinates are
            taken from the band with the most observations.

        Returns
        -------
        list[dict[str, Any]]
            One dict per source with gaia_parallax, gaia_parallax_error,
            gaia_bp_rp, gaia_astrometric_excess_noise.
            All values are None when no Gaia counterpart is found within
            xmatch_radius_arcsec, or when no client is available.
        """
        if not self._cfg.include_gaia or self._client is None:
            return [{} for _ in sources]

        # Collect (original_index, source_id, ra, dec) for all non-empty sources
        valid: list[tuple[int, str, float, float]] = []
        for i, lcs in enumerate(sources):
            if lcs:
                primary = max(lcs, key=lambda lc: lc.n_obs)
                valid.append((i, primary.source_id, primary.ra, primary.dec))

        results: list[dict[str, Any]] = [{} for _ in sources]

        if not valid:
            return results

        # Build multi-position radec dict: {source_id: [ra, dec], ...}
        # Kowalski echoes these keys in the response, allowing per-source lookup.
        radec = {sid: [ra, dec] for _, sid, ra, dec in valid}

        # Split radec dict across n_workers — each worker gets one cone_search
        # query covering a subset of sources.  Kowalski executes them in
        # parallel (max_n_threads).  Matches scope-ml's split_dict / Ncore
        # pattern in external_xmatch.py.
        n_workers = max(1, self._cfg.n_workers)
        items = list(radec.items())
        chunk_size = max(1, -(-len(items) // n_workers))   # ceiling division
        chunks = [dict(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]

        queries = [
            {
                "query_type": "cone_search",
                "query": {
                    "object_coordinates": {
                        "radec"              : chunk,
                        "cone_search_radius" : self._cfg.xmatch_radius_arcsec,
                        "cone_search_unit"   : "arcsec",
                    },
                    "catalogs": {
                        _GAIA_CATALOG: {
                            "filter"    : {},
                            "projection": _GAIA_PROJECTION,
                        }
                    },
                },
            }
            for chunk in chunks
        ]

        try:
            responses = self._client.query(
                queries=queries, use_batch_query=True, max_n_threads=n_workers
            )

            # Flatten response: data -> Gaia_EDR3 -> {source_id: [matches]}
            gaia_by_sid: dict[str, list] = {}
            for _instance, resp_list in responses.items():
                for resp in resp_list:
                    if resp.get("status") != "success":
                        continue
                    gaia_by_sid.update(
                        resp.get("data", {}).get(_GAIA_CATALOG, {})
                    )

            for src_idx, sid, ra, dec in valid:
                matches = gaia_by_sid.get(sid, [])
                if not matches:
                    results[src_idx] = self._no_match()
                else:
                    doc = self._nearest_match(ra, dec, matches)
                    results[src_idx] = self._parse_gaia_doc(doc)

        except Exception:
            # Network / auth failure — explicit no-match for all queried sources
            for src_idx, _, _, _ in valid:
                results[src_idx] = self._no_match()

        return results
