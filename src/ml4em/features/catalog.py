"""
Gaia EDR3 catalog cross-match extractor.

Enriches each source's FeatureVector with astrometric and photometric
properties from the nearest Gaia EDR3 counterpart within a configurable
cone radius.  These features help distinguish white dwarf systems:
- High parallax → nearby → WD candidate
- Blue BP-RP (< 0) → hot stellar atmosphere → WD nature
- Low RUWE (< 1.4) → single, unresolved source → clean astrometry

Status: stub — Gaia TAP query implementation pending.

Two possible backends (to be implemented):
1. astroquery.gaia.Gaia.cone_search (simplest; no auth required)
2. Kowalski Gaia_EDR3 cone search (consistent with ZTF data path)
"""

from __future__ import annotations

from typing import Any

from ml4em.config.schema import CatalogConfig
from ml4em.types import LightCurve


class CatalogExtractor:
    """Cross-match sources against Gaia EDR3 to add 4 astrometric features.

    Parameters
    ----------
    config:
        CatalogConfig from FeatureConfig.catalog.
    """

    def __init__(self, config: CatalogConfig) -> None:
        self._cfg = config

    def _query_gaia(self, ra: float, dec: float) -> dict[str, Any]:
        raise NotImplementedError(
            "CatalogExtractor._query_gaia is not yet implemented.\n"
            "Planned backends:\n"
            "  1. astroquery.gaia.Gaia.cone_search_async (public Gaia TAP+)\n"
            "  2. Kowalski Gaia_EDR3 cone_search (consistent with ZTF path)\n"
            f"Query will search radius={self._cfg.xmatch_radius_arcsec} arcsec\n"
            "around (ra, dec) and return parallax, parallax_error, bp_rp, ruwe."
        )

    def extract(
        self, sources: list[list[LightCurve]]
    ) -> list[dict[str, Any]]:
        """Return 4 Gaia features per source.

        Parameters
        ----------
        sources:
            Each element is all bands for one source.  Coordinates taken
            from the first LightCurve in each group.

        Returns
        -------
        list[dict[str, Any]]
            One dict per source with gaia_parallax, gaia_parallax_error,
            gaia_bp_rp, gaia_ruwe.  Empty dict if Gaia xmatch is not
            implemented or fails.
        """
        results: list[dict[str, Any]] = []
        for lcs in sources:
            if not lcs or not self._cfg.include_gaia:
                results.append({})
                continue
            try:
                ra, dec = lcs[0].ra, lcs[0].dec
                results.append(self._query_gaia(ra, dec))
            except (NotImplementedError, Exception):
                results.append({})
        return results
