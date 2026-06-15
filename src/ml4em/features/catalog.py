"""
Gaia EDR3 catalog cross-match extractor.

Enriches each source's FeatureVector with astrometric and photometric
properties from the nearest Gaia EDR3 counterpart within a configurable
cone radius.  These features help distinguish white dwarf systems:
- High parallax → nearby → WD candidate
- Blue BP-RP (< 0) → hot stellar atmosphere → WD nature
- Low RUWE (< 1.4) → single, unresolved source → clean astrometry

Preprocessing
-------------
Extract (RA, Dec) from the first LightCurve in the list.

Feature generation
------------------
Cone-search Gaia EDR3 for the nearest counterpart.
Return parallax, parallax_error, BP-RP colour, and RUWE.

Post-processing
---------------
None — raw Gaia values are directly interpretable.

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
        CatalogConfig from WDBConfig.features.catalog.

    Notes
    -----
    Implementation is a stub pending backend selection (astroquery vs Kowalski).
    The planned query (astroquery backend):

        from astroquery.gaia import Gaia
        result = Gaia.cone_search_async(
            coordinate=SkyCoord(ra, dec, unit="deg"),
            radius=Angle(radius_arcsec, "arcsec"),
        ).get_results()
        nearest = result[result["dist"].argmin()]

    Fields to retrieve: parallax, parallax_error, bp_rp, ruwe
    """

    def __init__(self, config: CatalogConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, lcs: list[LightCurve]) -> tuple[float, float]:
        """Extract (ra, dec) from the first LightCurve."""
        return lcs[0].ra, lcs[0].dec

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------

    def _query_gaia(self, ra: float, dec: float) -> dict[str, Any]:
        """Query Gaia EDR3 for the nearest counterpart.

        Parameters
        ----------
        ra, dec:
            Source coordinates in decimal degrees (J2000).

        Returns
        -------
        dict[str, Any]
            Keys: gaia_parallax, gaia_parallax_error, gaia_bp_rp, gaia_ruwe.
            All values are None if no counterpart is found.
        """
        raise NotImplementedError(
            "CatalogExtractor._query_gaia is not yet implemented.\n"
            "Planned backends:\n"
            "  1. astroquery.gaia.Gaia.cone_search_async (public Gaia TAP+)\n"
            "  2. Kowalski Gaia_EDR3 cone_search (consistent with ZTF path)\n"
            f"Query will search radius={self._cfg.xmatch_radius_arcsec} arcsec\n"
            "around (ra, dec) and return parallax, parallax_error, bp_rp, ruwe."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, lcs: list[LightCurve]) -> dict[str, Any]:
        """Return 4 Gaia features for the nearest counterpart.

        Parameters
        ----------
        lcs:
            All bands for the source.  Coordinates taken from lcs[0].

        Returns
        -------
        dict[str, Any]
            gaia_parallax, gaia_parallax_error, gaia_bp_rp, gaia_ruwe.
            Empty dict if Gaia query is not yet implemented or fails.
        """
        if not lcs or not self._cfg.include_gaia:
            return {}

        try:
            ra, dec = self._preprocess(lcs)
            return self._query_gaia(ra, dec)
        except NotImplementedError:
            return {}
        except Exception:
            return {}
