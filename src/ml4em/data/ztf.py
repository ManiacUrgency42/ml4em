"""
ZTF light curve source via Kowalski.

Requires the 'ztf' optional dependency group:
    pip install "ml4em[ztf]"

Token
-----
Set ML4EM_ZTF_TOKEN in your environment or .env file.
Never put the token in config.yaml.

    ML4EM_ZTF_TOKEN=your_kowalski_token

ZTF source model
----------------
Each document in ZTF_sources_* represents a single-band light curve.
The integer _id uniquely identifies one (sky position, filter) pair.
Filters: 1 → g, 2 → r, 3 → i.

To get multi-band coverage for a sky position, query multiple _ids
(one per band) that correspond to the same coordinates.  Use
fetch_by_position(ra, dec) to resolve a sky coordinate to light curves
directly via a Kowalski cone search.
"""

from __future__ import annotations

import numpy as np

from ml4em.config.schema import ZTFConfig
from ml4em.types import LightCurve


# ZTF filter integer → SDSS-like band name
_FILTER_MAP: dict[int, str] = {1: "g", 2: "r", 3: "i"}


class ZTFSource:
    """Fetch ZTF light curves from Kowalski using the penquins client.

    Parameters
    ----------
    config:
        ZTFConfig from WDBConfig.sources.ztf.
    token:
        Kowalski API token.  Obtain via ml4em.config.get_ztf_token().

    Examples
    --------
    >>> from ml4em.config import load_default_config, get_ztf_token
    >>> cfg = load_default_config()
    >>> source = ZTFSource(cfg.sources.ztf, get_ztf_token())
    >>> lcs = source.fetch("1234567890")
    >>> lcs = source.fetch_batch(["1234567890", "1234567891", "1234567892"])
    """

    def __init__(self, config: ZTFConfig, token: str) -> None:
        self._cfg = config
        self._token = token
        self._client = self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        """Establish a single-instance penquins Kowalski connection."""
        try:
            from penquins import Kowalski
        except ImportError as exc:
            raise ImportError(
                "penquins is required for ZTF data access.\n"
                "Install with: pip install 'ml4em[ztf]'"
            ) from exc

        return Kowalski(
            timeout=self._cfg.timeout,
            instances={
                "kowalski": {
                    "protocol": self._cfg.protocol,
                    "host": self._cfg.host,
                    "port": self._cfg.port,
                    "token": self._token,
                }
            },
        )

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    def _build_query(self, ids: list[int]) -> dict:
        """Build a Kowalski 'find' query for the given integer source IDs."""
        time_filter: dict = {"$gt": 0.0}
        if self._cfg.max_timestamp_hjd is not None:
            time_filter["$lte"] = self._cfg.max_timestamp_hjd

        return {
            "query_type": "find",
            "query": {
                "catalog": self._cfg.collection_sources,
                "filter": {
                    "_id": {"$in": ids},
                    "data.hjd": time_filter,
                },
                "projection": {
                    "_id": 1,
                    "filter": 1,   # band ID: 1=g, 2=r, 3=i
                    "ra": 1,
                    "dec": 1,
                    "data.hjd": 1,
                    "data.mag": 1,
                    "data.magerr": 1,
                    "data.catflags": 1,
                },
            },
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_responses(self, responses: dict) -> list[LightCurve]:
        """Parse multi-instance Kowalski response into LightCurve objects.

        Parameters
        ----------
        responses:
            Dict keyed by instance name (e.g. "kowalski"), each value is a
            list of per-query response dicts with "status" and "data" keys.
        """
        light_curves: list[LightCurve] = []
        for _instance, resp_list in responses.items():
            for resp in resp_list:
                if resp.get("status") != "success":
                    continue
                for doc in resp.get("data", []):
                    lc = self._doc_to_lightcurve(doc)
                    if lc is not None:
                        light_curves.append(lc)
        return light_curves

    def _doc_to_lightcurve(self, doc: dict) -> LightCurve | None:
        """Convert one Kowalski source document to a LightCurve.

        Returns None if:
        - the band is not in the configured bands list, or
        - the document has no clean (catflags == 0) observations, or
        - after cadence filtering fewer than 1 point remain.
        """
        filter_id = doc.get("filter")
        band = _FILTER_MAP.get(filter_id)
        if band is None or band not in self._cfg.bands:
            return None

        ra  = float(doc.get("ra", 0.0))
        dec = float(doc.get("dec", 0.0))

        # Discard flagged epochs (catflags != 0 → problematic photometry)
        clean = [pt for pt in doc.get("data", []) if pt.get("catflags", 1) == 0]
        if not clean:
            return None

        tme = np.array([[pt["hjd"], pt["mag"], pt["magerr"]] for pt in clean])
        # Sort chronologically
        tme = tme[np.argsort(tme[:, 0])]
        t, m, e = tme[:, 0], tme[:, 1], tme[:, 2]

        # Remove intra-night duplicates that bias period-finding
        if self._cfg.min_cadence_days > 0:
            t, m, e = _remove_high_cadence(t, m, e, self._cfg.min_cadence_days)

        if len(t) == 0:
            return None

        return LightCurve(
            source_id=str(doc["_id"]),
            time=t,
            mag=m,
            mag_err=e,
            band=band,
            survey="ztf",
            ra=ra,
            dec=dec,
        )

    # ------------------------------------------------------------------
    # Public interface  (satisfies LightCurveSource Protocol)
    # ------------------------------------------------------------------

    def fetch_by_position(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 2.0,
    ) -> list[LightCurve]:
        """Fetch light curves for all ZTF sources within radius_arcsec of (ra, dec).

        Sends a single Kowalski cone_search query and returns all matching
        single-band light curves, cleaned and cadence-filtered exactly as
        fetch_batch() does.

        Use this when you have sky coordinates (e.g. from a WDB catalog)
        and need to resolve them to ZTF source IDs and light curve data.
        For a catalog of N positions, call this N times or see
        scripts/prepare_labels.py for the batch workflow.

        Parameters
        ----------
        ra:
            Right ascension in decimal degrees (J2000).
        dec:
            Declination in decimal degrees (J2000).
        radius_arcsec:
            Cone search radius in arcseconds.  Default 2.0 arcsec matches
            the Gaia cross-match radius and is appropriate for isolated stars.
            Increase to 5–10 arcsec in crowded fields.

        Returns
        -------
        list[LightCurve]
            All matching light curves across all bands, cleaned and
            cadence-filtered.  Empty list if no ZTF source is found
            within the search radius.
        """
        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "cone_search_radius": radius_arcsec,
                    "cone_search_unit": "arcsec",
                    "radec": {"center": [ra, dec]},
                },
                "catalogs": {
                    self._cfg.collection_sources: {},
                },
            },
        }
        responses = self._client.query(
            queries=[query],
            use_batch_query=True,
            max_n_threads=1,
        )
        return self._parse_cone_responses(responses)

    def _parse_cone_responses(self, responses: dict) -> list[LightCurve]:
        """Parse a cone_search Kowalski response into LightCurve objects.

        The cone_search response nests results under the catalog name and
        position key ("center" for a single position), unlike the find
        response parsed by _parse_responses().
        """
        light_curves: list[LightCurve] = []
        for _instance, resp_list in responses.items():
            for resp in resp_list:
                if resp.get("status") != "success":
                    continue
                data = resp.get("data", {})
                catalog_data = data.get(self._cfg.collection_sources, {})
                # cone_search with a single position uses key "center"
                docs = catalog_data.get("center", [])
                for doc in docs:
                    lc = self._doc_to_lightcurve(doc)
                    if lc is not None:
                        light_curves.append(lc)
        return light_curves

    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        """Fetch light curves for multiple ZTF source _ids in one query.

        Parameters
        ----------
        source_ids:
            ZTF integer _ids represented as strings.

        Returns
        -------
        list[LightCurve]
            Clean, cadence-filtered LightCurves.  Empty list if none found.
        """
        ids = [int(sid) for sid in source_ids]
        query = self._build_query(ids)
        responses = self._client.query(
            queries=[query],
            use_batch_query=True,
            max_n_threads=1,
        )
        return self._parse_responses(responses)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _remove_high_cadence(
    t: np.ndarray,
    m: np.ndarray,
    e: np.ndarray,
    min_cadence_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove observations within min_cadence_days of the previous kept point.

    Eliminates same-night repeat observations that create aliased peaks in
    period-finding algorithms.  Input arrays must already be sorted by time.
    """
    if len(t) == 0:
        return t, m, e

    keep = np.ones(len(t), dtype=bool)
    last_kept = t[0]
    for i in range(1, len(t)):
        if t[i] - last_kept < min_cadence_days:
            keep[i] = False
        else:
            last_kept = t[i]

    return t[keep], m[keep], e[keep]
