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

import logging

import numpy as np

from ml4em.config.schema import ZTFConfig
from ml4em.types import LightCurve

log = logging.getLogger(__name__)


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

    @property
    def client(self):
        """The underlying penquins Kowalski client.

        Pass to CatalogExtractor (Gaia xmatch) and FeaturePipeline.default()
        so all catalog queries reuse the same authenticated connection.
        """
        return self._client

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

        filter_doc: dict = {
            "_id": {"$in": ids},
            "data.hjd": time_filter,
        }
        # Restrict to public + partnership + Caltech observations.
        # Matches scope-ml's program_id_selector default [1,2,3].
        if self._cfg.program_ids:
            filter_doc["data.programid"] = {"$in": self._cfg.program_ids}

        return {
            "query_type": "find",
            "query": {
                "catalog": self._cfg.collection_sources,
                "filter": filter_doc,
                "projection": {
                    "_id": 1,
                    "filter": 1,   # band ID: 1=g, 2=r, 3=i
                    "ra": 1,
                    "dec": 1,
                    "data.hjd": 1,
                    "data.mag": 1,
                    "data.magerr": 1,
                    "data.catflags": 1,
                    "data.programid": 1,
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
                    # Skipping silently would make a failed query
                    # indistinguishable from an empty region.
                    log.warning(
                        "[ztf] find query failed (status=%s): %s",
                        resp.get("status"), resp.get("message", "no message"),
                    )
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

        # Discard flagged epochs (catflags != 0 → problematic photometry).
        #
        # programid is filtered here, per epoch, and not only in the query.
        # The Mongo filter on "data.programid" selects *documents* containing
        # at least one matching epoch — it does not remove the non-matching
        # epochs from the returned array.  Relying on it alone would silently
        # return partnership and Caltech data to a run configured for public
        # data only.
        # An epoch with no programid field passes: some collections omit it,
        # and dropping those epochs would silently return empty light curves
        # rather than an error.
        allowed_programs = set(self._cfg.program_ids)
        clean = [
            pt for pt in doc.get("data", [])
            if pt.get("catflags", 1) == 0
            and pt.get("programid", None) in (allowed_programs | {None})
        ]
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
        For a catalog of N positions, call this N times; see
        docs/guides/label-preparation.md for the surrounding workflow.

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
                    log.warning(
                        "[ztf] cone_search query failed (status=%s): %s",
                        resp.get("status"), resp.get("message", "no message"),
                    )
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

    def near_ids(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        limit: int = 100_000,
    ) -> list[str]:
        """Return ZTF source IDs within radius_arcsec of (ra, dec).

        Sends a single Kowalski 'near' query against the spatial index.
        Returns only _id fields — no light curve data.  Use fetch_batch()
        on the returned IDs to get the actual light curves.

        This is round trip 1 of the two-hop pattern used by fetch_by_region()
        and matches scope-ml's get_lightcurves_via_coords near query exactly.

        Parameters
        ----------
        ra, dec:
            Region centre in decimal degrees (J2000).
        radius_arcsec:
            Search radius in arcseconds.
        limit:
            Maximum number of source IDs to return.  Default 100_000 covers
            a full ZTF quad footprint (~3600 arcsec diameter).

        Returns
        -------
        list[str]
            ZTF integer source IDs as strings.  Empty list if none found.
        """
        # Reject sparsely-sampled sources on the server.  These would be
        # dropped by FeatureConfig.min_observations anyway, and in a crowded
        # field they are the majority of hits — filtering here rather than
        # after the fetch avoids transferring light curves that are discarded
        # on arrival.
        near_filter: dict = {}
        if self._cfg.min_nobs > 0:
            near_filter["nobs"] = {"$gte": self._cfg.min_nobs}

        near_query = {
            "query_type": "near",
            "query": {
                "max_distance"   : radius_arcsec,
                "distance_units" : "arcsec",
                "radec"          : {"query_coords": [ra, dec]},
                "catalogs"       : {
                    self._cfg.collection_sources: {
                        "filter"    : near_filter,
                        "projection": {"_id": 1},
                    }
                },
            },
            "kwargs": {"max_time_ms": 30000, "limit": limit},
        }
        responses = self._client.query(
            queries=[near_query], use_batch_query=True, max_n_threads=1
        )

        source_ids: list[str] = []
        for _instance, resp_list in responses.items():
            for resp in resp_list:
                if resp.get("status") != "success":
                    log.warning(
                        "[ztf] near query failed (status=%s): %s",
                        resp.get("status"), resp.get("message", "no message"),
                    )
                    continue
                hits = (
                    resp.get("data", {})
                    .get(self._cfg.collection_sources, {})
                    .get("query_coords", [])
                )
                source_ids.extend(str(doc["_id"]) for doc in hits)

        # Kowalski truncates at `limit` without saying so, so a full result set
        # and a truncated one are the same value.  Hitting the limit exactly is
        # the only available signal that part of the region was dropped.
        if len(source_ids) >= limit:
            log.warning(
                "[ztf] near query returned %d ids, at the limit of %d — the "
                "region is probably truncated; raise `limit` or split it",
                len(source_ids), limit,
            )
        return source_ids

    def fetch_by_region(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
    ) -> tuple[list[str], list[LightCurve]]:
        """Fetch all ZTF sources in a sky region via two Kowalski round trips.

        Matches scope-ml's get_lightcurves_via_coords pattern exactly:
          Round trip 1 — near_ids(): spatial index lookup, returns source IDs only.
          Round trip 2 — fetch_batch(): fetches full light curve data for those IDs.

        Splitting the two steps is faster than a single cone_search because the
        near query hits a spatial index and returns only _ids (tiny response),
        while the find query can be parallelised across n_workers threads.

        Use this for production batch runs where you start from a sky region
        (e.g. a ZTF quad centre) rather than a pre-existing ID list.

        Parameters
        ----------
        ra:
            Right ascension of region centre in decimal degrees (J2000).
        dec:
            Declination of region centre in decimal degrees (J2000).
        radius_arcsec:
            Search radius in arcseconds.  A ZTF quad is roughly 3600 arcsec
            across; use 1800 arcsec for a ~quad-sized circular region.

        Returns
        -------
        source_ids : list[str]
            ZTF integer source IDs found in the region.
        light_curves : list[LightCurve]
            Clean, cadence-filtered LightCurves for those sources.
        """
        source_ids = self.near_ids(ra, dec, radius_arcsec)
        if not source_ids:
            return [], []
        light_curves = self.fetch_batch(source_ids)
        return source_ids, light_curves

    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        """Fetch light curves for multiple ZTF source _ids in parallel queries.

        Matches scope-ml's get_lightcurves_via_ids sliding-window pattern:
        - IDs are chunked into slices of limit_per_query (default 1000).
        - Each iteration dispatches min(n_remaining_chunks, n_workers) queries
          simultaneously (use_batch_query=True, max_n_threads=n_workers).
        - This keeps n_workers Kowalski threads saturated without sending all
          queries at once — important when len(source_ids) >> n_workers.

        Set ZTFConfig.n_workers ≥ 8 and limit_per_query = 1000 on MSI.

        Parameters
        ----------
        source_ids:
            ZTF integer _ids represented as strings.

        Returns
        -------
        list[LightCurve]
            Clean, cadence-filtered LightCurves.  Empty list if none found.
        """
        if not source_ids:
            return []

        ids = [int(sid) for sid in source_ids]
        n_workers     = max(1, self._cfg.n_workers)
        limit         = max(1, self._cfg.limit_per_query)

        # Build one query per limit-sized slice of IDs
        all_chunks = [ids[i : i + limit] for i in range(0, len(ids), limit)]
        all_queries = [self._build_query(chunk) for chunk in all_chunks]

        light_curves: list[LightCurve] = []

        # Sliding window: send n_workers queries at a time — matches scope-ml's
        # while True loop with Nqueries = min(len(queries), Ncore).
        i = 0
        while i < len(all_queries):
            batch = all_queries[i : i + n_workers]
            responses = self._client.query(
                queries=batch,
                use_batch_query=True,
                max_n_threads=len(batch),
            )
            light_curves.extend(self._parse_responses(responses))
            i += len(batch)

        return light_curves


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
