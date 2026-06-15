"""
Rubin DP1 light curve source via TAP.

Requires the 'rubin' optional dependency group:
    pip install "ml4em[rubin]"

Token
-----
Set ML4EM_RUBIN_TOKEN in your environment or .env file.
Never put the token in config.yaml.

    ML4EM_RUBIN_TOKEN=your_rsp_token

Rubin source model
------------------
Each Rubin source is identified by dp1.Object.objectId (a 64-bit integer).
Photometry is stored in dp1.ForcedSource (per-visit forced-position fluxes),
joined to dp1.Visit for the observation timestamp and filter.

One objectId may have observations in up to six bands (u, g, r, i, z, y),
so fetch() on one objectId may return up to six LightCurves.

Status: stub — ADQL query implementation pending Rubin DP1 schema review.
"""

from __future__ import annotations

from ml4em.config.schema import RubinConfig
from ml4em.types import LightCurve


class RubinSource:
    """Fetch Rubin DP1 light curves via the TAP service.

    Parameters
    ----------
    config:
        RubinConfig from WDBConfig.sources.rubin.
    token:
        RSP API token.  Obtain via ml4em.config.get_rubin_token().

    Notes
    -----
    Implementation is a stub pending Rubin DP1 schema confirmation.
    The planned query joins dp1.ForcedSource with dp1.Visit to retrieve
    per-epoch calibrated flux, MJD timestamp, and filter name, grouped
    by objectId.  Fluxes will be converted to AB magnitudes via:

        mag = -2.5 * log10(flux_nJy) + 31.4

    For local HPC / offline use, set config.data_path to a directory
    containing pre-downloaded parquet files; the implementation will
    read from disk instead of querying TAP.
    """

    def __init__(self, config: RubinConfig, token: str) -> None:
        self._cfg = config
        self._token = token

    # ------------------------------------------------------------------
    # Public interface  (satisfies LightCurveSource Protocol)
    # ------------------------------------------------------------------

    def fetch(self, source_id: str) -> list[LightCurve]:
        """Fetch all available bands for one dp1.Object.

        Parameters
        ----------
        source_id:
            dp1.Object.objectId cast to str.

        Returns
        -------
        list[LightCurve]
            One LightCurve per band that has observations.
        """
        return self.fetch_batch([source_id])

    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        """Fetch light curves for multiple dp1.Object objectIds.

        Parameters
        ----------
        source_ids:
            dp1.Object.objectId values cast to str.

        Returns
        -------
        list[LightCurve]
            All LightCurves across all requested sources and bands.
        """
        raise NotImplementedError(
            "RubinSource.fetch_batch is not yet implemented.\n"
            "Planned query:\n"
            "  SELECT obj.objectId, fs.psfFlux, fs.psfFluxErr,\n"
            "         v.expMidptMJD, v.band\n"
            "  FROM dp1.ForcedSource AS fs\n"
            "  JOIN dp1.Visit AS v ON fs.visitId = v.visitId\n"
            "  JOIN dp1.Object AS obj ON fs.objectId = obj.objectId\n"
            "  WHERE obj.objectId IN (...)\n"
            "  ORDER BY obj.objectId, v.band, v.expMidptMJD"
        )
