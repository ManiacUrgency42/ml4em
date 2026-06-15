"""
Protocol defining the data source interface.

Any object that implements fetch() and fetch_batch() is a valid
LightCurveSource — no base class required.  This enables plug-and-play
swapping between ZTF, Rubin, simulated, and future data sources without
touching the feature or training layers.

Usage
-----
    from ml4em.data import ZTFSource
    from ml4em.data.base import LightCurveSource

    source: LightCurveSource = ZTFSource(cfg.sources.ztf, token)
    lcs = source.fetch_batch(my_ids)

To add a new data source, implement a class with fetch() and fetch_batch()
matching the signatures below.  No registration or base class needed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ml4em.types import LightCurve


@runtime_checkable
class LightCurveSource(Protocol):
    """Contract every data source must satisfy.

    Structural Protocol — any class with compatible fetch() and
    fetch_batch() methods is automatically a LightCurveSource.

    Each source returns LightCurve objects — one per band per sky position.
    For ZTF, each source _id already encodes a single band, so fetch() on
    one ID returns a one-element list.  For Rubin, one objectId may have
    data in up to six bands, so fetch() may return up to six LightCurves.
    """

    def fetch(self, source_id: str) -> list[LightCurve]:
        """Fetch all available bands for a single source.

        Parameters
        ----------
        source_id:
            Survey-native identifier represented as a string.
            ZTF: integer _id cast to str (e.g. "1234567890").
            Rubin: dp1.Object.objectId cast to str.
            Simulated: path to an Lcurve model file, or grid index as str.

        Returns
        -------
        list[LightCurve]
            One LightCurve per band that has observations.
            May be empty if the source has no clean data in any band.
        """
        ...

    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
        """Fetch light curves for multiple sources in a single network call.

        Implementations should batch the requests where the API supports it
        (e.g. Kowalski's use_batch_query=True).

        Parameters
        ----------
        source_ids:
            Survey-native identifiers, each cast to str.

        Returns
        -------
        list[LightCurve]
            All LightCurves across all requested sources and bands.
            Order is not guaranteed.
        """
        ...
