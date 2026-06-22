"""
Protocol defining the data source interface.

Any object that implements fetch_batch() is a valid LightCurveSource —
no base class required.  This enables plug-and-play swapping between ZTF,
Rubin, simulated, and future data sources without touching the feature or
training layers.

Usage
-----
    from ml4em.data import ZTFSource
    from ml4em.data.base import LightCurveSource

    source: LightCurveSource = ZTFSource(cfg.sources.ztf, token)
    lcs = source.fetch_batch(my_ids)

    # Single source: just pass a one-element list
    lcs = source.fetch_batch([single_id])

To add a new data source, implement a class with fetch_batch() matching
the signature below.  No registration or base class needed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ml4em.types import LightCurve


@runtime_checkable
class LightCurveSource(Protocol):
    """Contract every data source must satisfy.

    Structural Protocol — any class with a compatible fetch_batch() method
    is automatically a LightCurveSource.

    Each source returns LightCurve objects — one per band per sky position.
    For ZTF, each source _id encodes a single band.  For Rubin, one
    objectId may have data in up to six bands.  For a single source, pass
    a one-element list: fetch_batch([source_id]).
    """

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
