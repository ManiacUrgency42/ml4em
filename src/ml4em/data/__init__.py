"""Data source implementations for ml4em.

Plug-and-play data sources via the LightCurveSource Protocol.
Any object implementing fetch() and fetch_batch() is a valid source.

Sources
-------
ZTFSource        Real ZTF photometry via Kowalski (penquins)
RubinSource      Rubin DP1 photometry via TAP (pyvo) — stub
SimulatedSource  Synthetic WDB light curves via Lcurve — stub

Usage
-----
    from ml4em.data import ZTFSource
    from ml4em.config import load_config, get_ztf_token

    cfg = load_config()
    source = ZTFSource(cfg.sources.ztf, get_ztf_token())
    lcs = source.fetch_batch(my_source_ids)
"""

from .base import LightCurveSource
from .rubin import RubinSource
from .simulation import SimulatedSource
from .ztf import ZTFSource

__all__ = [
    "LightCurveSource",
    "ZTFSource",
    "RubinSource",
    "SimulatedSource",
]
