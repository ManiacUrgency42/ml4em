"""
Protocol defining the feature extractor interface.

Any object with a compatible extract() signature is a valid FeatureExtractor.
No base class or registration required — structural typing via Protocol.

Design contract
---------------
- extract() receives a batch of sources, each as a list of LightCurves (one
  per band).
- It returns one dict per source, mapping FeatureVector field names to values.
- Keys absent from a dict leave the FeatureVector field at its default
  (np.nan for floats, None for Optional, "" for strings).
- extract() must never raise — catch exceptions internally and return a list
  of empty dicts so the pipeline can continue.
- prepare() is optional.  The pipeline calls it once with the *complete*
  source list before chunking, so an extractor can derive any quantity that
  must be identical across chunks.  PeriodExtractor uses it to build the
  frequency grid from the full-field baseline; without it the grid would be
  rebuilt per chunk and the same star would get different periods depending
  on which sources happened to share its batch.

Batch-first design
------------------
All extractors operate on a list of sources and delegate to periodfind in a
single batched call.  Passing a single source is just a batch of one:
    extractor.extract([lcs])[0]

Adding a new extractor
----------------------
Define a class with a compatible extract() signature.  Pass it to
FeaturePipeline and it will be called automatically.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ml4em.types import LightCurve


def to_float32_time(time: np.ndarray) -> np.ndarray:
    """Zero-offset a time array in float64, then cast to float32.

    Every extractor hands times to periodfind as float32.  Casting an absolute
    epoch directly is silently destructive: at HJD ~2.459e6 one float32 ULP is
    0.25 days, so *every* pair of observations less than six hours apart
    collapses to a separation of exactly zero.  Intra-night structure — the
    entire short-period regime — disappears without any error being raised.

    Subtracting the earliest epoch in float64 first moves the values into
    [0, baseline], where the ULP scales with the baseline rather than with
    the zero point: ~5 s at 1000 days, ~10 s over a full ZTF DR16 baseline.
    That is coarse enough to move a pair between adjacent dt bins at the
    short end of the dm/dt grid, but it preserves the intra-night structure
    that casting the absolute epoch destroys outright.  Absolute epoch is
    not something any feature here depends on; only differences matter.

    This applies to any survey whose time system has a large zero point
    (JD/HJD/BJD ~2.4e6).  MJD ~6e4 is less severe but still loses precision, so
    the offset is applied unconditionally.
    """
    t64 = np.asarray(time, dtype=np.float64)
    if not t64.size:
        return t64.astype(np.float32)
    # min() propagates a single NaN across the whole array, so one bad epoch
    # would blank every epoch.  LightCurve deliberately does not drop
    # non-finite values (see types.py), so they can reach here; nanmin keeps
    # the damage local to the offending index.
    finite = t64[np.isfinite(t64)]
    if not finite.size:
        return t64.astype(np.float32)
    return (t64 - finite.min()).astype(np.float32)


@runtime_checkable
class FeatureExtractor(Protocol):
    """Contract every feature extractor must satisfy.

    extract() is the only required member, so isinstance() against this
    protocol agrees with what the pipeline actually demands.  prepare() is
    deliberately not declared: it is optional, the pipeline probes for it
    with getattr, and declaring it here would make every extractor that
    does not need it — including both of the ones shipped in this package —
    fail its own isinstance() check.
    """

    def extract(
        self, sources: list[list[LightCurve]]
    ) -> list[dict[str, Any]]:
        """Compute features for a batch of sources.

        Parameters
        ----------
        sources:
            List of sources; each element is all LightCurves for one source
            (one per band).  Extractors select the band(s) they need internally.

        Returns
        -------
        list[dict[str, Any]]
            One flat dict per source mapping FeatureVector field names to values.
            Return a list of empty dicts (not raise) on failure.
        """
        ...
