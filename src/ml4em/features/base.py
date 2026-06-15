"""
Protocol defining the feature extractor interface.

Any object that implements extract(lcs) -> dict is a valid FeatureExtractor.
No base class or registration required — structural typing via Protocol.

Design contract
---------------
- extract() receives ALL bands for one source as a list of LightCurves.
- It returns a flat dict mapping FeatureVector field names to computed values.
- Keys that are absent from the dict leave the FeatureVector field at its
  default (np.nan for floats, None for Optional, "" for strings).
- extract() must never raise — catch exceptions internally and return an
  empty dict (or partial dict) so the pipeline can continue.

Adding a new extractor
----------------------
Define a class with a compatible extract() signature.  Pass it to
FeaturePipeline and it will be called automatically.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ml4em.types import LightCurve


@runtime_checkable
class FeatureExtractor(Protocol):
    """Contract every feature extractor must satisfy."""

    def extract(self, lcs: list[LightCurve]) -> dict[str, Any]:
        """Compute features for one source.

        Parameters
        ----------
        lcs:
            All LightCurves available for this source — one per band.
            Extractors select the band(s) they need internally.

        Returns
        -------
        dict[str, Any]
            Flat mapping of FeatureVector field names to computed values.
            Return an empty dict (not raise) on failure.
        """
        ...
