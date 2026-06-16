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

from ml4em.types import LightCurve


@runtime_checkable
class FeatureExtractor(Protocol):
    """Contract every feature extractor must satisfy."""

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
