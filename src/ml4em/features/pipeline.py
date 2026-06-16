"""
FeaturePipeline — compose extractors into a batch of FeatureVectors.

The pipeline:
1. Sets the periodfind device (CPU/GPU) once before processing.
2. Chunks the source list into batches of feature_batch_size.
3. Within each batch, partitions sources by min_observations.
4. Calls each extractor once for the valid sources in the batch.
5. Merges extractor outputs and constructs FeatureVectors.
6. Returns a default (all-NaN) FeatureVector for skipped sources.

Usage
-----
    from ml4em.features import FeaturePipeline
    from ml4em.config import load_config

    cfg = load_config()
    pipeline = FeaturePipeline.default(cfg.features)

    fvs = pipeline.run_batch(all_lcs_grouped_by_source)

Single-source use (e.g. tests):
    fv = pipeline.run_batch([lcs])[0]
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ml4em.config.schema import FeatureConfig
from ml4em.types import FeatureVector, LightCurve

from .base import FeatureExtractor
from .catalog import CatalogExtractor
from .dmdt import DmdtExtractor
from .period import PeriodExtractor
from .statistics import StatisticsExtractor


def _chunks(lst: list, size: int):
    """Yield successive chunks of `size` from `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


class FeaturePipeline:
    """Compose a list of FeatureExtractors into FeatureVectors.

    Parameters
    ----------
    extractors:
        Ordered list of extractors to run.  Results are merged left-to-right.
    min_observations:
        Sources with fewer observations than this in the primary band are
        skipped — a default (all-NaN) FeatureVector is returned.
    compute_dmdt:
        If False, the dmdt key is dropped from the final FeatureVector even
        if a DmdtExtractor is present.
    device:
        periodfind device — 'cpu', 'gpu', or 'auto'.
        Set once before the first batch and reused for all subsequent calls.
    batch_size:
        Number of sources per periodfind batch call.
        Lower this if GPU runs out of memory on long light curves.
    """

    def __init__(
        self,
        extractors: list[FeatureExtractor],
        min_observations: int = 50,
        compute_dmdt: bool = True,
        device: str = "auto",
        batch_size: int = 1000,
    ) -> None:
        self._extractors = extractors
        self._min_obs = min_observations
        self._compute_dmdt = compute_dmdt
        self._device = device
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def default(cls, config: FeatureConfig) -> "FeaturePipeline":
        """Build the standard pipeline from a FeatureConfig.

        Extractor order: statistics → period → dmdt → catalog.
        """
        extractors: list[FeatureExtractor] = [
            StatisticsExtractor(),
            PeriodExtractor(config.period),
        ]
        if config.compute_dmdt:
            extractors.append(DmdtExtractor(config.dmdt))
        extractors.append(CatalogExtractor(config.catalog))

        return cls(
            extractors=extractors,
            min_observations=config.min_observations,
            compute_dmdt=config.compute_dmdt,
            device=config.device,
            batch_size=config.feature_batch_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _primary_n_obs(self, lcs: list[LightCurve]) -> int:
        return max(lc.n_obs for lc in lcs) if lcs else 0

    def _build_feature_vector(
        self,
        lcs: list[LightCurve],
        features: dict[str, Any],
    ) -> FeatureVector:
        primary = max(lcs, key=lambda lc: lc.n_obs)
        valid_fields = {f.name for f in dataclasses.fields(FeatureVector)}

        kwargs: dict[str, Any] = {
            "source_id": primary.source_id,
            "survey"   : primary.survey,
            "ra"       : primary.ra,
            "dec"      : primary.dec,
        }
        for key, val in features.items():
            if key in valid_fields:
                kwargs[key] = val

        if not self._compute_dmdt:
            kwargs.pop("dmdt", None)

        return FeatureVector(**kwargs)

    def _default_fv(self, lcs: list[LightCurve]) -> FeatureVector:
        primary = max(lcs, key=lambda lc: lc.n_obs)
        return FeatureVector(
            source_id=primary.source_id,
            survey=primary.survey,
            ra=primary.ra,
            dec=primary.dec,
            n_obs=primary.n_obs,
        )

    def _process_chunk(
        self, chunk: list[list[LightCurve]]
    ) -> list[FeatureVector]:
        """Process one chunk of sources through all extractors."""
        valid_mask = [self._primary_n_obs(lcs) >= self._min_obs for lcs in chunk]

        valid_sources  = [lcs for lcs, ok in zip(chunk, valid_mask) if ok]
        valid_positions = [i for i, ok in enumerate(valid_mask) if ok]

        # Build per-source merged feature dicts for valid sources only
        merged: list[dict[str, Any]] = [{} for _ in valid_sources]
        if valid_sources:
            for extractor in self._extractors:
                partial = extractor.extract(valid_sources)
                for i, d in enumerate(partial):
                    merged[i].update(d)

        # Assemble final FeatureVectors in original chunk order
        fvs: list[FeatureVector | None] = [None] * len(chunk)
        valid_iter = iter(zip(valid_positions, valid_sources, merged))

        for pos, lcs, features in valid_iter:
            fvs[pos] = self._build_feature_vector(lcs, features)

        for i, (lcs, ok) in enumerate(zip(chunk, valid_mask)):
            if not ok:
                fvs[i] = self._default_fv(lcs)

        return fvs  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_batch(
        self,
        grouped_lcs: list[list[LightCurve]],
        batch_size: int | None = None,
    ) -> list[FeatureVector]:
        """Extract features for multiple sources.

        Parameters
        ----------
        grouped_lcs:
            Each element is all LightCurves for one source (one per band).
        batch_size:
            Override the instance batch_size for this call.

        Returns
        -------
        list[FeatureVector]
            One FeatureVector per source, in the same order as grouped_lcs.
        """
        if not grouped_lcs:
            return []

        import periodfind
        periodfind.set_device(self._device)
        size = batch_size or self._batch_size

        results: list[FeatureVector] = []
        for chunk in _chunks(grouped_lcs, size):
            results.extend(self._process_chunk(chunk))

        return results
