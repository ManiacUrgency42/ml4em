"""
FeaturePipeline — compose extractors into a FeatureVector.

The pipeline:
1. Validates min_observations against the primary band.
2. Runs each extractor in order, collecting partial feature dicts.
3. Merges all dicts and constructs a FeatureVector.
4. Returns a default (all-NaN) FeatureVector if the source has too few points.

Usage
-----
    from ml4em.features import FeaturePipeline
    from ml4em.features.statistics import StatisticsExtractor
    from ml4em.features.period import PeriodExtractor
    from ml4em.features.dmdt import DmdtExtractor
    from ml4em.features.catalog import CatalogExtractor
    from ml4em.config import load_config

    cfg = load_config()
    pipeline = FeaturePipeline.default(cfg.features)

    # Single source
    fv = pipeline.run(lcs)

    # Batch
    fvs = pipeline.run_batch(all_lcs_grouped_by_source)
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


class FeaturePipeline:
    """Compose a list of FeatureExtractors into a single FeatureVector.

    Parameters
    ----------
    extractors:
        Ordered list of extractors to run.  Results are merged left-to-right;
        later extractors can overwrite keys from earlier ones (rarely needed).
    min_observations:
        Sources with fewer observations than this in the primary band are
        skipped — a default (all-NaN) FeatureVector is returned.
    compute_dmdt:
        If False, the dm/dt extractor (if present) is still in the list but
        its output is discarded.  This flag is checked here so callers can
        pass the DmdtExtractor without branching logic.

    Notes
    -----
    FeatureExtractors are called in the order they appear in `extractors`.
    Each extractor receives the full list of LightCurves for the source so
    it can choose the band(s) it needs internally.
    """

    def __init__(
        self,
        extractors: list[FeatureExtractor],
        min_observations: int = 50,
        compute_dmdt: bool = True,
    ) -> None:
        self._extractors = extractors
        self._min_obs = min_observations
        self._compute_dmdt = compute_dmdt

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def default(cls, config: FeatureConfig) -> "FeaturePipeline":
        """Build the standard pipeline from a FeatureConfig.

        Extractor order: statistics → period → dmdt → catalog.
        This order matters: statistics fills n_obs first (used for logging),
        period runs before dmdt so the period window does not affect the image.
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
        )

    # ------------------------------------------------------------------
    # Core run logic
    # ------------------------------------------------------------------

    def _primary_n_obs(self, lcs: list[LightCurve]) -> int:
        """Return the observation count of the band with the most points."""
        return max(lc.n_obs for lc in lcs) if lcs else 0

    def _merge_features(
        self, partial_dicts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Merge extractor outputs left-to-right into a single flat dict."""
        merged: dict[str, Any] = {}
        for d in partial_dicts:
            merged.update(d)
        return merged

    def _build_feature_vector(
        self,
        lcs: list[LightCurve],
        features: dict[str, Any],
    ) -> FeatureVector:
        """Construct a FeatureVector from the merged feature dict.

        Only keys that correspond to FeatureVector fields are used;
        extra keys are silently dropped.  Missing keys remain at their
        FeatureVector defaults (np.nan / None / "").
        """
        primary = max(lcs, key=lambda lc: lc.n_obs)
        valid_fields = {f.name for f in dataclasses.fields(FeatureVector)}

        kwargs: dict[str, Any] = {
            "source_id": primary.source_id,
            "survey":    primary.survey,
        }
        for key, val in features.items():
            if key in valid_fields:
                kwargs[key] = val

        # Drop dmdt if the pipeline was configured not to compute it
        if not self._compute_dmdt:
            kwargs.pop("dmdt", None)

        return FeatureVector(**kwargs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, lcs: list[LightCurve]) -> FeatureVector:
        """Extract features for all bands of one source.

        Parameters
        ----------
        lcs:
            All LightCurves for a single source (one per band).
            Must share the same source_id.

        Returns
        -------
        FeatureVector
            Fully populated feature vector.  Fields that could not be
            computed are np.nan / None / "".
            If the primary band has fewer than min_observations points, a
            default (all-NaN) FeatureVector is returned immediately.
        """
        if not lcs:
            raise ValueError("lcs must contain at least one LightCurve")

        primary = max(lcs, key=lambda lc: lc.n_obs)

        if primary.n_obs < self._min_obs:
            return FeatureVector(
                source_id=primary.source_id,
                survey=primary.survey,
                n_obs=primary.n_obs,
            )

        partial_dicts = [extractor.extract(lcs) for extractor in self._extractors]
        features = self._merge_features(partial_dicts)
        return self._build_feature_vector(lcs, features)

    def run_batch(
        self, grouped_lcs: list[list[LightCurve]]
    ) -> list[FeatureVector]:
        """Extract features for multiple sources.

        Parameters
        ----------
        grouped_lcs:
            List of groups; each group is all LightCurves for one source
            (i.e. one element per source, each element is a list of bands).

        Returns
        -------
        list[FeatureVector]
            One FeatureVector per source, in the same order.
        """
        return [self.run(lcs) for lcs in grouped_lcs]
