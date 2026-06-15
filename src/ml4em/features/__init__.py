"""Feature extraction submodules for ml4em.

Each submodule is a self-contained extractor with its own preprocessing,
feature generation, and (where applicable) post-processing.  All extractors
implement the FeatureExtractor Protocol from base.py.

Submodules
----------
StatisticsExtractor  22 scalar light curve variability statistics
PeriodExtractor      Best period + 14 Fourier harmonic coefficients
DmdtExtractor        26×26 Δmag/Δt pairwise histogram (image feature)
CatalogExtractor     4 Gaia EDR3 astrometric/photometric features (stub)

Pipeline
--------
FeaturePipeline composes the extractors in the correct order and produces
a FeatureVector from their merged outputs.

Usage
-----
    from ml4em.features import FeaturePipeline
    from ml4em.config import load_config

    cfg = load_config()
    pipeline = FeaturePipeline.default(cfg.features)
    feature_vector = pipeline.run(lcs)           # list[LightCurve] → FeatureVector
    feature_vectors = pipeline.run_batch(groups) # list[list[LightCurve]] → list[FeatureVector]
"""

from .base import FeatureExtractor
from .catalog import CatalogExtractor
from .dmdt import DmdtExtractor
from .period import PeriodExtractor
from .pipeline import FeaturePipeline
from .statistics import StatisticsExtractor

__all__ = [
    "FeatureExtractor",
    "StatisticsExtractor",
    "PeriodExtractor",
    "DmdtExtractor",
    "CatalogExtractor",
    "FeaturePipeline",
]
