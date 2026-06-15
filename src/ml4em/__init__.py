"""ml4em — machine learning for EM light curve analysis."""

from . import constants
from .types import Candidate, FeatureVector, LabeledSample, LightCurve

__version__ = "0.1.0"

__all__ = [
    "LightCurve",
    "FeatureVector",
    "LabeledSample",
    "Candidate",
    "constants",
]
