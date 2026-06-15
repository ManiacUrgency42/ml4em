"""ml4em — machine learning for EM light curve analysis."""

from . import constants
from .types import FeatureVector, LightCurve, WDBCandidate

__version__ = "0.1.0"

__all__ = [
    "LightCurve",
    "FeatureVector",
    "WDBCandidate",
    "constants",
]
