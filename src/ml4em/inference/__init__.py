"""Inference pipeline for ml4em.

Components
----------
Predictor           Protocol — any object with predict()
StandardPredictor   Concrete predictor for any MLModel
load_model          Load a saved MLModel from disk by manifest
postprocess         probabilities_to_candidates utility

Usage
-----
    from ml4em.inference import StandardPredictor, load_model

    model = load_model("models/dnn_v1/")
    predictor = StandardPredictor(model, cfg.inference)
    candidates = predictor.predict(feature_vectors)
"""

from .base import Predictor
from .loader import load_model
from .postprocess import probabilities_to_candidates
from .predictor import StandardPredictor

__all__ = [
    "Predictor",
    "StandardPredictor",
    "load_model",
    "probabilities_to_candidates",
]
