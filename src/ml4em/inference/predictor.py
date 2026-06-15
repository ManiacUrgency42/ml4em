"""
StandardPredictor — run inference with any MLModel.

The predictor is model-agnostic.  It accepts any object satisfying MLModel,
calls predict_proba(), then passes results to postprocess for Candidate
construction.

Usage
-----
    from ml4em.inference import StandardPredictor, load_model
    from ml4em.config import load_config

    cfg = load_config("config.yaml")
    model = load_model("models/dnn_v1/")

    predictor = StandardPredictor(model, cfg.inference)
    candidates = predictor.predict(feature_vectors)

Status
------
predict() is a partial shell — the call chain is defined but depends on
model.predict_proba() being implemented.  postprocess is fully functional.
"""

from __future__ import annotations

from ml4em.config.schema import InferenceConfig
from ml4em.inference.postprocess import probabilities_to_candidates
from ml4em.models.base import MLModel
from ml4em.types import Candidate, FeatureVector


class StandardPredictor:
    """Run inference on FeatureVectors using any MLModel.

    Parameters
    ----------
    model:
        Any object satisfying the MLModel Protocol.
        Typically loaded via inference.loader.load_model(path).
    config:
        InferenceConfig from PipelineConfig.inference.
        Controls batch_size and confidence_thresholds.
    """

    def __init__(self, model: MLModel, config: InferenceConfig) -> None:
        self._model = model
        self._cfg = config

    # ------------------------------------------------------------------
    # Predictor Protocol — public interface
    # ------------------------------------------------------------------

    def predict(self, features: list[FeatureVector]) -> list[Candidate]:
        """Run inference and return classification results.

        Processes features in batches of config.batch_size to avoid
        out-of-memory errors on large feature sets.

        Parameters
        ----------
        features:
            One FeatureVector per source.

        Returns
        -------
        list[Candidate]
            One Candidate per source, in the same order as input.
        """
        if not features:
            return []

        import numpy as np

        all_probs: list[float] = []
        batch_size = self._cfg.batch_size

        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size]
            probs = self._model.predict_proba(batch)
            all_probs.extend(probs.tolist())

        return probabilities_to_candidates(
            features,
            np.array(all_probs, dtype=np.float32),
            self._cfg,
        )
