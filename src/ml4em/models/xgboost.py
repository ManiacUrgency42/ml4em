"""
XGBoost classifier for scalar light curve features.

Architecture
------------
Gradient-boosted decision tree ensemble using XGBoost.  Operates on the
same 43 scalar FeatureVector fields as DNNClassifier.  The dm/dt image
field is ignored — XGBoost cannot consume image data natively.

This model is particularly effective as a baseline and for interpretability:
XGBoost feature importance scores directly map to FeatureVector fields,
making it easy to understand which features drive WDB classification.

Model config
------------
XGBoostConfig is defined here alongside the model.  Architecture (tree
structure) hyperparameters belong in code, not in config.yaml.

Usage
-----
    from ml4em.models import XGBoostClassifier, XGBoostConfig

    cfg = XGBoostConfig(n_estimators=500, max_depth=6, learning_rate=0.05)
    model = XGBoostClassifier(config=cfg)

    trainer = StandardTrainer(model, training_cfg)
    trainer.fit(dataset)
    trainer.save("models/xgb_v1/")

    model = XGBoostClassifier.load("models/xgb_v1/")
    probs = model.predict_proba(feature_vectors)

Requires: xgboost  (pip install xgboost)
Note: xgboost is not included in any ml4em optional dep group by default.
      Add it explicitly: pip install xgboost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ml4em.types import FeatureVector
from ml4em.models.dnn import _SCALAR_FIELDS, N_SCALAR_FEATURES  # reuse field list


@dataclass
class XGBoostConfig:
    """Architecture hyperparameters for XGBoostClassifier.

    These map directly to XGBoost's XGBClassifier constructor arguments.
    See https://xgboost.readthedocs.io/en/stable/parameter.html for full docs.

    Fields
    ------
    n_estimators:
        Number of boosting rounds (trees).
    max_depth:
        Maximum tree depth.  Lower = more regularization.
    learning_rate:
        Step size shrinkage (eta).  Lower = slower learning, often better
        generalization with more estimators.
    subsample:
        Fraction of training samples used per tree.  < 1.0 = stochastic
        gradient boosting, reduces overfitting.
    colsample_bytree:
        Fraction of features used per tree.
    min_child_weight:
        Minimum sum of instance weights in a child node.
    use_gpu:
        If True, uses CUDA tree method (requires xgboost GPU build).
    """

    n_estimators     : int   = 300
    max_depth        : int   = 6
    learning_rate    : float = 0.05
    subsample        : float = 0.8
    colsample_bytree : float = 0.8
    min_child_weight : int   = 5
    use_gpu          : bool  = False


class XGBoostClassifier:
    """XGBoost gradient-boosted tree classifier for WDB detection.

    Implements the MLModel Protocol — works as a drop-in replacement for
    DNNClassifier with StandardTrainer and StandardPredictor.

    Parameters
    ----------
    config:
        Architecture hyperparameters.  Defaults to XGBoostConfig().
    """

    def __init__(self, config: Optional[XGBoostConfig] = None) -> None:
        self._config = config or XGBoostConfig()
        self._model = None   # set after fit(); type: xgboost.XGBClassifier

    # ------------------------------------------------------------------
    # Feature extraction (reuses the same field list as DNNClassifier)
    # ------------------------------------------------------------------

    @staticmethod
    def _features_to_array(features: list[FeatureVector]) -> np.ndarray:
        """Extract scalar fields from FeatureVectors into a (N, 43) array."""
        rows = []
        for fv in features:
            row = [float(getattr(fv, f, np.nan)) for f in _SCALAR_FIELDS]
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # MLModel Protocol — public interface
    # ------------------------------------------------------------------

    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray:
        """Compute P(WDB) for each source from scalar features.

        Parameters
        ----------
        features:
            One FeatureVector per source.  Only scalar fields are used.

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            WDB probability in [0, 1] for each source.
        """
        raise NotImplementedError(
            "XGBoostClassifier.predict_proba is not yet implemented.\n"
            "The model must be trained via StandardTrainer before inference.\n"
            "Next step: implement fit() in StandardTrainer and call trainer.fit(dataset)."
        )

    def save(self, path: str) -> None:
        """Save model weights and config to directory at path.

        Writes:
          {path}/model.ubj      — XGBoost model in binary JSON format
          {path}/manifest.json  — model class + config

        Parameters
        ----------
        path:
            Directory path.  Created if it does not exist.
        """
        raise NotImplementedError(
            "XGBoostClassifier.save is not yet implemented.\n"
            "Requires a trained model (self._model is not None).\n"
            "Implement after the training loop is complete."
        )

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        """Load a previously saved XGBoostClassifier from directory at path.

        Reads {path}/manifest.json for config, then loads {path}/model.ubj.

        Parameters
        ----------
        path:
            Directory written by XGBoostClassifier.save().
        """
        raise NotImplementedError(
            "XGBoostClassifier.load is not yet implemented.\n"
            "Implement after save() is finalized."
        )
