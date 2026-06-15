"""
XGBoost classifier — reference implementation of the MLModel Protocol.

This module serves as the canonical example of how to add a model to ml4em.
It is not the definitive model for any science case; the researcher chooses
the model that fits their needs and adds it following this pattern.

Pattern summary
---------------
1. Define a config dataclass (XGBoostConfig) with architecture/hyperparameters.
   - Lives here alongside the model, NOT in PipelineConfig.
   - Set in code by the researcher, not in config.yaml.
2. Define the model class implementing:
   - predict_proba(features) → np.ndarray   (satisfies MLModel Protocol)
   - save(path)                              (satisfies MLModel Protocol)
   - @classmethod load(cls, path)           (for inference/loader.py dispatch)
3. Register the class name in inference/loader.py _MODEL_REGISTRY.

To add your own model (e.g. a PyTorch DNN, a CNN, a transformer):
  cp models/xgboost.py models/my_model.py  and edit from there.

Architecture
------------
Gradient-boosted decision tree ensemble using XGBoost.  Operates on the
43 scalar FeatureVector fields defined in models/base.SCALAR_FIELDS.
The dm/dt image (FeatureVector.dmdt) is ignored — XGBoost cannot consume
2-D image data natively.

Usage
-----
    from ml4em.models import XGBoostClassifier, XGBoostConfig

    cfg = XGBoostConfig(n_estimators=500, max_depth=6, learning_rate=0.05)
    model = XGBoostClassifier(config=cfg)

    trainer = StandardTrainer(model, pipeline_cfg.training)
    trainer.fit(dataset)
    trainer.save("models/xgb_v1/")

    model = XGBoostClassifier.load("models/xgb_v1/")
    probs = model.predict_proba(feature_vectors)

Requires: xgboost  (pip install xgboost)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ml4em.models.base import SCALAR_FIELDS, features_to_array
from ml4em.types import FeatureVector


@dataclass
class XGBoostConfig:
    """Architecture and tree hyperparameters for XGBoostClassifier.

    Set in code, not in config.yaml.  Training loop params (early stopping
    rounds, eval metric) are derived from PipelineConfig.training.patience.

    See https://xgboost.readthedocs.io/en/stable/parameter.html for docs.

    Fields
    ------
    n_estimators:
        Number of boosting rounds (trees).
    max_depth:
        Maximum tree depth.  Lower = more regularization.
    learning_rate:
        Step size shrinkage (eta).
    subsample:
        Fraction of training samples per tree.  < 1.0 reduces overfitting.
    colsample_bytree:
        Fraction of features used per tree.
    min_child_weight:
        Minimum sum of instance weights required in a child node.
    use_gpu:
        If True, sets tree_method="gpu_hist" (requires XGBoost GPU build).
    """

    n_estimators     : int   = 300
    max_depth        : int   = 6
    learning_rate    : float = 0.05
    subsample        : float = 0.8
    colsample_bytree : float = 0.8
    min_child_weight : int   = 5
    use_gpu          : bool  = False


class XGBoostClassifier:
    """XGBoost gradient-boosted tree classifier.

    Reference implementation of the MLModel Protocol.  Use this as a
    template when adding a new model type to ml4em.

    Parameters
    ----------
    config:
        Architecture hyperparameters.  Defaults to XGBoostConfig().
    """

    def __init__(self, config: Optional[XGBoostConfig] = None) -> None:
        self._config = config or XGBoostConfig()
        self._model = None   # xgboost.XGBClassifier; set after fit()

    # ------------------------------------------------------------------
    # Input preparation  (scalar features only — image branch not used)
    # ------------------------------------------------------------------

    def _to_array(self, features: list[FeatureVector]) -> np.ndarray:
        """Extract SCALAR_FIELDS into a (N, 43) float32 array."""
        return features_to_array(features)

    # ------------------------------------------------------------------
    # MLModel Protocol
    # ------------------------------------------------------------------

    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray:
        """Compute P(positive class) for each source from scalar features.

        Parameters
        ----------
        features:
            One FeatureVector per source.  Only SCALAR_FIELDS are used;
            the dmdt image is ignored.

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            Class probability in [0, 1] for each source.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet (self._model is None).
        """
        if self._model is None:
            raise RuntimeError(
                "XGBoostClassifier has not been trained.\n"
                "Call StandardTrainer(model, cfg).fit(dataset) first."
            )
        X = self._to_array(features)
        # XGBClassifier.predict_proba returns shape (N, 2); column 1 = P(positive)
        return self._model.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: str) -> None:
        """Save model weights and config to directory at path.

        Writes:
          {path}/model.ubj      XGBoost binary JSON model
          {path}/manifest.json  {"model_class": "XGBoostClassifier", "config": {...}}

        Parameters
        ----------
        path:
            Directory path.  Created if it does not exist.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if self._model is None:
            raise RuntimeError(
                "Cannot save: XGBoostClassifier has not been trained."
            )
        os.makedirs(path, exist_ok=True)
        self._model.save_model(os.path.join(path, "model.ubj"))
        manifest = {
            "model_class": "XGBoostClassifier",
            "config": asdict(self._config),
        }
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        """Load a previously saved XGBoostClassifier.

        Reads {path}/manifest.json to reconstruct config, then loads
        {path}/model.ubj into a new XGBClassifier.

        Parameters
        ----------
        path:
            Directory written by XGBoostClassifier.save().
        """
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required to load an XGBoostClassifier.\n"
                "Install with: pip install xgboost"
            ) from exc

        manifest_path = os.path.join(path, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        config = XGBoostConfig(**manifest["config"])
        instance = cls(config=config)
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(os.path.join(path, "model.ubj"))
        return instance
