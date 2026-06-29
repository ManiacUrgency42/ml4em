"""
Logistic regression in PyTorch — end-to-end pipeline demo.

This module shows how to wire a PyTorch model into the ml4em MLModel
Protocol so it runs through the full feature → train → inference path.
It is intentionally minimal: one linear layer, BCEWithLogitsLoss, Adam.

Use this to prove the pipeline is wired correctly on MSI before committing
to a more expressive architecture (DNN, CNN + dm/dt image, etc.).

What "logistic regression" means here
--------------------------------------
A single nn.Linear(N_SCALAR_FEATURES, 1) layer followed by sigmoid is
exactly logistic regression.  During training we pass raw logits to
BCEWithLogitsLoss (numerically stable).  At inference time we apply
sigmoid to obtain P(positive) ∈ [0, 1].

Why this is a useful demo
--------------------------
- The weight vector (42 numbers) is directly interpretable: large |w_i|
  means feature i is driving the prediction.  Print or plot them after
  training to see which features the model finds informative.
- Any subsequent model (DNN, XGBoost) should beat this accuracy — it
  gives you a concrete baseline.
- Training completes in seconds on any hardware.

Pattern summary
---------------
1. LogisticExampleConfig — hyperparameters (n_epochs, learning_rate).
   Lives here, not in PipelineConfig.
2. _LogisticModule — the bare nn.Module (one linear layer).
3. LogisticExampleClassifier —
   - fit(features, labels)       train on labeled FeatureVectors
   - predict_proba(features)     satisfies MLModel Protocol
   - save(path)                  satisfies MLModel Protocol
   - classmethod load(cls, path) for inference/loader.py dispatch

Usage
-----
    from ml4em.models import LogisticExampleClassifier, LogisticExampleConfig

    cfg = LogisticExampleConfig(n_epochs=300, learning_rate=1e-2)
    model = LogisticExampleClassifier(config=cfg)

    # Training (called directly — StandardTrainer.fit is not yet implemented)
    model.fit(feature_vectors, labels)

    # Inspect learned weights to see which features matter
    weights = model.weights()   # dict[feature_name → float]

    # Round-trip through the inference layer
    from ml4em.inference.loader import load_model
    model.save("models/logistic_v1/")
    loaded = load_model("models/logistic_v1/")
    probs = loaded.predict_proba(feature_vectors)   # shape (N,), float32

Requires: torch (in the [training] optional dependency group)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ml4em.models.base import SCALAR_FIELDS, N_SCALAR_FEATURES, features_to_array
from ml4em.types import FeatureVector


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LogisticExampleConfig:
    """Training hyperparameters for LogisticExampleClassifier.

    Fields
    ------
    n_epochs:
        Number of full passes over the training set.
        200–500 is usually enough for convergence on this feature scale.
    learning_rate:
        Adam step size.  1e-2 converges quickly; lower if loss is unstable.
    """

    n_epochs      : int   = 300
    learning_rate : float = 1e-2


# ---------------------------------------------------------------------------
# PyTorch module  (kept private — callers only interact with the classifier)
# ---------------------------------------------------------------------------

class _LogisticModule:
    """One linear layer.  Not an nn.Module subclass to avoid a hard torch
    import at module load time — torch is imported lazily inside fit/load."""

    def __init__(self, n_features: int) -> None:
        import torch.nn as nn
        import torch

        class _Net(nn.Module):
            def __init__(self, n: int) -> None:
                super().__init__()
                self.linear = nn.Linear(n, 1)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                # Returns raw logits; sigmoid applied at inference time
                return self.linear(x).squeeze(1)

        self._net = _Net(n_features)

    # Expose the underlying nn.Module so the classifier can call .parameters(),
    # .state_dict(), .load_state_dict(), .train(), .eval(), and forward().
    @property
    def net(self):
        return self._net


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class LogisticExampleClassifier:
    """Logistic regression classifier — PyTorch pipeline demo.

    Wraps a single nn.Linear layer to satisfy the MLModel Protocol.
    Operates on the 42 scalar SCALAR_FIELDS from each FeatureVector;
    the dm/dt image is not used.

    NaN values (sources below min_observations, or features that failed
    to compute) are replaced with 0.0 before fitting and inference.
    This is intentional for a demo — use proper imputation in production.

    Parameters
    ----------
    config:
        Training hyperparameters.  Defaults to LogisticExampleConfig().
    """

    def __init__(self, config: Optional[LogisticExampleConfig] = None) -> None:
        self._config = config or LogisticExampleConfig()
        self._module: Optional[_LogisticModule] = None   # set after fit() or load()

    # ------------------------------------------------------------------
    # Training  (outside the MLModel Protocol — called directly for demo)
    # ------------------------------------------------------------------

    def fit(self, features: list[FeatureVector], labels: list[int]) -> None:
        """Train on labeled FeatureVectors using BCEWithLogitsLoss + Adam.

        Parameters
        ----------
        features:
            One FeatureVector per source.  Same format as predict_proba.
        labels:
            Binary labels (0 = negative, 1 = positive), one per source.
            Must be the same length as features.

        Notes
        -----
        NaN scalar values are zeroed before fitting.  The training loop
        runs for config.n_epochs full passes over the data — there is no
        early stopping.  For a demo dataset this converges cleanly; for
        production, add a validation split and early stopping.
        """
        import torch
        import torch.nn as nn

        X = torch.tensor(
            np.nan_to_num(features_to_array(features), nan=0.0),
            dtype=torch.float32,
        )
        y = torch.tensor(labels, dtype=torch.float32)

        self._module = _LogisticModule(N_SCALAR_FEATURES)
        net = self._module.net

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self._config.learning_rate
        )
        criterion = nn.BCEWithLogitsLoss()

        net.train()
        for _ in range(self._config.n_epochs):
            optimizer.zero_grad()
            logits = net(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        net.eval()

    # ------------------------------------------------------------------
    # MLModel Protocol
    # ------------------------------------------------------------------

    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray:
        """Compute P(positive class) for each source from scalar features.

        Parameters
        ----------
        features:
            One FeatureVector per source.  Only SCALAR_FIELDS are used.

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            Class probability in [0, 1] for each source.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if self._module is None:
            raise RuntimeError(
                "LogisticExampleClassifier has not been trained.\n"
                "Call model.fit(feature_vectors, labels) first, or load a\n"
                "saved model with inference.loader.load_model(path)."
            )

        import torch

        net = self._module.net
        X = torch.tensor(
            np.nan_to_num(features_to_array(features), nan=0.0),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = net(X)
            probs = torch.sigmoid(logits)

        return probs.numpy().astype(np.float32)

    def save(self, path: str) -> None:
        """Save model weights and config to a directory.

        Writes:
          {path}/weights.pt       PyTorch state_dict (CPU tensors)
          {path}/manifest.json    {"model_class": "LogisticExampleClassifier",
                                   "config": {...}}

        Parameters
        ----------
        path:
            Directory path.  Created if it does not exist.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if self._module is None:
            raise RuntimeError(
                "Cannot save: LogisticExampleClassifier has not been trained."
            )

        import torch

        os.makedirs(path, exist_ok=True)
        torch.save(
            self._module.net.state_dict(),
            os.path.join(path, "weights.pt"),
        )
        manifest = {
            "model_class": "LogisticExampleClassifier",
            "config": asdict(self._config),
        }
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LogisticExampleClassifier":
        """Load a previously saved LogisticExampleClassifier.

        Reads {path}/manifest.json to reconstruct config, then loads
        {path}/weights.pt into a fresh _LogisticModule.

        Parameters
        ----------
        path:
            Directory written by LogisticExampleClassifier.save().
        """
        import torch

        manifest_path = os.path.join(path, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        config = LogisticExampleConfig(**manifest["config"])
        instance = cls(config=config)
        instance._module = _LogisticModule(N_SCALAR_FEATURES)
        instance._module.net.load_state_dict(
            torch.load(
                os.path.join(path, "weights.pt"),
                map_location="cpu",
                weights_only=True,
            )
        )
        instance._module.net.eval()
        return instance

    # ------------------------------------------------------------------
    # Interpretability helper
    # ------------------------------------------------------------------

    def weights(self) -> dict[str, float]:
        """Return the learned weight for each scalar feature.

        Useful for understanding which features drive the model's predictions.
        Large positive weight → feature pushes toward positive class.
        Large negative weight → feature pushes toward negative class.

        Returns
        -------
        dict[str, float]
            Mapping from SCALAR_FIELDS name to learned weight value.
            Ordered by descending absolute weight (most important first).

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if self._module is None:
            raise RuntimeError(
                "No weights available: model has not been trained."
            )

        import torch

        with torch.no_grad():
            w = self._module.net.linear.weight.squeeze(0).numpy()

        weight_map = dict(zip(SCALAR_FIELDS, w.tolist()))
        return dict(sorted(weight_map.items(), key=lambda kv: abs(kv[1]), reverse=True))
