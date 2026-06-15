"""
PyTorch MLP classifier for scalar light curve features.

Architecture
------------
A fully-connected network (MLP) that operates on the 43 scalar features
in FeatureVector.  The dm/dt image field is intentionally ignored —
add a separate CNN model (e.g. models/cnn.py) to handle image inputs,
or a hybrid model (models/hybrid.py) that fuses both branches.

Model config
------------
DNNConfig is defined here alongside the model, not in WDBConfig.
Architecture hyperparameters belong in code, not in config.yaml.
Training loop hyperparameters (lr, batch_size, epochs) live in
WDBConfig.training (TrainingConfig) and are passed to StandardTrainer.

Usage
-----
    from ml4em.models import DNNClassifier, DNNConfig

    cfg = DNNConfig(hidden_dims=[512, 256, 128], dropout=0.2)
    model = DNNClassifier(n_scalars=43, config=cfg)

    # training
    trainer = StandardTrainer(model, training_cfg)
    trainer.fit(dataset)
    trainer.save("models/dnn_v1/")

    # inference
    model = DNNClassifier.load("models/dnn_v1/")
    probs = model.predict_proba(feature_vectors)

Requires: torch  (pip install "ml4em[training]" or "ml4em[inference]")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ml4em.types import FeatureVector

# Scalar field names extracted from FeatureVector for DNN input.
# Order is fixed — changing this order invalidates saved models.
_SCALAR_FIELDS: list[str] = [
    "n_obs",
    "median", "wmean", "chi2red", "roms", "wstd",
    "norm_peak_to_peak_amp", "norm_excess_var", "median_abs_dev",
    "iqr", "i60r", "i70r", "i80r", "i90r",
    "skew", "small_kurt", "inv_von_neumann",
    "stetson_i", "stetson_j", "stetson_k",
    "anderson_darling", "shapiro_wilk",
    "period", "period_significance",
    "f1_power", "f1_bic", "f1_a", "f1_b", "f1_amp", "f1_phi0",
    "f1_relamp1", "f1_relphi1",
    "f1_relamp2", "f1_relphi2",
    "f1_relamp3", "f1_relphi3",
    "f1_relamp4", "f1_relphi4",
    "gaia_parallax", "gaia_parallax_error", "gaia_bp_rp", "gaia_ruwe",
]

N_SCALAR_FEATURES: int = len(_SCALAR_FIELDS)  # 43


@dataclass
class DNNConfig:
    """Architecture hyperparameters for DNNClassifier.

    These are set in code, not in config.yaml.  Training loop params
    (lr, batch_size, epochs) live separately in TrainingConfig.

    Fields
    ------
    hidden_dims:
        Width of each hidden layer, in order.  The network is:
        input(n_scalars) → hidden[0] → hidden[1] → ... → output(1)
    dropout:
        Dropout rate applied after each hidden layer.  Set 0.0 to disable.
    activation:
        Activation function name.  Currently supported: "relu", "gelu".
    """

    hidden_dims : list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout     : float     = 0.3
    activation  : str       = "relu"


class DNNClassifier:
    """Fully-connected MLP classifier for WDB detection from scalar features.

    Wraps a torch.nn.Module and implements the MLModel Protocol so it can
    be used with StandardTrainer and StandardPredictor without changes.

    Parameters
    ----------
    n_scalars:
        Number of scalar input features.  Use N_SCALAR_FEATURES (43) for
        the full FeatureVector scalar set.
    config:
        Architecture hyperparameters.  Defaults to DNNConfig().
    """

    def __init__(
        self,
        n_scalars: int = N_SCALAR_FEATURES,
        config: Optional[DNNConfig] = None,
    ) -> None:
        self._n_scalars = n_scalars
        self._config = config or DNNConfig()
        self._net = None   # built lazily on first access of self.module

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_network(self):
        """Construct the torch.nn.Sequential MLP."""
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "torch is required for DNNClassifier.\n"
                "Install with: pip install 'ml4em[training]'"
            ) from exc

        try:
            act_fn = {"relu": nn.ReLU, "gelu": nn.GELU}[self._config.activation]
        except KeyError:
            raise ValueError(
                f"Unknown activation '{self._config.activation}'. "
                f"Supported: 'relu', 'gelu'."
            )

        layers = []
        in_dim = self._n_scalars
        for out_dim in self._config.hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                act_fn(),
                nn.Dropout(self._config.dropout),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))   # single logit → sigmoid for P(WDB)
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Feature extraction from FeatureVector
    # ------------------------------------------------------------------

    @staticmethod
    def _features_to_array(features: list[FeatureVector]) -> "np.ndarray":
        """Extract scalar fields from FeatureVectors into a (N, 43) array.

        NaN values (uncomputed features) are preserved — the caller or
        a preprocessing step is responsible for imputation before training.
        """
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
            One FeatureVector per source.  Only scalar fields are used;
            the dmdt image is ignored by this model.

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            WDB probability in [0, 1] for each source.
        """
        raise NotImplementedError(
            "DNNClassifier.predict_proba is not yet implemented.\n"
            "The network architecture (forward pass) must be finalized and "
            "the model must be trained before inference can run.\n"
            "Next step: implement forward() and train via StandardTrainer."
        )

    def save(self, path: str) -> None:
        """Save model weights and config to directory at path.

        Writes:
          {path}/weights.pt     — torch state_dict
          {path}/manifest.json  — model class + architecture config

        Parameters
        ----------
        path:
            Directory path.  Created if it does not exist.
        """
        raise NotImplementedError(
            "DNNClassifier.save is not yet implemented.\n"
            "Requires a trained model (state_dict) to serialize.\n"
            "Implement after the training loop is complete."
        )

    @classmethod
    def load(cls, path: str) -> "DNNClassifier":
        """Load a previously saved DNNClassifier from directory at path.

        Reads {path}/manifest.json for architecture config, then loads
        {path}/weights.pt into the reconstructed network.

        Parameters
        ----------
        path:
            Directory written by DNNClassifier.save().
        """
        raise NotImplementedError(
            "DNNClassifier.load is not yet implemented.\n"
            "Implement after save() is finalized."
        )

    # ------------------------------------------------------------------
    # Torch module access (for training loop)
    # ------------------------------------------------------------------

    @property
    def module(self):
        """Return the underlying torch.nn.Sequential for use in training loops.

        Builds the network on first access (lazy init) so that importing
        DNNClassifier does not require torch to be installed.
        """
        if self._net is None:
            self._net = self._build_network()
        return self._net
