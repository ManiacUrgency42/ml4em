"""
StandardTrainer — training loop for any MLModel.

The trainer is model-agnostic.  It accepts any object satisfying the
MLModel Protocol and drives the training loop using TrainingConfig for
loop hyperparameters (lr, batch_size, max_epochs, patience).

Architecture-specific decisions (hidden_dims, dropout, n_estimators)
live in the model's own config (DNNConfig, XGBoostConfig), not here.

Usage
-----
    from ml4em.models import DNNClassifier, DNNConfig
    from ml4em.training import FeatureDataset, StandardTrainer
    from ml4em.config import load_config

    cfg = load_config("config.yaml")
    model = DNNClassifier(n_scalars=43, config=DNNConfig())
    dataset = FeatureDataset.from_storage(cfg.storage, "labels.csv")
    train, val, _ = dataset.split(cfg.training.val_fraction,
                                  cfg.training.test_fraction,
                                  cfg.training.seed)

    trainer = StandardTrainer(model, cfg.training)
    trainer.fit(train, val)
    trainer.save("models/run_v1/")

Status
------
fit() is a stub — the training loop depends on the model type (PyTorch vs
XGBoost have different training APIs) and must be implemented once the
model architecture is finalized.

save() is fully implemented — it delegates to model.save().
"""

from __future__ import annotations

import os

from ml4em.config.schema import TrainingConfig
from ml4em.models.base import MLModel
from ml4em.training.dataset import FeatureDataset


class StandardTrainer:
    """Train any MLModel on a labeled FeatureDataset.

    Parameters
    ----------
    model:
        Any object satisfying the MLModel Protocol.
        Pass DNNClassifier, XGBoostClassifier, or your own model.
    config:
        Training loop hyperparameters from PipelineConfig.training.
        Controls lr, batch_size, max_epochs, patience — NOT model architecture.
    """

    def __init__(self, model: MLModel, config: TrainingConfig) -> None:
        self._model = model
        self._cfg = config

    # ------------------------------------------------------------------
    # Trainer Protocol — public interface
    # ------------------------------------------------------------------

    def fit(self, dataset: FeatureDataset) -> None:
        """Train the model on a labeled dataset.

        Splits the dataset into train/val using config.val_fraction and
        config.test_fraction, then runs the training loop with early stopping
        based on config.patience.

        Parameters
        ----------
        dataset:
            Full labeled dataset.  Internally split into train/val/test.

        Notes
        -----
        Implementation depends on the model type:
        - PyTorch models (DNNClassifier): use torch.optim.Adam, BCEWithLogitsLoss,
          DataLoader, epoch loop with validation, early stopping.
        - XGBoost models (XGBoostClassifier): call xgb.fit() with eval_set for
          early stopping — no manual epoch loop needed.

        A dispatch pattern or model-type check is needed here, or the
        training loop can be moved onto the model itself (model.fit(dataset, cfg)).
        """
        raise NotImplementedError(
            "StandardTrainer.fit is not yet implemented.\n\n"
            "Implementation notes:\n"
            "  1. Split dataset: train, val, test = dataset.split(\n"
            "         self._cfg.val_fraction, self._cfg.test_fraction, self._cfg.seed)\n"
            "  2. For PyTorch models:\n"
            "       - Extract scalar arrays via DNNClassifier._features_to_array()\n"
            "       - Build DataLoader from (X_train, y_train)\n"
            "       - Optimizer: Adam(model.module.parameters(), lr=self._cfg.learning_rate)\n"
            "       - Loss: BCEWithLogitsLoss()\n"
            "       - Epoch loop with val loss tracking + early stopping\n"
            "  3. For XGBoost models:\n"
            "       - Extract scalar arrays\n"
            "       - Call model._model.fit(X_train, y_train, eval_set=[(X_val, y_val)],\n"
            "                               early_stopping_rounds=self._cfg.patience)\n"
            "  4. Log train/val loss per epoch (use Python logging, not print).\n"
            "  5. Store best model weights for save()."
        )

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Delegates to self._model.save(path).

        Parameters
        ----------
        path:
            Directory path.  Created if it does not exist.
        """
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
