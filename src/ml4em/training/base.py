"""
Protocol defining the trainer interface.

Any object with fit() and save() is a valid Trainer — no base class
required.  Structural typing via Protocol, consistent with the pattern
used in data/base.py (LightCurveSource) and features/base.py (FeatureExtractor).

Design
------
The Trainer contract is intentionally minimal:
  - fit(dataset) runs the training loop and updates model weights in place.
  - save(path) persists the trained model to disk.

The model architecture and its config are passed to the concrete Trainer
implementation at construction time — not here.  This Protocol only defines
the interface that the training pipeline depends on.

Adding a new trainer
--------------------
Define a class with fit() and save() matching the signatures below.
No registration needed — it automatically satisfies Trainer.

Example: a cross-validation trainer, a distributed trainer, or a
Bayesian hyperparameter search trainer would all implement this Protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ml4em.training.dataset import FeatureDataset


@runtime_checkable
class Trainer(Protocol):
    """Contract every trainer must satisfy.

    Structural Protocol — any class with compatible fit() and save()
    methods is automatically a Trainer.
    """

    def fit(self, dataset: "FeatureDataset") -> None:
        """Train the model on a labeled dataset.

        Parameters
        ----------
        dataset:
            Labeled feature vectors split into train/val sets.
            The trainer reads TrainingConfig for loop hyperparameters
            (lr, batch_size, max_epochs, patience).
        """
        ...

    def save(self, path: str) -> None:
        """Persist the trained model to disk.

        Delegates to the underlying MLModel.save(path).

        Parameters
        ----------
        path:
            Directory path.  Created if it does not exist.
        """
        ...
