"""
Protocol defining the model interface.

Any class with predict_proba() and save() is a valid MLModel — no base
class or registration required.  Structural typing via Protocol.

Design
------
MLModel is the shared contract between the training and inference layers.
The training layer produces an MLModel (via Trainer.fit()).
The inference layer consumes an MLModel (via Predictor.predict()).
Neither layer imports from the other.

Adding a new model
------------------
1. Create models/my_model.py with MyModelConfig (dataclass) and MyModel.
2. Implement predict_proba() and save() / classmethod load().
3. That's it — MyModel automatically satisfies MLModel without registration.

load() is NOT on this Protocol
-------------------------------
torch.load and joblib.load work differently; each model loads itself via
a classmethod.  The inference/loader.py load_model() function dispatches
to the right class by reading manifest.json written by save().
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ml4em.types import FeatureVector


@runtime_checkable
class MLModel(Protocol):
    """Contract every classifier must satisfy.

    Structural Protocol — any class with compatible predict_proba() and
    save() methods is automatically an MLModel.

    Notes
    -----
    - predict_proba() receives list[FeatureVector] so each model can
      extract exactly the fields it needs (scalars, dmdt image, or both).
    - save() must write a manifest.json alongside model weights so that
      inference/loader.py can reconstruct the model without knowing its type.
    """

    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray:
        """Compute WDB probability for each source.

        Parameters
        ----------
        features:
            One FeatureVector per source.  Each model extracts the
            fields it needs internally (scalars, dmdt image, or both).

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            P(WDB) in [0, 1] for each source, in the same order as input.
        """
        ...

    def save(self, path: str) -> None:
        """Serialize model weights and config to a directory.

        Implementations must write a manifest.json at {path}/manifest.json
        containing at minimum {"model_class": "<ClassName>"} so that
        inference.loader.load_model() can reconstruct the model.

        Parameters
        ----------
        path:
            Directory path.  Created if it does not exist.
        """
        ...
