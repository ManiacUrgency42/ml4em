"""
Protocol defining the predictor interface.

Any object with predict() is a valid Predictor — no base class required.
Structural typing via Protocol, consistent with the pattern used across
data/base.py, features/base.py, and training/base.py.

Design
------
The Predictor is the inference layer's public interface.  It consumes
FeatureVectors and produces Candidate results.  It does not import from
the training layer.

Adding a new predictor
----------------------
Define a class with predict() matching the signature below.
No registration needed — it automatically satisfies Predictor.

Example: a batched GPU predictor, an ensemble predictor that combines
multiple MLModels, or a streaming predictor for real-time survey alerts
would all implement this Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ml4em.types import Candidate, FeatureVector


@runtime_checkable
class Predictor(Protocol):
    """Contract every predictor must satisfy.

    Structural Protocol — any class with a compatible predict() method
    is automatically a Predictor.
    """

    def predict(self, features: list[FeatureVector]) -> list[Candidate]:
        """Run inference and return classification results.

        Parameters
        ----------
        features:
            One FeatureVector per source.

        Returns
        -------
        list[Candidate]
            One Candidate per source, in the same order as input.
            Probability and confidence fields are filled by the model and
            postprocessor.
        """
        ...
