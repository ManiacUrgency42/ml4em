"""Model definitions and the MLModel Protocol.

Model implementations
---------------------
DNNClassifier     Fully-connected MLP on scalar features (PyTorch)
XGBoostClassifier Gradient-boosted trees on scalar features

Adding a new model
------------------
Create models/my_model.py with:
  - MyModelConfig  dataclass  (architecture hyperparameters)
  - MyModel        class      (implements predict_proba + save + classmethod load)

The class automatically satisfies MLModel without registration.

Usage
-----
    from ml4em.models import DNNClassifier, DNNConfig, MLModel
"""

from .base import MLModel
from .dnn import DNNClassifier, DNNConfig, N_SCALAR_FEATURES, _SCALAR_FIELDS
from .xgboost import XGBoostClassifier, XGBoostConfig

__all__ = [
    "MLModel",
    "DNNClassifier",
    "DNNConfig",
    "XGBoostClassifier",
    "XGBoostConfig",
    "N_SCALAR_FEATURES",
    "_SCALAR_FIELDS",
]
