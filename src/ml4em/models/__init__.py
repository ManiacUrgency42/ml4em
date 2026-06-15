"""Model definitions and the MLModel Protocol.

This package defines the contract (MLModel) and provides a reference
implementation (XGBoostClassifier) showing how to add a new model.

There is no canonical model for ml4em — the researcher chooses the
architecture that fits their science case and adds it here following
the XGBoost pattern.

Reference implementation
------------------------
XGBoostClassifier   Gradient-boosted trees on scalar features

Shared utilities
----------------
SCALAR_FIELDS       Ordered list of scalar FeatureVector field names (43)
N_SCALAR_FEATURES   len(SCALAR_FIELDS)
features_to_array   Extract SCALAR_FIELDS → np.ndarray (N, 43)

Adding a new model
------------------
See models/xgboost.py for the full pattern.  In short:
  1. Create models/my_model.py with MyModelConfig + MyModel
  2. Implement predict_proba(), save(), classmethod load()
  3. Register in inference/loader.py _MODEL_REGISTRY
"""

from .base import SCALAR_FIELDS, MLModel, N_SCALAR_FEATURES, features_to_array
from .xgboost import XGBoostClassifier, XGBoostConfig

__all__ = [
    # Protocol
    "MLModel",
    # Scalar utilities
    "SCALAR_FIELDS",
    "N_SCALAR_FEATURES",
    "features_to_array",
    # Reference implementation
    "XGBoostClassifier",
    "XGBoostConfig",
]
