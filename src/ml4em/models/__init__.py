"""Model definitions and the MLModel Protocol.

This package defines the contract (MLModel) and provides a reference
implementation (LogisticExampleClassifier) showing how to wire any
estimator into the ml4em pipeline.

There is no canonical model for ml4em — the researcher chooses the
architecture that fits their science case and adds it here following
the logistic_example pattern.

Reference implementation
------------------------
LogisticExampleClassifier   Logistic regression on scalar features (demo)

Shared utilities
----------------
SCALAR_FIELDS       Ordered list of scalar FeatureVector field names (42)
N_SCALAR_FEATURES   len(SCALAR_FIELDS)
features_to_array   Extract SCALAR_FIELDS → np.ndarray (N, 42)

Adding a new model
------------------
See models/logistic_example.py for the full pattern.  In short:
  1. Create models/my_model.py with MyModelConfig + MyModel
  2. Implement predict_proba(), save(), classmethod load()
  3. Register in inference/loader.py _MODEL_REGISTRY
"""

from .base import SCALAR_FIELDS, MLModel, N_SCALAR_FEATURES, features_to_array
from .logistic_example import LogisticExampleClassifier, LogisticExampleConfig

__all__ = [
    # Protocol
    "MLModel",
    # Scalar utilities
    "SCALAR_FIELDS",
    "N_SCALAR_FEATURES",
    "features_to_array",
    # Reference implementation
    "LogisticExampleClassifier",
    "LogisticExampleConfig",
]
