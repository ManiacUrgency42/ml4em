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
3. Register the class name in inference/loader.py _MODEL_REGISTRY.
4. That's it — MyModel automatically satisfies MLModel without registration.

See models/logistic_example.py for a reference implementation.

load() is NOT on this Protocol
-------------------------------
Each model loads itself differently (torch state_dict, custom serialisation).
The inference/loader.py load_model() function dispatches to the right class
by reading manifest.json written by save().

Scalar field list
-----------------
SCALAR_FIELDS and N_SCALAR_FEATURES are defined here (not inside any
model) because they describe the FeatureVector contract, not a specific
model's internals.  Any model that consumes scalar features uses these.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np

from ml4em.types import FeatureVector


# ---------------------------------------------------------------------------
# Scalar feature field list
#
# Ordered list of FeatureVector field names whose values are plain floats
# (or ints castable to float).  Excludes: source_id, survey, ra, dec,
# period_algorithm (str), and dmdt (ndarray image).
#
# Order is stable across releases — changing it invalidates saved models
# that were trained on this ordering.
# ---------------------------------------------------------------------------

SCALAR_FIELDS: list[str] = [
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

N_SCALAR_FEATURES: int = len(SCALAR_FIELDS)   # 42


def features_to_array(features: list[FeatureVector]) -> np.ndarray:
    """Extract SCALAR_FIELDS from a list of FeatureVectors into a 2-D array.

    Parameters
    ----------
    features:
        One FeatureVector per source.

    Returns
    -------
    np.ndarray, shape (N, N_SCALAR_FEATURES), dtype float32
        Row i corresponds to features[i].  NaN is preserved for any field
        that was not computed.  Imputation (if needed) is the model's
        responsibility.
    """
    rows = [
        [float(getattr(fv, f, np.nan)) for f in SCALAR_FIELDS]
        for fv in features
    ]
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# MLModel Protocol
# ---------------------------------------------------------------------------

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
        """Compute P(positive class) for each source.

        Parameters
        ----------
        features:
            One FeatureVector per source.  Each model extracts the
            fields it needs internally (scalars, dmdt image, or both).

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            Probability in [0, 1] for each source, in the same order as input.
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
