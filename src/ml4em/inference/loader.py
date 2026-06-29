"""
Model loader — reconstruct any saved MLModel from disk.

Design
------
Each MLModel.save() writes a manifest.json alongside the model weights:

    {path}/manifest.json   {"model_class": "LogisticExampleClassifier"}
    {path}/weights.pt      (PyTorch state_dict)

load_model() reads the manifest, resolves the class, and delegates to
that class's classmethod load().

This is the only place in the inference layer that knows about specific
model types.  StandardPredictor and the Predictor Protocol work entirely
with the MLModel interface — they never import a concrete model class.

Adding a new model
------------------
Register the class name in _MODEL_REGISTRY below.  That's the only change
needed — the rest of the inference layer is unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml4em.models.base import MLModel


# Map class name (as written in manifest.json) to the importable class.
# Update this dict when adding a new model to models/.
_MODEL_REGISTRY: dict[str, str] = {
    "LogisticExampleClassifier": "ml4em.models.logistic_example",
}


def load_model(path: str) -> MLModel:
    """Load a previously saved MLModel from a directory.

    Reads {path}/manifest.json to determine the model class, then calls
    that class's classmethod load(path) to reconstruct the model.

    Parameters
    ----------
    path:
        Directory written by an MLModel.save() call.
        Must contain manifest.json with a "model_class" key.

    Returns
    -------
    MLModel
        The reconstructed model, ready for predict_proba().

    Raises
    ------
    FileNotFoundError
        If path or manifest.json does not exist.
    KeyError
        If manifest.json does not contain "model_class".
    ValueError
        If the model class is not in _MODEL_REGISTRY.
    """
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found at {manifest_path}.\n"
            "The model directory must be written by MLModel.save()."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    class_name = manifest.get("model_class")
    if class_name is None:
        raise KeyError(
            f"'model_class' key missing from {manifest_path}.\n"
            f"Found keys: {list(manifest.keys())}"
        )

    module_path = _MODEL_REGISTRY.get(class_name)
    if module_path is None:
        raise ValueError(
            f"Unknown model class '{class_name}'.\n"
            f"Registered classes: {list(_MODEL_REGISTRY.keys())}\n"
            "To add a new model, register it in inference/loader.py _MODEL_REGISTRY."
        )

    # Dynamic import — avoids circular imports and keeps inference layer
    # independent of specific model implementations at module load time.
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls.load(path)
