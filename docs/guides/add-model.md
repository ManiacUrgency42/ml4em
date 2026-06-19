# Guide: Add a New Model

Adding a new ML model requires **one new file** plus a one-line registration. Training,
inference, and postprocessing are unchanged.

---

## Step 1 — Create the file

```
src/ml4em/models/my_model.py
```

## Step 2 — Implement the class

A valid `MLModel` needs three things: `predict_proba`, `save`, and `@classmethod load`.

```python
import json
import os
import numpy as np
from dataclasses import dataclass
from ml4em.types import FeatureVector
from ml4em.models import features_to_array

@dataclass
class MyModelConfig:
    hidden_dim: int = 256
    dropout: float = 0.3
    # model architecture parameters go here
    # loop parameters (lr, epochs) go in cfg.training — NOT here

class MyModel:
    def __init__(self, config: MyModelConfig = None):
        self.config = config or MyModelConfig()
        self._model = self._build()

    def _build(self):
        # build your model here (PyTorch, scikit-learn, etc.)
        ...

    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray:
        """
        Returns (N, 2) array: column 0 = P(background), column 1 = P(positive).
        """
        X = features_to_array(features)   # (N, 43) float32
        # or use dmdt: stack [fv.dmdt for fv in features]  → (N, 26, 26)
        probs = self._model.predict_proba(X)
        return probs   # shape (N, 2)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # save model weights
        self._save_weights(path)
        # REQUIRED: write manifest so load_model can reconstruct this class
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump({"model_class": "MyModel"}, f)

    @classmethod
    def load(cls, path: str) -> "MyModel":
        model = cls()
        # load model weights
        model._load_weights(path)
        return model
```

**`predict_proba` contract:**
- Input: `list[FeatureVector]` — arbitrary length
- Output: `np.ndarray` of shape `(N, 2)`, dtype float
- Column 0 = probability of background class
- Column 1 = probability of positive class
- Probabilities must sum to 1 per row

**`manifest.json` is required.** `load_model` reads it to know which class to
instantiate. Use the exact class name as the string.

## Step 3 — Register in the loader

Add one entry to `src/ml4em/inference/loader.py`:

```python
_MODEL_REGISTRY: dict[str, str] = {
    "XGBoostClassifier": "ml4em.models.xgboost",
    "MyModel": "ml4em.models.my_model",   # add this
}
```

## Step 4 — Use it

Swap one import and one constructor in your training script:

```python
# Before:
# from ml4em.models import XGBoostClassifier, XGBoostConfig
# model = XGBoostClassifier(config=XGBoostConfig(n_estimators=500))

# After:
from ml4em.models.my_model import MyModel, MyModelConfig
model = MyModel(config=MyModelConfig(hidden_dim=512))

# Everything else is unchanged:
trainer = StandardTrainer(model, cfg.training)
trainer.fit(dataset)
trainer.save("models/my_model_v1/")
```

Loading at inference time:

```python
from ml4em.inference import load_model

model = load_model("models/my_model_v1/")   # reads manifest, calls MyModel.load()
predictor = StandardPredictor(model, cfg.inference)
candidates = predictor.predict(feature_vectors)
```

---

## Notes

**SCALAR_FIELDS ordering:** `features_to_array` produces columns in the exact order
of `models.SCALAR_FIELDS`. If you train with one version of `SCALAR_FIELDS` and then
change it, your saved model will produce wrong predictions on the new ordering. Always
retrain after changing `SCALAR_FIELDS`.

**Using the dmdt image:** `XGBoostClassifier` ignores `dmdt` because gradient-boosted
trees don't have spatial awareness. A CNN can use it:
```python
images = np.stack([fv.dmdt for fv in features], axis=0)   # (N, 26, 26)
```

**Class imbalance:** if `dataset.positive_fraction()` is small (< 10%), consider
weighting the loss function or using `scale_pos_weight` (XGBoost) / `class_weight`
(scikit-learn). The training layer exposes `dataset.class_counts()` for this purpose.
