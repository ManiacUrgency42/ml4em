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
        Returns a (N,) float32 array: P(positive class) for each source.
        """
        X = features_to_array(features)   # (N, 45) float32
        # or use dmdt: stack [fv.dmdt for fv in features]  → (N, 26, 26)
        return self._model.predict(X).astype(np.float32)   # shape (N,)

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
- Output: `np.ndarray` of shape `(N,)`, dtype float32
- Each value is P(positive class) in `[0, 1]`
- Same order as the input; one entry per input, no filtering

A single column, not a per-class matrix. `probabilities_to_candidates` and
`InferenceConfig.confidence_thresholds` both operate on one probability per source, and
for a binary problem the second column is redundant. If your backend returns `(N, 2)`,
take `[:, 1]`.

**`manifest.json` is required.** `load_model` reads it to know which class to
instantiate. Use the exact class name as the string. Anything `load()` needs to
reconstruct the model — a serialised config, for example — should go in the manifest too;
see `LogisticExampleClassifier.save()` for the pattern.

## Step 3 — Register in the loader

Add one entry to `src/ml4em/inference/loader.py`:

```python
_MODEL_REGISTRY: dict[str, str] = {
    "LogisticExampleClassifier": "ml4em.models.logistic_example",
    "MyModel": "ml4em.models.my_model",   # add this
}
```

## Step 4 — Use it

Swap one import and one constructor in your training script:

```python
# Before:
# from ml4em.models import LogisticExampleClassifier, LogisticExampleConfig
# model = LogisticExampleClassifier(config=LogisticExampleConfig(n_epochs=300))

# After:
from ml4em.models.my_model import MyModel, MyModelConfig
model = MyModel(config=MyModelConfig(hidden_dim=512))

# Everything else is unchanged:
trainer = StandardTrainer(model, cfg.training)
trainer.fit(dataset)
trainer.save("models/my_model_v1/")
```

`StandardTrainer.fit` currently raises `NotImplementedError` — see
[Training](../layers/training.md#standardtrainer). Until it is filled in, give your model
its own `fit()` and call it directly, as `LogisticExampleClassifier` does.

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

**Missing values:** `features_to_array` preserves `np.nan`, and maps the optional Gaia
fields' `None` to NaN as well. Imputation is your model's decision. Tree ensembles handle
NaN natively; a neural net does not, and `LogisticExampleClassifier` zeroes them with
`np.nan_to_num` before fitting.

**Using the dmdt image:** a scalar-only model ignores `dmdt`; gradient-boosted trees have
no spatial awareness, so the flattened histogram would just be 676 weak columns. A CNN can
use it:
```python
images = np.stack([fv.dmdt for fv in features], axis=0)   # (N, 26, 26)
```
`fv.dmdt` is `None` when `compute_dmdt` was off or the band was too short, so filter or
substitute a zero image before stacking.

**One row per (source, band):** `FeaturePipeline` emits a separate `FeatureVector` for
each band of a source. A model sees them as independent rows. If you want a per-source
prediction, aggregate the per-band probabilities afterwards, and remember that
`source_id` is not unique across rows.

**Class imbalance:** if `dataset.positive_fraction()` is small (< 10%), consider
weighting the loss function or a class-weighting option in your backend. The training
layer exposes `dataset.class_counts()` for this purpose, and it generalises to more than
two classes where `positive_fraction()` does not.
