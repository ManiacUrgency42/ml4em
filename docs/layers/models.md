# Models Layer

The models layer defines the **contract** between training and inference. It does not
perform training or inference itself — it provides the Protocol that both layers depend
on, plus utilities for scalar feature extraction.

```
src/ml4em/models/
  base.py         MLModel Protocol + SCALAR_FIELDS utilities
  xgboost.py      XGBoostClassifier — reference implementation
```

---

## Protocol — `MLModel`

```python
class MLModel(Protocol):
    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
```

Any class implementing these two methods satisfies `MLModel`.

**`predict_proba(features)`** returns an `(N, 2)` array of class probabilities. Column
0 = P(background), column 1 = P(positive class). The inference layer uses column 1.

**`save(path)`** writes the model to a directory. The directory must contain a
`manifest.json` file with at least `{"model_class": "ClassName"}` so that `load_model`
knows how to reconstruct it.

**Why is `load` not on the Protocol?** Because different model backends serialize
differently (joblib for XGBoost, `torch.save` for neural nets). The dispatch lives in
`inference/loader.py` which uses the manifest to call the right `@classmethod load()`.

---

## Scalar field utilities

Defined in `models/base.py`, available to any model that operates on scalar features:

### `SCALAR_FIELDS`

An ordered list of 43 float field names from `FeatureVector` — everything except
`source_id`, `survey`, `period_algorithm`, and `dmdt`.

```python
from ml4em.models import SCALAR_FIELDS, N_SCALAR_FEATURES
print(N_SCALAR_FEATURES)   # 43
print(SCALAR_FIELDS[:5])   # ['n_obs', 'median', 'wmean', 'chi2red', 'roms']
```

!!! warning "Field order is stable — do not reorder"
    The ordering of `SCALAR_FIELDS` is fixed. A model trained on one ordering cannot
    be used with a different ordering. Changing `SCALAR_FIELDS` without retraining all
    existing models will produce silently wrong predictions.

### `features_to_array`

```python
from ml4em.models import features_to_array

X = features_to_array(feature_vectors)   # (N, 43) float32 array
```

Extracts the 43 scalar fields from a list of `FeatureVector` objects in the order
defined by `SCALAR_FIELDS`. NaN values are preserved (XGBoost handles them natively).

---

## `XGBoostClassifier` — reference implementation

`XGBoostClassifier` is the reference implementation of `MLModel`. It demonstrates the
correct pattern for a new model. It is **not** the committed model for any science case
— treat it as a template.

!!! info "What is XGBoost? (for astrophysicists)"
    XGBoost (eXtreme Gradient Boosting) is an ensemble method that trains many simple
    decision trees in sequence, where each tree corrects the errors of the previous ones.
    The result is a strong classifier built from many weak ones.

    It has no sense of sequence or spatial structure — it sees the 43 scalar features as
    a flat feature vector and learns a boundary in that 43-dimensional space. This is why
    it ignores the `dmdt` image.

### Configuration

```python
from ml4em.models import XGBoostClassifier
from ml4em.models.xgboost import XGBoostConfig

model = XGBoostClassifier(config=XGBoostConfig(
    n_estimators=500,     # number of boosting rounds (trees)
    max_depth=6,          # maximum depth of each tree
    learning_rate=0.05,   # step size shrinkage (eta)
    subsample=0.8,        # fraction of training samples per tree
    colsample_bytree=0.8, # fraction of features per tree
    min_child_weight=1,   # minimum sum of weights in a leaf
    use_gpu=False,        # True to use GPU training (tree_method="gpu_hist")
))
```

### Save and load

```python
model.save("models/xgb_v1/")
# Creates:
#   models/xgb_v1/model.ubj       (XGBoost binary format)
#   models/xgb_v1/manifest.json   ({"model_class": "XGBoostClassifier"})

model = XGBoostClassifier.load("models/xgb_v1/")
```

The `.ubj` format (Universal Binary JSON) is XGBoost's native binary format —
compact and fast to load.

---

## Adding a new model

Four steps, one new file:

**1.** Create `src/ml4em/models/my_model.py`:

```python
@dataclass
class MyModelConfig:
    hidden_dim: int = 256
    dropout: float = 0.3

class MyModel:
    def __init__(self, config: MyModelConfig = MyModelConfig()):
        self.config = config
        self._model = build_network(config)  # your architecture

    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray:
        X = features_to_array(features)      # (N, 43)
        # or use dmdt: features[i].dmdt      # (26, 26)
        return self._model.predict_proba(X)  # (N, 2)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self._model.state_dict(), f"{path}/model.pt")
        with open(f"{path}/manifest.json", "w") as f:
            json.dump({"model_class": "MyModel"}, f)

    @classmethod
    def load(cls, path: str) -> "MyModel":
        manifest = json.load(open(f"{path}/manifest.json"))
        model = cls()
        model._model.load_state_dict(torch.load(f"{path}/model.pt"))
        return model
```

**2.** Register in `src/ml4em/inference/loader.py`:

```python
_MODEL_REGISTRY = {
    "XGBoostClassifier": "ml4em.models.xgboost",
    "MyModel": "ml4em.models.my_model",      # add this
}
```

**3.** Use it — swap one import and one constructor:

```python
from ml4em.models.my_model import MyModel, MyModelConfig
model = MyModel(config=MyModelConfig(hidden_dim=512))
trainer = StandardTrainer(model, cfg.training)
```

**4.** Training, inference, and postprocessing are unchanged.

See [Guide: Add a Model](../guides/add-model.md) for step-by-step instructions.
