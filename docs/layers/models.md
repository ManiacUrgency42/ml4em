# Models Layer

Defines the contract between training and inference. The models layer does not perform training or inference — it provides the `MLModel` Protocol that both layers depend on, plus utilities for extracting scalar features from `FeatureVector` objects.

**Consumes:** `list[FeatureVector]` (via `predict_proba`)
**Emits:** `np.ndarray` of shape `(N, 2)` — per-class probabilities

```
src/ml4em/models/
  base.py       MLModel Protocol + SCALAR_FIELDS utilities
  xgboost.py    XGBoostClassifier — reference implementation
```

## Contents

- [MLModel Protocol](#mlmodel)
- [SCALAR\_FIELDS and features\_to\_array](#scalar-fields)
- [XGBoostClassifier](#xgboostclassifier)

---

## `MLModel` Protocol { #mlmodel }

The contract every model must satisfy. Any class with compatible `predict_proba` and `save` methods is a valid `MLModel`.

**Consumes:** `list[FeatureVector]`
**Emits:** `np.ndarray` shape `(N, 2)` — `[:, 0]` is P(background), `[:, 1]` is P(positive class)

```python
class MLModel(Protocol):
    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
```

`save` must write a `manifest.json` containing at least `{"model_class": "ClassName"}` so that `inference.load_model` can reconstruct the model. `load` is not on the Protocol because backends serialize differently (`joblib` for XGBoost, `torch.save` for neural nets) — dispatch lives in `inference/loader.py`.

---

## `SCALAR_FIELDS` and `features_to_array` { #scalar-fields }

Defined in `models/base.py`. Shared by all model implementations that operate on scalar features.

### `SCALAR_FIELDS`

An ordered list of 43 float field names from `FeatureVector` — everything except `source_id`, `survey`, `period_algorithm`, and `dmdt`.

```python
from ml4em.models import SCALAR_FIELDS, N_SCALAR_FEATURES

print(N_SCALAR_FEATURES)   # 43
print(SCALAR_FIELDS[:5])   # ['n_obs', 'median', 'wmean', 'chi2red', 'roms']
```

!!! warning "Field order is fixed"
    The ordering of `SCALAR_FIELDS` is stable across versions. A model trained on one ordering cannot be used with a different ordering. Never reorder `SCALAR_FIELDS` without retraining.

### `features_to_array`

```python
from ml4em.models import features_to_array

X = features_to_array(feature_vectors)   # np.ndarray, shape (N, 43), dtype float32
```

Extracts the 43 scalar fields in `SCALAR_FIELDS` order. `np.nan` values are preserved — XGBoost handles them natively.

---

## `XGBoostClassifier` { #xgboostclassifier }

The reference implementation of `MLModel`. Demonstrates the correct pattern for implementing a new model. It is not the committed model for any science case — treat it as a template.

**Consumes:** `list[FeatureVector]` — uses the 43 scalar fields only (ignores `dmdt`)
**Emits:** `np.ndarray` shape `(N, 2)` — class probabilities

```python
from ml4em.models import XGBoostClassifier
from ml4em.models.xgboost import XGBoostConfig

model = XGBoostClassifier(config=XGBoostConfig(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
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

### `XGBoostConfig` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 500 | Number of boosting rounds (trees) |
| `max_depth` | 6 | Maximum depth of each tree |
| `learning_rate` | 0.05 | Step size shrinkage (eta) |
| `subsample` | 0.8 | Fraction of training samples per tree |
| `colsample_bytree` | 0.8 | Fraction of features sampled per tree |
| `min_child_weight` | 1 | Minimum sum of weights in a leaf |
| `use_gpu` | `False` | Use GPU training (`tree_method="gpu_hist"`) |

See [Guide: Add a Model](../guides/add-model.md) to add a new model backend.

---

[← Features](features.md){ .md-button } [Training →](training.md){ .md-button .md-button--primary } [Inference →](inference.md){ .md-button .md-button--primary }
