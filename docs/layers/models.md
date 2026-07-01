# Models Layer

Defines the contract between training and inference. The models layer does not perform training or inference — it provides the `MLModel` Protocol that both layers depend on, plus utilities for extracting scalar features from `FeatureVector` objects.

**Consumes:** `list[FeatureVector]` (via `predict_proba`)

**Emits:** `np.ndarray` of shape `(N, 2)` — per-class probabilities

```
src/ml4em/models/
  base.py       MLModel Protocol + SCALAR_FIELDS utilities
```

## Contents

- [MLModel Protocol](#mlmodel)
- [SCALAR\_FIELDS and features\_to\_array](#scalar-fields)

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

`save` must write a `manifest.json` containing at least `{"model_class": "ClassName"}` so that `inference.load_model` can reconstruct the model. `load` is not on the Protocol because backends serialize differently — dispatch lives in `inference/loader.py`.

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

Extracts the 43 scalar fields in `SCALAR_FIELDS` order. `np.nan` values are preserved (models that handle missing data natively, such as gradient boosting trees, can use them directly).

---

[← Features](features.md){ .md-button } [Training →](training.md){ .md-button .md-button--primary } [Inference →](inference.md){ .md-button .md-button--primary }
