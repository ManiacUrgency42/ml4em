# Models Layer

Defines the contract between training and inference. The models layer does not perform training or inference — it provides the `MLModel` Protocol that both layers depend on, plus utilities for extracting scalar features from `FeatureVector` objects.

**Consumes:** `list[FeatureVector]` (via `predict_proba`)

**Emits:** `np.ndarray` of shape `(N,)` — P(positive class) per source

```
src/ml4em/models/
  base.py              MLModel Protocol + SCALAR_FIELDS utilities
  logistic_example.py  LogisticExampleClassifier — reference implementation
```

## Contents

- [MLModel Protocol](#mlmodel)
- [SCALAR\_FIELDS and features\_to\_array](#scalar-fields)
- [LogisticExampleClassifier](#logistic-example)

---

## `MLModel` Protocol { #mlmodel }

The contract every model must satisfy. Any class with compatible `predict_proba` and `save` methods is a valid `MLModel` — no base class, no registration.

**Consumes:** `list[FeatureVector]`

**Emits:** `np.ndarray` shape `(N,)`, dtype float32 — P(positive class) in `[0, 1]`, in the same order as the input

```python
@runtime_checkable
class MLModel(Protocol):
    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
```

`predict_proba` receives `list[FeatureVector]` rather than a pre-flattened array so each model can extract exactly the fields it needs — scalars, the dm/dt image, or both.

`save` must write a `manifest.json` containing at least `{"model_class": "ClassName"}` so that `inference.load_model` can reconstruct the model. `load` is not on the Protocol because backends serialize differently — dispatch lives in `inference/loader.py`, which maps the manifest's class name to a module via `_MODEL_REGISTRY`.

`fit` is not on the Protocol either. Training signatures differ per backend, and both the training and inference layers only ever need to *use* a model.

---

## `SCALAR_FIELDS` and `features_to_array` { #scalar-fields }

Defined in `models/base.py` rather than inside any model, because they describe the `FeatureVector` contract and not one model's internals.

### `SCALAR_FIELDS`

An ordered list of **45** field names from `FeatureVector` whose values are plain floats,
or ints castable to float. It excludes `source_id`, `survey`, `band`, `ra`, `dec`, the
string fields `period_algorithm` and `period_family_algorithm`, the `period_top` /
`significance_top` dicts, and the `dmdt` image.

`ra` and `dec` are excluded deliberately: they identify where a source is, not how it
varies, and a model given them can learn the survey footprint instead of the physics.

```python
from ml4em.models import SCALAR_FIELDS, N_SCALAR_FEATURES

print(N_SCALAR_FEATURES)   # 45
print(SCALAR_FIELDS[:5])   # ['n_obs', 'median', 'wmean', 'chi2red', 'roms']
```

The list is validated against `FeatureVector` at import. A name that the dataclass does
not declare would be invisible at runtime — `features_to_array()` would fall back to NaN
and every model would train on a dead column — so the mismatch raises immediately.

!!! warning "Field order is fixed"
    The ordering of `SCALAR_FIELDS` is stable across versions. A model trained on one ordering cannot be used with a different ordering. Never reorder `SCALAR_FIELDS` without retraining.

### `features_to_array`

```python
from ml4em.models import features_to_array

X = features_to_array(feature_vectors)   # np.ndarray, shape (N, 45), dtype float32
```

Extracts the 45 scalar fields in `SCALAR_FIELDS` order. `np.nan` is preserved, so models
that handle missing data natively (gradient boosting trees, for instance) can use it
directly; imputation is the model's decision, not this function's.

The Gaia fields are `Optional[float]` and are `None` — not absent — whenever a source has
no catalogue counterpart, so `float(None)` would raise. `features_to_array` maps `None` to
NaN, because an unmatched source is missing data rather than a zero.

---

## `LogisticExampleClassifier` { #logistic-example }

A deliberately minimal reference implementation: one `nn.Linear(45, 1)` layer,
`BCEWithLogitsLoss`, Adam. That is exactly logistic regression. It exists to prove the
feature → train → inference path is wired correctly before committing to a more
expressive architecture, and to give any later model a concrete accuracy baseline to
beat.

```python
from ml4em.models import LogisticExampleClassifier, LogisticExampleConfig

model = LogisticExampleClassifier(LogisticExampleConfig(n_epochs=300, learning_rate=1e-2))
model.fit(feature_vectors, labels)
probs = model.predict_proba(feature_vectors)   # shape (N,), float32

model.save("models/logistic_v1/")              # weights.pt + manifest.json
```

`LogisticExampleConfig` lives in this module, not in `PipelineConfig` — model
architecture hyperparameters are code, not configuration. See
[Design Principles](../architecture/design-principles.md#2-code-controls-architecture-config-controls-parameters).

`weights()` returns the learned weight per `SCALAR_FIELDS` name, sorted by descending
absolute value. With 45 numbers and one layer, that mapping is directly readable: a large
`|w|` means the model is leaning on that feature.

NaN scalars are zeroed before fitting and before prediction. The training loop runs a
fixed `n_epochs` full-batch passes with no validation split and no early stopping, which
is fine for a demo dataset and not what you want in production.

`torch` is imported lazily inside `fit`, `predict_proba` and `load`, so importing
`ml4em.models` does not require it. It ships in the `[training]` optional dependency
group.

---

[← Features](features.md){ .md-button } [Training →](training.md){ .md-button .md-button--primary } [Inference →](inference.md){ .md-button .md-button--primary }
