# Inference Layer

Loads a trained model and converts `FeatureVector` objects into `Candidate` predictions. Parallel to the training layer — neither imports from the other.

**Consumes:** `list[FeatureVector]` + a saved model directory

**Emits:** `list[Candidate]` — one per source, with probability, confidence tier, and period

```
src/ml4em/inference/
  base.py         Predictor Protocol
  loader.py       load_model(path) → MLModel
  predictor.py    StandardPredictor
  postprocess.py  probabilities_to_candidates
```

## Contents

- [Predictor Protocol](#predictor)
- [load\_model](#load-model)
- [StandardPredictor](#standardpredictor)
- [probabilities\_to\_candidates](#probabilities-to-candidates)

---

## `Predictor` Protocol { #predictor }

The contract every predictor must satisfy.

```python
@runtime_checkable
class Predictor(Protocol):
    def predict(self, features: list[FeatureVector]) -> list[Candidate]: ...
```

---

## `load_model` { #load-model }

Reads a saved model directory and returns an `MLModel` instance. The only place in the inference layer that knows about concrete model types.

**Consumes:** Path to a saved model directory containing `manifest.json`

**Emits:** `MLModel` instance

```python
from ml4em.inference import load_model

model = load_model("models/logistic_v1/")
```

Reads `{path}/manifest.json`, finds `"model_class"`, and dispatches to that class's `@classmethod load()`:

```json
{
  "model_class": "LogisticExampleClassifier",
  "config": {"n_epochs": 300, "learning_rate": 0.01}
}
```

The model registry in `inference/loader.py` maps class names to module paths:

```python
_MODEL_REGISTRY = {
    "LogisticExampleClassifier": "ml4em.models.logistic_example",
}
```

The module is imported dynamically at call time. That avoids a circular import and keeps
the inference layer independent of specific model implementations at module load, so
importing `ml4em.inference` does not pull in torch or any other backend.

A missing `manifest.json` raises `FileNotFoundError`; an unregistered class name raises
`ValueError` listing the names that *are* registered.

To register a new model, add one entry to `_MODEL_REGISTRY`. See [Guide: Add a Model](../guides/add-model.md).

---

## `StandardPredictor` { #standardpredictor }

Model-agnostic. Runs `model.predict_proba` over slices of `cfg.inference.batch_size`
(default 10,000) to bound peak memory on large feature sets, concatenates the
probabilities, and hands them to `probabilities_to_candidates`.

**Consumes:** `list[FeatureVector]` + `MLModel`

**Emits:** `list[Candidate]` — one per input, in the same order

```python
from ml4em.inference import StandardPredictor, load_model

model = load_model("models/logistic_v1/")
predictor = StandardPredictor(model, cfg.inference)
candidates = predictor.predict(feature_vectors)
```

An empty input returns an empty list without touching the model.

---

## `probabilities_to_candidates` { #probabilities-to-candidates }

Converts raw model probabilities into `Candidate` objects. Fully implemented.

**Consumes:** `list[FeatureVector]` + `np.ndarray` of shape `(N,)` — P(positive class) per source

**Emits:** `list[Candidate]` — one per source, in the same order as input

```python
from ml4em.inference.postprocess import probabilities_to_candidates

candidates = probabilities_to_candidates(features, probs, cfg.inference)
```

Steps:

1. Assigns a confidence tier based on `cfg.inference.confidence_thresholds`
2. Copies `source_id`, `ra`, `dec`, `survey`, `period`, `period_algorithm` from each `FeatureVector` — `period` is NaN and `period_algorithm` is `""` if the period extractor did not run or failed
3. Returns one frozen `Candidate` per source

A length mismatch between `features` and `probabilities` raises `ValueError` rather than
zipping to the shorter of the two, since silently truncating would drop sources from the
output without any indication.

This module has no dependency on model internals — only on types and config — and makes
no science-case assumption. What the positive class means is decided by the labels the
model was trained on.

### Confidence tier assignment

```yaml
inference:
  confidence_thresholds:
    high:   0.9   # default
    medium: 0.7   # default
```

| Probability | Confidence |
|-------------|------------|
| ≥ `high` threshold | `"high"` |
| ≥ `medium` threshold | `"medium"` |
| below `medium` threshold | `"low"` |

Tune thresholds to match your purity/completeness requirements. Raising `high` reduces false positives; lowering `medium` increases recall at the cost of more false positives.

---

[← Models](models.md){ .md-button } [← Training](training.md){ .md-button }
