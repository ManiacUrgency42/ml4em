# Inference Layer

Loads a trained model and converts `FeatureVector` objects into `Candidate` predictions. Parallel to the training layer — neither imports from the other.

**Consumes:** `list[FeatureVector]` + a saved model directory

**Emits:** `list[Candidate]` — one per source, with probability, confidence tier, and period

```
src/ml4em/inference/
  base.py         Predictor Protocol
  loader.py       load_model(path) → MLModel
  predictor.py    StandardPredictor   [shell]
  postprocess.py  probabilities_to_candidates   [implemented]
```

## Contents

- [Predictor Protocol](#predictor)
- [load\_model](#load-model)
- [StandardPredictor (shell)](#standardpredictor)
- [probabilities\_to\_candidates](#probabilities-to-candidates)

---

## `Predictor` Protocol { #predictor }

The contract every predictor must satisfy.

```python
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

model = load_model("models/xgb_v1/")
```

Reads `{path}/manifest.json`, finds `"model_class"`, and dispatches to the appropriate `@classmethod load()`:

```json
{"model_class": "XGBoostClassifier"}
```

The model registry in `inference/loader.py` maps class names to module paths:

```python
_MODEL_REGISTRY = {
    "XGBoostClassifier": "ml4em.models.xgboost",
}
```

To register a new model, add one entry to `_MODEL_REGISTRY`. See [Guide: Add a Model](../guides/add-model.md).

---

## `StandardPredictor` *(shell)* { #standardpredictor }

Runs `model.predict_proba` in batches and passes results to `probabilities_to_candidates`.

**Consumes:** `list[FeatureVector]` + `MLModel`

**Emits:** `list[Candidate]`

```python
from ml4em.inference import StandardPredictor

predictor = StandardPredictor(model, cfg.inference)
candidates = predictor.predict(feature_vectors)
```

> **Status:** `predict` is a shell — pending completion of the model implementation. `probabilities_to_candidates` (called internally) is fully implemented.

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
2. Copies `source_id`, `ra`, `dec`, `survey`, `period`, `period_algorithm` from each `FeatureVector`
3. Returns one frozen `Candidate` per source

### Confidence tier assignment

```yaml
inference:
  confidence_thresholds:
    high: 0.9
    medium: 0.5
```

| Probability | Confidence |
|-------------|------------|
| ≥ `high` threshold | `"high"` |
| ≥ `medium` threshold | `"medium"` |
| below `medium` threshold | `"low"` |

Tune thresholds to match your purity/completeness requirements. Raising `high` reduces false positives; lowering `medium` increases recall at the cost of more false positives.

---

[← Models](models.md){ .md-button } [← Training](training.md){ .md-button }
