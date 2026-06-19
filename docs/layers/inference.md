# Inference Layer

The inference layer loads a trained model and converts `FeatureVector` objects into
`Candidate` predictions.

```
src/ml4em/inference/
  base.py         Predictor Protocol
  loader.py       load_model(path) → MLModel
  predictor.py    StandardPredictor
  postprocess.py  probabilities_to_candidates            [fully implemented]
```

---

## Protocol — `Predictor`

```python
class Predictor(Protocol):
    def predict(self, features: list[FeatureVector]) -> list[Candidate]: ...
```

---

## `load_model`

```python
from ml4em.inference import load_model

model = load_model("models/xgb_v1/")
```

`load_model` reads `{path}/manifest.json`, finds `"model_class"`, and dispatches to
the appropriate `@classmethod load()`:

```json
// models/xgb_v1/manifest.json
{"model_class": "XGBoostClassifier"}
```

The model registry in `inference/loader.py` maps class names to their module paths:

```python
_MODEL_REGISTRY = {
    "XGBoostClassifier": "ml4em.models.xgboost",
}
```

This is the **only place** that knows about concrete model types. Everything else in
the inference layer is model-agnostic.

To register a new model: add one entry to `_MODEL_REGISTRY`. See
[Guide: Add a Model](../guides/add-model.md).

---

## `StandardPredictor` *(shell)*

```python
from ml4em.inference import StandardPredictor

predictor = StandardPredictor(model, cfg.inference)
candidates = predictor.predict(feature_vectors)
```

Calls `model.predict_proba(features)` in batches of `cfg.inference.batch_size`, then
passes the resulting probabilities to `postprocess.probabilities_to_candidates`.

!!! note "Status"
    `StandardPredictor.predict` is a shell pending completion of the model
    implementation. `probabilities_to_candidates` (the postprocessing step) is fully
    implemented.

---

## `probabilities_to_candidates` — fully implemented

Converts raw model probabilities into `Candidate` objects:

```python
from ml4em.inference.postprocess import probabilities_to_candidates

candidates = probabilities_to_candidates(features, probs, cfg.inference)
```

Steps:
1. Takes `probs[:, 1]` — the positive-class probability from the `(N, 2)` output array
2. Assigns confidence tier (`"high"` / `"medium"` / `"low"`) based on thresholds
3. Copies `source_id`, `ra`, `dec`, `survey`, `period`, `period_algorithm` from each
   `FeatureVector`
4. Returns one `Candidate` (frozen dataclass) per source

### Confidence tier assignment

```yaml
# config.yaml
inference:
  confidence_thresholds:
    high: 0.9
    medium: 0.5
```

| Probability | Confidence |
|------------|------------|
| ≥ high threshold | `"high"` |
| ≥ medium threshold | `"medium"` |
| below medium threshold | `"low"` |

The thresholds are tunable. There is no hardcoded science meaning — set them to match
the purity/completeness trade-off your analysis requires.

**High purity run:** set `high=0.95` to minimize false positives, accepting that some
true positives will only appear in `"medium"` or `"low"`.

**High completeness run:** set `medium=0.3` to catch more true positives in the
`"medium"` tier, accepting more false positives.
