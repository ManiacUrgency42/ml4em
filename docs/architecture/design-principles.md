# Design Principles

Four principles govern how ml4em is built. Understanding them makes the codebase
predictable — when you understand the rules, you can always guess where something lives
and why.

---

## 1. Protocols over inheritance

Every layer boundary is a `typing.Protocol`. Any class implementing the right methods
satisfies the contract — no base class, no registration, no import needed.

**Concrete consequence:** adding a new data source, extractor, or model is **one new
file**. Nothing else changes.

=== "Adding a new source"

    ```python
    # data/my_survey.py
    class MySurveySource:
        def fetch(self, source_id: str) -> list[LightCurve]:
            ...
        def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]:
            ...
    ```

    That's it. `FeaturePipeline` will accept this object without modification because
    it only checks for `fetch()` and `fetch_batch()` at runtime.

=== "What you don't need to do"

    ```python
    # You do NOT need to:
    # - inherit from LightCurveSource
    # - register MySurveySource anywhere
    # - modify data/__init__.py (optional, for convenience)
    # - modify any other file
    ```

=== "Contrast with inheritance"

    In a traditional inheritance design, you would write:
    ```python
    class MySurveySource(BaseLightCurveSource):  # must inherit
        ...
    ```
    And the feature layer would check `isinstance(source, BaseLightCurveSource)`.
    The Protocol approach avoids this coupling.

!!! info "For astrophysicists: what is a Protocol?"
    Think of it like a FITS header convention: if a file has `NAXIS`, `BITPIX`, and
    the required keywords, it's a valid FITS file — regardless of which program created
    it. A Protocol says: "if this object has these methods with these signatures, it
    satisfies the interface."

---

## 2. Code controls architecture, config controls parameters

**Architecture decisions** (which model, which extractors, which source) are made in
code by importing the relevant class. **Parameter tuning** (period search range, batch
size, learning rate, confidence thresholds) lives in `config.yaml`.

| Belongs in code | Belongs in `config.yaml` |
|----------------|------------------------|
| Which model to use (`XGBoostClassifier`) | Learning rate, number of estimators |
| Which extractors to compose | Period search bounds |
| XGBoost tree depth | Batch size |
| Number of Fourier harmonics | Confidence thresholds |
| Which survey to fetch from | Authentication tokens (→ `.env`) |

**Why this rule?** It prevents the config file from becoming a second programming
language. Architecture choices have discrete alternatives (XGBoost vs. neural net) that
require changing imports and class instantiations. Parameter choices are continuous
values that users legitimately need to tune without modifying Python code.

**Model hyperparameters specifically:** hyperparameters that define model architecture
(tree depth, number of layers, dropout) are NOT in `PipelineConfig`. They live in
per-model config dataclasses (`XGBoostConfig`, etc.) and are set at construction time
in code:

```python
# Architecture: in code
model = XGBoostClassifier(config=XGBoostConfig(n_estimators=500, max_depth=6))

# Loop parameters: in config.yaml
trainer = StandardTrainer(model, cfg.training)  # cfg.training has lr, batch_size, etc.
```

---

## 3. Explicit data contracts

Three dataclasses (`LightCurve`, `FeatureVector`, `Candidate`) are the **only objects
that cross layer boundaries**. No raw dicts, no tuples, no numpy arrays with implicit
column ordering.

**Why this matters:**

Raw dicts and tuples make a codebase fragile:
```python
# BAD: what are these columns? what order?
features = extractor.extract(lcs)  # returns list of tuples
model.predict(features[3], features[7])  # which index is which?
```

Named dataclasses make mistakes impossible to hide:
```python
# GOOD: the name is the documentation
fv = feature_pipeline.run(lcs)       # returns FeatureVector
print(fv.chi2red, fv.stetson_j)      # explicit field access
model.predict_proba([fv])            # typed input
```

!!! info "For astrophysicists: what is a dataclass?"
    A dataclass is a Python class whose only purpose is to hold named data fields with
    type annotations. Think of it like a FITS table row definition — it specifies what
    columns exist and what types they are. Unlike a dict, you get tab-completion, type
    checking, and an error if you misspell a field name.

    ```python
    @dataclass
    class LightCurve:
        source_id : str
        time      : np.ndarray
        mag       : np.ndarray
        ...
    ```

**All float fields in `FeatureVector` default to `np.nan`.** This means a partial
extraction (e.g. period extractor returns nothing for a source with too few
observations) produces a valid `FeatureVector` with NaN in the period fields — not an
exception. XGBoost handles NaN natively; other models may need to impute.

---

## 4. Partial execution is safe

The pipeline never raises on partial data. Instead, it fills missing values with
`np.nan` and continues.

**The contract for extractors:** the `FeatureExtractor.extract()` method must never
raise. On failure (network error, too few observations, algorithm timeout), it returns
an empty dict `{}` for that source. `FeaturePipeline` merges the empty dict with the
other extractors' outputs, leaving those fields as NaN.

**The minimum observations threshold:** sources with fewer than `min_observations`
(default 50) return an all-NaN `FeatureVector` immediately without running any extractor.

**What this means in practice:**

- You can run `FeaturePipeline` with `CatalogExtractor` stubbed out: the 4 Gaia
  fields will be NaN, and XGBoost will handle them.
- You can process a batch of 10,000 sources even if 200 of them have network errors —
  they produce NaN vectors, which can be filtered after the fact.
- No try/except logic is needed in calling code.
