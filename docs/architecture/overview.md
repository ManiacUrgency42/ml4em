# Architecture Overview

ml4em is organized into six layers. Each layer has a single responsibility, a
well-defined Protocol interface, and strict dependency rules.

---

```mermaid
flowchart TD
    F["<b>FOUNDATION</b><br/>──────────────────────────<br/><b>Data contracts</b><br/>LightCurve · FeatureVector<br/>LabeledSample · Candidate<br/>──────────────────────────<br/><b>Survey constants</b> · <b>Pipeline config</b>"]

    D["<b>DATA</b><br/>──────────────────────────<br/><b>Protocol:</b> LightCurveSource<br/>──────────────────────────<br/>ZTFSource · RubinSource"]

    FE["<b>FEATURES</b><br/>──────────────────────────<br/><b>Protocol:</b> FeatureExtractor<br/>──────────────────────────<br/>StatisticsExtractor · PeriodExtractor<br/>DmdtExtractor · CatalogExtractor<br/>FeaturePipeline"]

    M["<b>MODELS</b><br/>──────────────────────────<br/><b>Protocol:</b> MLModel<br/>──────────────────────────<br/>XGBoostClassifier · SCALAR_FIELDS"]

    T["<b>TRAINING</b><br/>──────────────────────────<br/><b>Protocol:</b> Trainer<br/>──────────────────────────<br/>FeatureDataset · StandardTrainer"]

    I["<b>INFERENCE</b><br/>──────────────────────────<br/><b>Protocol:</b> Predictor<br/>──────────────────────────<br/>load_model · StandardPredictor<br/>probabilities_to_candidates"]

    F --> D
    D -->|"list[LightCurve]"| FE
    FE -->|"list[FeatureVector]"| T
    FE -->|"list[FeatureVector]"| M
    M -->|"MLModel"| I
    T -. "saved model" .-> I
```

**Dependency rule:** each layer imports only from layers above it. Training and
inference are parallel — neither imports from the other.

---

## Data flow — one source, end to end

Here is what happens to a single astronomical source as it passes through the pipeline:

```mermaid
flowchart TD
    A["<b>source_id</b><br/>686149073900013696"]
    B["<b>list[LightCurve]</b><br/>LightCurve(g) · LightCurve(r) · LightCurve(i)"]
    C["<b>FeatureVector</b><br/>median=18.4 · chi2red=52.3<br/>period=0.12 · dmdt=(26×26)"]
    D["<b>Candidate</b><br/>probability=0.92<br/>confidence=high · period=0.12"]

    A -->|"ZTFSource.fetch_batch()"| B
    B -->|"FeaturePipeline.run_batch()"| C
    C -->|"StandardPredictor.predict()"| D
```

---

## Training vs. inference — parallel branches

Training and inference are **parallel branches** that share only the `FeatureVector`
(input) and `MLModel` (the contract) from the models layer:

```mermaid
flowchart TD
    FV["<b>list[FeatureVector]</b>"]
    M["<b>MLModel</b><br/>shared contract"]

    T["<b>TRAINING</b><br/>──────────────────────────<br/>FeatureDataset · StandardTrainer<br/>model.fit() → model.save()"]
    I["<b>INFERENCE</b><br/>──────────────────────────<br/>load_model() → StandardPredictor<br/>model.predict_proba() → list[Candidate]"]

    FV --> T & I
    M --> T & I
```

Neither training nor inference imports from the other. You can run inference without
ever training (load a pre-trained model), or train without doing inference.

---

## Protocol table

Every layer boundary is a Protocol. Here is the complete list:

| Protocol | Defined in | Method signatures | Concrete implementations |
|----------|-----------|------------------|--------------------------|
| `LightCurveSource` | `data/base.py` | `fetch_batch()` | `ZTFSource`, `RubinSource` |
| `FeatureExtractor` | `features/base.py` | `extract()` | `StatisticsExtractor`, `PeriodExtractor`, `DmdtExtractor`, `CatalogExtractor` |
| `MLModel` | `models/base.py` | `predict_proba()`, `save()` | `XGBoostClassifier` |
| `Trainer` | `training/base.py` | `fit()`, `save()` | `StandardTrainer` |
| `Predictor` | `inference/base.py` | `predict()` | `StandardPredictor` |

---

## The three shared types

Three dataclasses are the only objects that cross layer boundaries:

| Type | Flows between | Description |
|------|--------------|-------------|
| `LightCurve` | Data → Features | Single-band time series for one source |
| `FeatureVector` | Features → Models / Training / Inference | Fully extracted feature set |
| `Candidate` | Inference → output | Classification result for one source |

See [Data Contracts](../data-contracts.md) for full field tables.

---

## Navigate to a layer

| # | Layer | Responsibility | Receives | Produces |
|---|-------|---------------|----------|---------|
| 1 | [**Foundation**](../layers/foundation.md) | Shared types, constants, config | — | `LightCurve`, `FeatureVector`, `PipelineConfig` |
| 2 | [**Data**](../layers/data.md) | Fetch light curves from surveys | `source_id` | `list[LightCurve]` |
| 3 | [**Features**](../layers/features.md) | Extract numerical features | `list[LightCurve]` | `FeatureVector` |
| 4 | [**Models**](../layers/models.md) | Define the ML model contract | `FeatureVector` | `np.ndarray` (probabilities) |
| 5 | [**Training**](../layers/training.md) | Load labels and fit a model | `FeatureVector` + labels CSV | saved model file |
| 6 | [**Inference**](../layers/inference.md) | Run a saved model on new data | `FeatureVector` + model file | `list[Candidate]` |
