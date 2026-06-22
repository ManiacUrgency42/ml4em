# Architecture Overview

ml4em is organized into six layers. Each layer has a single responsibility, a
well-defined Protocol interface, and strict dependency rules.

---

## The six layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  Foundation                                                          │
│  types.py  ·  constants.py  ·  config/                              │
│  Data contracts, physical constants, validated pipeline config       │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────────────┐
│  Data                  data/                                         │
│  Protocol: LightCurveSource                                          │
│  fetch(source_id) → list[LightCurve]                                 │
│  Implementations: ZTFSource · RubinSource · SimulatedSource          │
└──────────────────┬───────────────────────────────────────────────────┘
                   │ LightCurve
┌──────────────────▼───────────────────────────────────────────────────┐
│  Features              features/                                     │
│  Protocol: FeatureExtractor                                          │
│  extract(lcs) → dict                                                 │
│  Extractors: StatisticsExtractor · PeriodExtractor                   │
│              DmdtExtractor · CatalogExtractor                        │
│  Composer:   FeaturePipeline                                         │
└──────────┬───────────────────────────────────────────────────────────┘
           │ FeatureVector
     ┌─────┴──────┐
     │            │
┌────▼────┐  ┌────▼─────────────────────────────────────────────────┐
│Training │  │  Models               models/                        │
│training/│  │  Protocol: MLModel                                   │
│         │  │  predict_proba(features) → np.ndarray                │
│Trainer  │  │  Reference: XGBoostClassifier                        │
│Protocol │  │  Utilities: SCALAR_FIELDS · features_to_array        │
│         │  └────┬─────────────────────────────────────────────────┘
│         │       │ MLModel
│FeatureD-│  ┌────▼─────────────────────────────────────────────────┐
│ataset   │  │  Inference            inference/                     │
│Standard-│  │  Protocol: Predictor                                 │
│Trainer  │  │  predict(features) → list[Candidate]                 │
└─────────┘  │  StandardPredictor · load_model · postprocess        │
             └──────────────────────────────────────────────────────┘
```

**Dependency rule:** each layer imports only from layers above it. Training and
inference are parallel — neither imports from the other.

---

## Data flow — one source, end to end

Here is what happens to a single astronomical source as it passes through the pipeline:

```
source_id (e.g. "686149073900013696")
  │
  ▼  Data layer
  ZTFSource.fetch_batch([source_id])
  → [LightCurve(g), LightCurve(r), LightCurve(i)]   ← one per band
  │
  ▼  Feature layer
  FeaturePipeline.run_batch([[lc_g, lc_r, lc_i]])
  → [FeatureVector(median=18.4, chi2red=52.3, period=0.12, ...)]
  │
  ▼  Models + Inference layers
  StandardPredictor.predict([feature_vector])
  → [Candidate(source_id="...", probability=0.92, confidence="high", period=0.12)]
```

---

## Training vs. inference — parallel branches

Training and inference are **parallel branches** that share only the `FeatureVector`
(input) and `MLModel` (the contract) from the models layer:

```
          FeatureVector
               │
       ┌───────┴───────┐
       │               │
   Training         Inference
   ─────────        ─────────
   FeatureDataset   StandardPredictor
   StandardTrainer  load_model
       │               │
   model.save()    model.predict_proba()
                       │
                   list[Candidate]
```

Neither training nor inference imports from the other. You can run inference without
ever training (load a pre-trained model), or train without doing inference.

---

## Protocol table

Every layer boundary is a Protocol. Here is the complete list:

| Protocol | Defined in | Method signatures | Concrete implementations |
|----------|-----------|------------------|--------------------------|
| `LightCurveSource` | `data/base.py` | `fetch()`, `fetch_batch()` | `ZTFSource`, `RubinSource`, `SimulatedSource` |
| `FeatureExtractor` | `features/base.py` | `extract()` | `StatisticsExtractor`, `PeriodExtractor`, `DmdtExtractor`, `CatalogExtractor` |
| `MLModel` | `models/base.py` | `predict_proba()`, `save()` | `XGBoostClassifier` |
| `Trainer` | `training/base.py` | `fit()`, `save()` | `StandardTrainer` |
| `Predictor` | `inference/base.py` | `predict()` | `StandardPredictor` |

!!! info "What is a Protocol? (for astrophysicists)"
    In Python, a `Protocol` is a structural interface — it defines what methods a class
    must have, without requiring that class to inherit from anything.

    Any class that implements the right methods automatically satisfies the Protocol.
    No registration, no base class, no decorators needed.

    This is called **duck typing**: "if it has `fetch()` and `fetch_batch()`, it is a
    `LightCurveSource`." This is why adding a new data source requires zero changes to
    any other file — the feature layer accepts any object with those methods.

---

## The three shared types

Three dataclasses are the only objects that cross layer boundaries:

| Type | Flows between | Description |
|------|--------------|-------------|
| `LightCurve` | Data → Features | Single-band time series for one source |
| `FeatureVector` | Features → Models/Training/Inference | Fully extracted feature set |
| `Candidate` | Inference → output | Classification result for one source |

See [Data Contracts](../data-contracts.md) for full field tables.

---

## Navigate to a layer

Each row is a clickable link to the full layer reference page.

| # | Layer | Responsibility | Receives | Produces |
|---|-------|---------------|----------|---------|
| 1 | [**Foundation**](../layers/foundation.md) | Shared types, constants, config | — | `LightCurve`, `FeatureVector`, `PipelineConfig` |
| 2 | [**Data**](../layers/data.md) | Fetch light curves from surveys | `source_id` | `list[LightCurve]` |
| 3 | [**Features**](../layers/features.md) | Extract numerical features | `list[LightCurve]` | `FeatureVector` |
| 4 | [**Models**](../layers/models.md) | Define the ML model contract | `FeatureVector` | `np.ndarray` (probabilities) |
| 5 | [**Training**](../layers/training.md) | Load labels and fit a model | `FeatureVector` + labels CSV | saved model file |
| 6 | [**Inference**](../layers/inference.md) | Run a saved model on new data | `FeatureVector` + model file | `list[Candidate]` |
