# ml4em

Machine learning for electromagnetic light curve analysis.

A general-purpose, modular library for building ML pipelines on top of photometric time series data from astronomical surveys. Science-case agnostic by design — the researcher defines the target class (WDB, AGN, RR Lyrae, etc.) through training labels and model choice.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Design Principles](#design-principles)
- [Data Contracts](#data-contracts-typespy)
  - [`LightCurve`](#lightcurve--data--feature)
  - [`FeatureVector`](#featurevector--feature--training--inference)
  - [`LabeledSample`](#labeledsample--label-preparation--training)
  - [`Candidate`](#candidate--inference--output)
- [Layer Reference](#layer-reference)
  - [Foundation](#foundation)
  - [Data layer](#data-layer--data)
  - [Feature layer](#feature-layer--features)
  - [Models layer](#models-layer--models)
  - [Training layer](#training-layer--training)
  - [Inference layer](#inference-layer--inference)
- [Dependencies](#dependencies)
- [Build & Deployment](#build--deployment)
  - [Why Docker, not conda](#why-docker-not-conda)
  - [Build targets](#build-targets)
  - [GHCR](#ghcr-github-container-registry)
  - [MSI / Apptainer](#msi--apptainer)
  - [Initialising the submodule](#initialising-the-submodule)
  - [Updating periodfind](#updating-periodfind)
- [Implementation Status](#implementation-status)

---

## Architecture Overview

The library is organized into six layers. Each layer has a single responsibility, a well-defined Protocol, and strict dependency boundaries.

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

**Dependency rule:** each layer imports only from layers above it. Training and inference are parallel — neither imports from the other.

---

## Design Principles

**Protocols over inheritance.** Every layer boundary is defined by a `typing.Protocol`. Any class implementing the right methods satisfies the contract with no registration or base class required. Adding a new data source, extractor, or model is one new file.

**Code controls architecture, config controls parameters.** Model architecture (which model, layer widths, tree depth) is chosen in code by importing the relevant class. `PipelineConfig` / `config.yaml` controls loop and pipeline parameters (learning rate, batch size, period search range, storage paths).

**Explicit data contracts.** Three dataclasses are the only shared language between layers. No raw tuples or dicts cross a module boundary.

**Partial execution is safe.** All `FeatureVector` float fields default to `np.nan`. A source with too few observations returns an all-NaN vector, not an exception. The pipeline continues.

---

## Data Contracts (`types.py`)

The four types defined here are the only objects that cross layer boundaries.

### `LightCurve` — Data → Feature

Single-band photometric time series for one source.

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `str` | Survey-native identifier |
| `time` | `ndarray (N,)` | Observation times in MJD |
| `mag` | `ndarray (N,)` | Apparent magnitude |
| `mag_err` | `ndarray (N,)` | 1-sigma uncertainty |
| `band` | `Band` | Photometric filter: `u g r i z y` |
| `survey` | `Survey` | `"ztf"` \| `"rubin"` \| `"simulated"` |
| `ra`, `dec` | `float` | Sky position, decimal degrees (J2000) |

### `FeatureVector` — Feature → Training / Inference

Fully extracted feature set for one source. All float fields default to `np.nan`.

| Group | Fields | Count |
|-------|--------|-------|
| Sky position | `ra`, `dec` | 2 |
| LC statistics | `median` `wmean` `chi2red` `roms` `wstd` `norm_peak_to_peak_amp` `norm_excess_var` `median_abs_dev` `iqr` `i60r` `i70r` `i80r` `i90r` `skew` `small_kurt` `inv_von_neumann` `stetson_i` `stetson_j` `stetson_k` `anderson_darling` `shapiro_wilk` `n_obs` | 22 |
| Period | `period` `period_significance` `period_algorithm` | 3 |
| Fourier | `f1_power` `f1_bic` `f1_a` `f1_b` `f1_amp` `f1_phi0` `f1_relamp1..4` `f1_relphi1..4` | 14 |
| dm/dt image | `dmdt` — shape `(26, 26)` ndarray | 1 |
| Gaia | `gaia_parallax` `gaia_parallax_error` `gaia_bp_rp` `gaia_ruwe` | 4 |

**42 scalar fields** (everything except `source_id`, `survey`, `period_algorithm`, `dmdt`). These are listed in `models.SCALAR_FIELDS` for use by any scalar-based model.

### `LabeledSample` — Label preparation → Training

```python
@dataclass
class LabeledSample:
    feature : FeatureVector
    label   : int   # 1 = positive class, 0 = background
```

Labels are never generated by ml4em. They come from the researcher's upstream preparation step (e.g. a catalog cross-match).

### `Candidate` — Inference → Output

Immutable inference result for one source (`frozen=True`).

| Field | Type | Description |
|-------|------|-------------|
| `source_id` `ra` `dec` `survey` | — | Source identity |
| `probability` | `float` | P(positive class) in [0, 1] |
| `period` `period_algorithm` | — | Dominant period from feature layer |
| `confidence` | `"high"` \| `"medium"` \| `"low"` | Derived from `InferenceConfig.confidence_thresholds` |

---

## Layer Reference

### Foundation

```
types.py          Data contracts (LightCurve, FeatureVector, LabeledSample, Candidate)
constants.py      Survey constants, dm/dt bin parameters, cross-match geometry
config/
  schema.py       Pydantic models — PipelineConfig and all sub-configs
  loader.py       YAML loader + env-var secret accessors
```

**`PipelineConfig`** is the pipeline expressed as configuration. Each section maps to exactly one layer:

```
PipelineConfig.sources.ztf    →  ZTFSource
PipelineConfig.sources.rubin  →  RubinSource
PipelineConfig.features       →  FeaturePipeline
PipelineConfig.storage        →  all layers (shared file paths)
PipelineConfig.training       →  StandardTrainer (loop params only)
PipelineConfig.inference      →  StandardPredictor
```

Model architecture hyperparameters (layer widths, tree depth, dropout) are **not** in `PipelineConfig`. They live in per-model config dataclasses (`XGBoostConfig`, etc.) and are set in code.

Secrets (`ML4EM_ZTF_TOKEN`, `ML4EM_RUBIN_TOKEN`) are never stored in config. Read them with `config.get_ztf_token()` / `config.get_rubin_token()` which pull from environment variables or a `.env` file.

---

### Data layer — `data/`

```
data/
  base.py         LightCurveSource Protocol
  ztf.py          ZTFSource   — Kowalski/penquins client      [implemented]
  rubin.py        RubinSource — Rubin DP1 via TAP             [stub]
  simulation.py   SimulatedSource — Lcurve wrapper            [stub]
```

#### Protocol — `LightCurveSource`

```python
class LightCurveSource(Protocol):
    def fetch(self, source_id: str) -> list[LightCurve]: ...
    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]: ...
```

Any class with these two methods is a valid source. The feature layer accepts any compliant object — it never imports a concrete source class directly.

#### `ZTFSource`

Fetches from the Kowalski database via the `penquins` client (requires `pip install "ml4em[ztf]"`).

- Each ZTF `_id` encodes one (sky position, band) pair → one `LightCurve`
- `fetch_batch` sends a single batched Kowalski `find` query
- Data quality: discards flagged epochs (`catflags != 0`) and removes intra-night duplicates within `min_cadence_days` (default 30 min) to prevent period-finding aliases

#### `RubinSource` *(stub)*

Planned TAP query against `dp1.ForcedSource ⋈ dp1.Visit ⋈ dp1.Object`. One `objectId` may return up to six `LightCurve` objects (one per band: u g r i z y).

#### `SimulatedSource` *(stub)*

Will wrap Tom Marsh's Lcurve code to produce physics-based synthetic light curves. `source_id` is a path to an Lcurve `.mod` parameter file or a grid index.

**Adding a new source:** create a file in `data/` with a class implementing `fetch()` and `fetch_batch()`. No registration needed.

---

### Feature layer — `features/`

```
features/
  base.py         FeatureExtractor Protocol
  statistics.py   StatisticsExtractor  — 22 scalar LC statistics    [implemented]
  period.py       PeriodExtractor      — period finding + 14 Fourier [implemented]
  dmdt.py         DmdtExtractor        — 26×26 pairwise histogram    [implemented]
  catalog.py      CatalogExtractor     — 4 Gaia EDR3 features        [stub]
  pipeline.py     FeaturePipeline      — composer                    [implemented]
```

#### Protocol — `FeatureExtractor`

```python
class FeatureExtractor(Protocol):
    def extract(self, sources: list[list[LightCurve]]) -> list[dict[str, Any]]: ...
```

Batch-first interface: each element of `sources` is all bands for one source (one `LightCurve` per filter). Returns one dict per source. A single source is just a batch of one. Extractors must never raise — return an empty dict on failure so the pipeline continues.

All feature computation is delegated to **periodfind** (a Rust/CUDA-backed library), which processes the entire batch in a single kernel call. `periodfind` is a hard dependency — see [Build & Deployment](#build--deployment) for how to install it.

#### `StatisticsExtractor`

Delegates to `periodfind.BasicStats().calc(times, mags, errs)` — a Rust-backed batched implementation. Selects the primary band (most observations) per source, casts to float32, and makes one call for all N sources, returning an `(N, 22)` array. Column order is defined by `BasicStats.STAT_NAMES`; names are remapped to `FeatureVector` field names (e.g. `"RoMS"` → `"roms"`, `"WelchI"` → `"stetson_i"`).

No sigma-clipping — consistent with scope-ml's periodfind-based pipeline.

#### `PeriodExtractor`

Delegates to periodfind's GPU-accelerated algorithms. Algorithm objects and the period grid are built once at construction and reused across all `extract()` calls.

**Core algorithms (scope-ml production set):**

| Key | Algorithm | periodfind class |
|-----|-----------|-----------------|
| `CE`  | Conditional Entropy        | `ConditionalEntropy(n_phase=20, n_mag=10)` |
| `AOV` | Analysis of Variance       | `AOV(n_phase=20)` |
| `LS`  | Lomb-Scargle               | `LombScargle()` |
| `MHF` | Multi-Harmonic Fourier     | `MultiHarmonicFourier(max_harmonics=5)` |

Each algorithm runs over all N sources in one batched call (`output='peaks'`). Cross-algorithm agreement scoring picks the period confirmed by the most algorithms within a 2% fractional tolerance; falls back to highest significance if no pair agrees.

Fourier decomposition is then run via `periodfind.FourierDecomposition().calc()` on the valid-period subset, returning 14 features per source: `[power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]`.

Configurable via `PeriodConfig`: `min_period_days`, `max_period_days`, `n_freq_grid`, `algorithms`, `top_n_periods`.

#### `DmdtExtractor`

Delegates to `periodfind.DmDt().calc(times, mags, dt_edges, dm_edges)` — a Rust-backed batched implementation. Δt edges are log-spaced (float32), Δmag edges are linear (float32), both pre-built once in `__init__`. Returns an `(N, n_dm, n_dt)` array.

Set `features.compute_dmdt: false` in config to skip this extractor entirely for scalar-only models.

#### `CatalogExtractor` *(stub)*

Will query Gaia EDR3 for the nearest counterpart within `xmatch_radius_arcsec` (default 2 arcsec). Returns `gaia_parallax`, `gaia_parallax_error`, `gaia_bp_rp`, `gaia_ruwe`. Two planned backends: astroquery TAP+ or Kowalski Gaia catalog.

#### `FeaturePipeline`

Composes extractors in order (statistics → period → dmdt → catalog), merges their output dicts, and builds a `FeatureVector`. Sources with fewer than `min_observations` (default 50) return an all-NaN vector immediately.

```python
pipeline = FeaturePipeline.default(cfg.features)   # standard ordering
fvs = pipeline.run_batch(grouped_lcs)              # one FeatureVector per source
fv  = pipeline.run_batch([lcs])[0]                 # single-source (batch of 1)
```

`run_batch()` calls `periodfind.set_device(device)` once before processing, then chunks `grouped_lcs` into batches of `feature_batch_size` (default 1000). The `device` setting (`"cpu"` / `"gpu"` / `"auto"`) is orthogonal to batch size and is read from `FeatureConfig.device`.

**Adding a new extractor:** create a file in `features/` implementing the batch `extract()` interface. Pass it to `FeaturePipeline` — no other changes needed.

---

### Models layer — `models/`

```
models/
  base.py         MLModel Protocol + SCALAR_FIELDS utilities
  xgboost.py      XGBoostClassifier — reference implementation
```

The models layer defines the shared contract between training and inference. It does not perform training or inference itself.

#### Protocol — `MLModel`

```python
class MLModel(Protocol):
    def predict_proba(self, features: list[FeatureVector]) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
```

Any class implementing these two methods satisfies `MLModel`. `load` is a `@classmethod` on each concrete model, not on the Protocol (torch and joblib load differently; the dispatch lives in `inference/loader.py`).

#### Scalar field utilities

Defined in `models/base.py`, shared by any scalar-based model:

| Name | Description |
|------|-------------|
| `SCALAR_FIELDS` | Ordered list of 42 float `FeatureVector` field names |
| `N_SCALAR_FEATURES` | `42` |
| `features_to_array(features)` | Extracts `SCALAR_FIELDS` → `np.ndarray (N, 42)` |

Field order in `SCALAR_FIELDS` is stable. Changing it invalidates saved models trained on that ordering.

#### `XGBoostClassifier` — reference implementation

Gradient-boosted tree classifier. Uses `SCALAR_FIELDS` only; `dmdt` image is ignored. Implements `predict_proba`, `save` (writes `model.ubj` + `manifest.json`), and `@classmethod load`.

This class exists as a **pattern reference**, not as the committed model for any science case. When adding your own model, follow this file as the template.

#### Adding a new model

1. Create `models/my_model.py` with `MyModelConfig` (dataclass) and `MyModel`
2. Implement `predict_proba()`, `save()`, `@classmethod load()`
3. Add one entry to `inference/loader.py` `_MODEL_REGISTRY`
4. Import and use — training, inference, and postprocess are unchanged

```python
# Swap model = one import + one constructor
from ml4em.models import XGBoostClassifier, XGBoostConfig
# from ml4em.models.my_model import MyModel, MyModelConfig

model = XGBoostClassifier(config=XGBoostConfig(n_estimators=500))
trainer = StandardTrainer(model, cfg.training)
trainer.fit(dataset)
```

---

### Training layer — `training/`

```
training/
  base.py         Trainer Protocol
  dataset.py      FeatureDataset — load features + join labels
  trainer.py      StandardTrainer
```

Training and inference are parallel — neither imports from the other. Both consume `FeatureVector` (from the feature layer) and `MLModel` (from the models layer).

#### Protocol — `Trainer`

```python
class Trainer(Protocol):
    def fit(self, dataset: FeatureDataset) -> None: ...
    def save(self, path: str) -> None: ...
```

#### `FeatureDataset`

Loads `FeatureVector` objects from `storage.features_dir` (parquet files written by `FeaturePipeline`) and joins them with a researcher-supplied labels CSV.

```
labels.csv format:
  source_id,label
  1234567890,1
  0987654321,0
```

Labels must be `0` (negative/background) or `1` (positive class). Sources present in only one of features or labels are silently skipped.

```python
dataset = FeatureDataset.from_storage(cfg.storage, labels_path="labels.csv")
train, val, test = dataset.split(val_fraction=0.1, test_fraction=0.1, seed=42)
print(dataset.class_counts())        # {0: 5000, 1: 320}
print(dataset.positive_fraction())   # 0.060
```

Parquet loading is a stub pending the feature layer writing output files.

#### `StandardTrainer` *(shell)*

```python
trainer = StandardTrainer(model, cfg.training)
trainer.fit(dataset)   # → NotImplementedError (training loop pending)
trainer.save(path)     # delegates to model.save(path) — implemented
```

`cfg.training` controls loop params (`lr`, `batch_size`, `max_epochs`, `patience`, `seed`). Model architecture is set at construction time on the model object itself.

---

### Inference layer — `inference/`

```
inference/
  base.py         Predictor Protocol
  loader.py       load_model(path) → MLModel
  predictor.py    StandardPredictor
  postprocess.py  probabilities_to_candidates            [fully implemented]
```

#### Protocol — `Predictor`

```python
class Predictor(Protocol):
    def predict(self, features: list[FeatureVector]) -> list[Candidate]: ...
```

#### `load_model`

```python
model = load_model("models/xgb_v1/")
```

Reads `{path}/manifest.json` → `{"model_class": "XGBoostClassifier"}` → dispatches to `XGBoostClassifier.load(path)`. This is the only place that knows about concrete model types. Everything else in the inference layer is model-agnostic.

To register a new model: add one entry to `_MODEL_REGISTRY` in `inference/loader.py`.

#### `StandardPredictor` *(shell)*

```python
predictor = StandardPredictor(model, cfg.inference)
candidates = predictor.predict(feature_vectors)
```

Calls `model.predict_proba()` in batches of `cfg.inference.batch_size`, then passes probabilities to `postprocess`.

#### `postprocess.probabilities_to_candidates`

Fully implemented. Converts raw probabilities → `list[Candidate]` by:
1. Applying `cfg.inference.confidence_thresholds` to assign `"high"` / `"medium"` / `"low"`
2. Copying `source_id`, `ra`, `dec`, `survey`, `period`, `period_algorithm` from each `FeatureVector`

```python
candidates = probabilities_to_candidates(features, probs, cfg.inference)
```

---

## Dependencies

Core install requires `numpy`, `pydantic`, `pyyaml`, `python-dotenv`, and **`periodfind`**.

`periodfind` cannot be installed via `pip install periodfind` on Python ≥ 3.11 (no pre-built wheels). It must be compiled from source inside a Docker image — see [Build & Deployment](#build--deployment). The submodule lives at `external/periodfind`.

Optional extras:

```bash
pip install "ml4em[ztf]"        # ZTF via Kowalski  (penquins)
pip install "ml4em[rubin]"      # Rubin via TAP     (pyvo, pyarrow)
pip install "ml4em[catalog]"    # Gaia xmatch       (astropy)
pip install "ml4em[training]"   # Training           (torch, scikit-learn)
pip install "ml4em[inference]"  # Inference          (torch)
pip install "ml4em[dev]"        # Dev tools          (pytest, ruff)
```

---

## Build & Deployment

ml4em bundles `periodfind` as a git submodule at `external/periodfind` and bakes the entire build stack into Docker images. Researchers do not install Rust, maturin, or CUDA separately.

### Why Docker, not conda

`periodfind` has a two-layer build:

1. **`periodfind_cpu`** — a Rust extension built with `maturin` (requires the Rust toolchain)
2. **`periodfind`** — a Cython extension; CUDA GPU extensions compile automatically when `nvcc` is present, skipped silently when absent

Managing Rust + maturin + optional CUDA in a conda env on HPC clusters caused repeated dependency conflicts. Docker bakes the full toolchain once and ships a reproducible binary image. MSI (Minnesota Supercomputing Institute) runs Apptainer (formerly Singularity), which converts Docker images into `.sif` files that run without root privileges.

### Build targets

The `Dockerfile` defines two named targets:

| Target | Base image | periodfind GPU extensions | Use for |
|--------|-----------|--------------------------|---------|
| `cpu`  | `ubuntu:22.04` | no (`nvcc` absent) | local dev, CI, CPU-only runs |
| `gpu`  | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` | yes (`nvcc` present) | MSI GPU nodes |

```bash
# CPU image — local development and CI
docker build --target cpu -t ml4em:cpu .

# GPU image — MSI production
docker build --target gpu -t ml4em:gpu .
```

Both targets execute the same build sequence:
1. Install Python 3.11, build tools, Rust toolchain via `rustup`
2. Build `periodfind_cpu` from `external/periodfind/rust` with `maturin build --release`
3. Install `periodfind` from `external/periodfind` (`setup.py` auto-detects `nvcc`)
4. Install `ml4em` in editable mode

### GHCR (GitHub Container Registry)

Push built images so MSI can pull them:

```bash
docker push ghcr.io/<org>/ml4em:cpu
docker push ghcr.io/<org>/ml4em:gpu
```

### MSI / Apptainer

MSI GPU nodes cannot run Docker directly (no root). Pull the image as an Apptainer `.sif` file once, then run it for every job.

```bash
# One-time pull — store in /scratch, not $HOME (.sif files are 5–8 GB)
apptainer pull /scratch/$USER/ml4em_gpu.sif docker://ghcr.io/<org>/ml4em:gpu

# Run the pipeline with GPU passthrough
apptainer run --nv \
    --bind /scratch/$USER/data:/data \
    /scratch/$USER/ml4em_gpu.sif \
    python -m ml4em.run --config /data/config.yaml
```

Key flags:
- `--nv` — pass NVIDIA GPU through to the container (required for GPU period-finding)
- `--bind /scratch/$USER/data:/data` — mount your scratch data directory inside the container at `/data`

The GPU device is controlled by `features.device` in `config.yaml` (`"cpu"` / `"gpu"` / `"auto"`). The container handles everything else.

### Initialising the submodule

When cloning ml4em for the first time:

```bash
git clone --recurse-submodules <ml4em-repo-url>
# or, after a plain clone:
git submodule update --init
```

### Updating periodfind

The submodule is pinned to a specific commit. To advance it:

```bash
cd external/periodfind
git fetch origin
git checkout <new-commit-or-tag>
cd ../..
git add external/periodfind
git commit -m "chore: bump periodfind to <new-version>"
```

Rebuild and push a new Docker image after updating.

---

## Implementation Status

| Module | File | Status |
|--------|------|--------|
| Foundation | `types.py` `constants.py` `config/` | Complete |
| Data | `data/ztf.py` | Complete |
| Data | `data/rubin.py` | Stub — TAP query pending |
| Data | `data/simulation.py` | Stub — Lcurve integration pending |
| Features | `features/statistics.py` | Complete — periodfind BasicStats backend |
| Features | `features/period.py` | Complete — CE/AOV/LS/MHF via periodfind |
| Features | `features/dmdt.py` | Complete — periodfind DmDt backend |
| Features | `features/catalog.py` | Stub — Gaia TAP query pending |
| Features | `features/pipeline.py` | Complete |
| Models | `models/base.py` | Complete |
| Models | `models/xgboost.py` | Reference pattern (predict/save/load shells) |
| Training | `training/dataset.py` | Partial — label join complete; parquet load stub |
| Training | `training/trainer.py` | Shell — training loop pending |
| Inference | `inference/postprocess.py` | Complete |
| Inference | `inference/loader.py` | Complete |
| Inference | `inference/predictor.py` | Shell — depends on model implementation |
