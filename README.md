# ml4em

Machine learning for electromagnetic light curve analysis.

A modular, general-purpose library for processing photometric time series data from astronomical surveys. Designed for tasks like White Dwarf Binary (WDB) detection, but not coupled to any single science case.

---

## Design Philosophy

ml4em is built around three principles pulled directly from the ml4gw architecture:

1. **Strict layer separation.** Each pipeline layer owns one config section and one data type. A layer never reads another layer's config.
2. **Protocol-based composition, not inheritance.** Data sources and feature extractors are pluggable via structural typing (`typing.Protocol`). No base class registration required — implement the right methods and it works.
3. **Contracts at every boundary.** Three dataclasses define the shared language between all layers. No raw tuples or dicts cross a module boundary.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Foundation                                                   │
│  types.py · constants.py · config/schema.py · config/loader  │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│  Data Layer         data/                                     │
│  LightCurveSource Protocol                                    │
│  ZTFSource · RubinSource · SimulatedSource                    │
│  output: list[LightCurve]                                     │
└──────────────────────────────┬───────────────────────────────┘
                               │ LightCurve
┌──────────────────────────────▼───────────────────────────────┐
│  Feature Layer      features/                                 │
│  FeatureExtractor Protocol                                    │
│  StatisticsExtractor · PeriodExtractor                        │
│  DmdtExtractor · CatalogExtractor                             │
│  FeaturePipeline (composer)                                   │
│  output: FeatureVector                                        │
└──────────────┬───────────────────────────┬───────────────────┘
               │ FeatureVector             │ FeatureVector
┌──────────────▼──────────┐   ┌───────────▼───────────────────┐
│  Training Layer         │   │  Inference Layer               │
│  (pending)              │   │  (pending)                     │
│  output: model weights  │   │  output: WDBCandidate          │
└─────────────────────────┘   └───────────────────────────────┘
```

Training and inference are siblings — they share the feature layer output but diverge completely after that. Neither touches the other's config or artifacts.

---

## Data Contracts

Three dataclasses are the only shared language between layers. Nothing else crosses a module boundary.

```python
LightCurve      # Data layer → Feature layer
FeatureVector   # Feature layer → Training / Inference
WDBCandidate    # Inference layer → Output  (frozen=True)
```

**`LightCurve`** — single-band photometric time series for one source.

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `str` | Survey-native ID |
| `time` | `ndarray (N,)` | Observation times in MJD |
| `mag` | `ndarray (N,)` | Apparent magnitude |
| `mag_err` | `ndarray (N,)` | 1-sigma magnitude uncertainty |
| `band` | `Band` | Photometric filter (`g`, `r`, `i`, …) |
| `survey` | `Survey` | `"ztf"` \| `"rubin"` \| `"simulated"` |
| `ra`, `dec` | `float` | Sky position, decimal degrees (J2000) |

**`FeatureVector`** — 43+ features extracted from a `LightCurve`, ready for ML.

| Group | Features | Notes |
|-------|----------|-------|
| LC statistics | 22 scalars | chi2red, Stetson I/J/K, skew, kurtosis, MAD, IQR, … |
| Period detection | 3 | period (days), significance, algorithm name |
| Fourier decomposition | 14 | amplitude, phase, relative amplitudes of harmonics 1–5 |
| dm/dt histogram | `(26, 26)` ndarray | pairwise (Δt, Δmag) image; `None` if not computed |
| Gaia cross-match | 4 | parallax, parallax_error, BP-RP colour, RUWE |

All float fields default to `np.nan`. Partial feature extraction is explicit — uncomputed fields are `nan`, not absent.

**`WDBCandidate`** — inference result for a single source. Immutable (`frozen=True`).

| Field | Description |
|-------|-------------|
| `source_id`, `ra`, `dec`, `survey` | Source identity |
| `probability` | Model output in [0, 1] |
| `period`, `period_algorithm` | Best detected period |
| `confidence` | `"high"` \| `"medium"` \| `"low"` |

---

## Layer Details

### Foundation

```
src/ml4em/
├── types.py          # LightCurve, FeatureVector, WDBCandidate
├── constants.py      # Physical + survey constants, dmdt bin parameters
└── config/
    ├── schema.py     # Pydantic models — WDBConfig and all sub-configs
    └── loader.py     # YAML loader + env-var secret accessors
```

**Config design.** `WDBConfig` is the pipeline architecture expressed in configuration form. Each section maps to exactly one layer:

```
WDBConfig
├── sources.ztf       →  ZTFSource
├── sources.rubin     →  RubinSource
├── features.period   →  PeriodExtractor
├── features.dmdt     →  DmdtExtractor
├── features.catalog  →  CatalogExtractor
├── storage           →  all layers (file paths only)
├── training          →  Trainer
└── inference         →  Predictor
```

Pydantic validation catches config errors at startup, before any data is fetched. Validators encode domain knowledge: `PeriodConfig` rejects negative period bounds; `InferenceConfig` rejects a "high" confidence threshold below "medium".

**Secrets.** Tokens (`WDB_ZTF_TOKEN`, `WDB_RUBIN_TOKEN`) are never stored in `WDBConfig`. They are read from environment variables at call time. Config files can be committed freely.

---

### Data Layer (`data/`)

```
src/ml4em/data/
├── base.py           # LightCurveSource Protocol
├── ztf.py            # ZTFSource — Kowalski/penquins client
├── rubin.py          # RubinSource — TAP stub
└── simulation.py     # SimulatedSource — Lcurve stub
```

**`LightCurveSource` Protocol.** Any class with `fetch(source_id)` and `fetch_batch(source_ids)` satisfies the protocol. The feature layer accepts any compliant object — it never imports `ZTFSource` directly.

```python
class LightCurveSource(Protocol):
    def fetch(self, source_id: str) -> list[LightCurve]: ...
    def fetch_batch(self, source_ids: list[str]) -> list[LightCurve]: ...
```

Swapping ZTF for Rubin is one line at the call site. Adding a new survey is one new file — nothing else changes.

**`ZTFSource`** (implemented). Fetches from the Kowalski database via `penquins`. Each ZTF `_id` encodes a single (sky position, band) pair. `fetch_batch` sends a single batched Kowalski `find` query. Applies two data quality steps: discard flagged epochs (`catflags != 0`) and remove intra-night duplicates closer than `min_cadence_days` (default 30 min) to avoid period-finding aliases.

**`RubinSource`** (stub). Planned TAP query against `dp1.ForcedSource ⋈ dp1.Visit ⋈ dp1.Object`. One `objectId` may return up to six `LightCurve` objects (one per band). Pending Rubin DP1 schema review.

**`SimulatedSource`** (stub). Will wrap Tom Marsh's Lcurve code to produce physics-based synthetic WDB light curves. Planned interface: `source_id` is a path to an Lcurve `.mod` parameter file. Noise and realistic cadence will be injected after model evaluation. Used for training when labeled survey data is insufficient.

---

### Feature Layer (`features/`)

```
src/ml4em/features/
├── base.py           # FeatureExtractor Protocol
├── statistics.py     # StatisticsExtractor — 22 LC statistics
├── period.py         # PeriodExtractor — multi-algorithm period finding + Fourier
├── dmdt.py           # DmdtExtractor — 2-D pairwise histogram
├── catalog.py        # CatalogExtractor — Gaia EDR3 cross-match
└── pipeline.py       # FeaturePipeline — composer
```

**`FeatureExtractor` Protocol.** Any class with `extract(lcs: list[LightCurve]) -> dict[str, Any]` satisfies the protocol. Extractors receive all bands for one source and select the band(s) they need internally. They must never raise — return a partial or empty dict on failure so the pipeline continues.

**`FeaturePipeline`** composes extractors in order: statistics → period → dmdt → catalog. The factory method `FeaturePipeline.default(cfg.features)` builds the standard pipeline from config. Results are merged left-to-right into a `FeatureVector`. Sources with fewer than `min_observations` points (default 50) are skipped and returned as all-NaN vectors.

**Period finding.** `PeriodExtractor` runs multiple algorithms (CE, AOV, LS, BLS) across the configured period range and scores agreement across algorithms. The period with the highest cross-algorithm agreement is selected. After period detection, a Fourier series is fit to the phased light curve through the 5th harmonic.

**dm/dt histogram.** `DmdtExtractor` computes all pairwise (Δt, Δmag) values across the light curve and bins them into a 26×26 image. Time differences are log-spaced (minutes to years); magnitude differences are linear (±3 mag). The image is L2-normalised and stored as the `dmdt` field. Set `features.compute_dmdt: false` to skip this for XGBoost-only runs (avoids the O(N²) pairwise cost).

**Gaia cross-match.** `CatalogExtractor` queries Gaia EDR3 for the nearest source within 2 arcsec of the light curve's (ra, dec). Returns parallax, parallax_error, BP-RP colour, and RUWE. RUWE < 1.4 indicates a clean astrometric solution (Lindegren et al. 2021). Note: Gaia is not a light curve source — it is a per-source feature enrichment step in the feature layer.

---

### Training and Inference Layers

Not yet implemented. Planned interfaces:

- **Training**: reads `FeatureVector` objects from `storage.features_dir`, trains a model, writes weights to `storage.models_dir`.
- **Inference**: reads `FeatureVector` objects and a trained model, writes `WDBCandidate` results to `storage.predictions_dir`.

Config sections (`TrainingConfig`, `InferenceConfig`) are already defined.

---

## Dependencies

Core (`pip install ml4em`) requires only `numpy`, `pydantic`, `pyyaml`, `python-dotenv`. Every other dependency is opt-in:

```bash
pip install "ml4em[ztf]"        # ZTF via Kowalski
pip install "ml4em[rubin]"      # Rubin via TAP
pip install "ml4em[features]"   # astropy, scipy, numba, fast-histogram
pip install "ml4em[training]"   # torch, scikit-learn
pip install "ml4em[inference]"  # torch
pip install "ml4em[all]"        # everything
```

---

## Gaia Cross-Match Note

Gaia has two distinct roles in the WDB pipeline, handled at different stages:

1. **Candidate pre-filter (outside ml4em, one-time)** — the Gaia WD catalog (Gentile Fusillo et al. 2021, ~1.3M WD candidates) is cross-matched against the ZTF source catalog to produce the initial candidate list. This is a notebook-level operation in the upstream `wdb-ml` repo, not part of the feature pipeline.

2. **Per-source feature enrichment (inside ml4em)** — for each source being processed, `CatalogExtractor` appends Gaia astrometric properties to the `FeatureVector`. These are used by the classifier to confirm WD nature (high parallax = nearby, blue BP-RP = hot, low RUWE = clean astrometry).

---

## Repository Relation

```
wdb-ml/              (application repo — notebooks, cross-match, experiments)
  external/
    ml4em/           (this repo — reusable library, imported as submodule)
    scope-ml/        (existing ZTF variability library, reference implementation)
```

ml4em is infrastructure. Science-case-specific logic (labeling, cross-match notebooks, experiment configs) lives in `wdb-ml`.

---

## Implementation Status

| Layer | Status |
|-------|--------|
| Foundation (`types`, `constants`, `config`) | Complete |
| Data — ZTFSource | Complete |
| Data — RubinSource | Stub (TAP query pending) |
| Data — SimulatedSource | Stub (Lcurve integration pending) |
| Features — StatisticsExtractor | Complete |
| Features — PeriodExtractor | Complete |
| Features — DmdtExtractor | Complete |
| Features — CatalogExtractor | Complete |
| Features — FeaturePipeline | Complete |
| Training layer | Not started |
| Inference layer | Not started |
