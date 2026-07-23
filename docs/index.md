# ml4em

**Production-scale machine learning for electromagnetic light curve analysis.**

ml4em is a modular, science-case-agnostic ML pipeline designed for **massive HPC/GPU
production runs** — not single-source exploratory work.  The architecture is built around
processing tens of thousands of ZTF sources per batch on GPU clusters (MSI, TACC, etc.):
GPU-batched period finding via periodfind (Rust/CUDA), batched Kowalski queries,
chunked FeatureVector assembly, and parquet I/O at quad scale.

If you are running on a single light curve to explore a result, use the benchmark script
(`scripts/benchmark_single.py`).  The pipeline itself is sized for production.

!!! warning "Scale is the design constraint"
    Every architectural decision — batch sizes, GPU memory management, Kowalski query
    batching, the FeatureVector data contract — is optimized for throughput across
    100k+ sources, not latency on one.  Do not judge performance from single-source runs.

!!! note "Science-case agnostic by design"
    ml4em does not decide what you are looking for. The target class (white dwarf binaries,
    AGN, RR Lyrae, eclipsing binaries, etc.) is defined entirely by the labels you provide
    and the model you choose. The pipeline is identical in every case.

---

## Where to start

<div class="grid cards" markdown>

- **New to the codebase?**

    Start with the architecture overview to understand how the six layers fit together,
    then read the data contracts page to understand the three types that flow between them.

    [Architecture overview →](architecture/overview.md)

- **Unfamiliar with the astrophysics?**

    The Background section explains every domain concept used in this codebase — light
    curves, surveys, period-finding algorithms, variability statistics — from first
    principles, no astronomy background assumed.

    [Background →](background/index.md)

- **Ready to run something?**

    The Deployment section gets you from a clone to a running pipeline on MSI.

    [Deployment →](deployment.md)

- **Adding to the library?**

    Step-by-step guides for adding a new data source, feature extractor, or model —
    each requires exactly one new file.

    [Guides →](guides/add-data-source.md)

</div>

---

## Implementation status

| Module | File | Status |
|--------|------|--------|
| Foundation | `types.py` `constants.py` `config/` | Complete |
| Data | `data/ztf.py` | Complete |
| Data | `data/rubin.py` | Stub — TAP query pending |
| Data | `data/simulation.py` | Stub — Lcurve integration pending |
| Features | `features/statistics.py` | Complete — periodfind BasicStats backend |
| Features | `features/period.py` | Complete — CE/AOV/LS/MHF via periodfind |
| Features | `features/dmdt.py` | Complete — periodfind DmDt backend |
| Features | `features/catalog.py` | Complete — Kowalski Gaia_EDR3 cone search, batch |
| Features | `features/pipeline.py` | Complete |
| Models | `models/base.py` | Complete |
| Models | `models/xgboost.py` | Reference pattern (predict/save/load shells) |
| Training | `training/dataset.py` | Partial — label join complete; parquet load stub |
| Training | `training/trainer.py` | Shell — training loop pending |
| Inference | `inference/postprocess.py` | Complete |
| Inference | `inference/loader.py` | Complete |
| Inference | `inference/predictor.py` | Shell — depends on model implementation |
