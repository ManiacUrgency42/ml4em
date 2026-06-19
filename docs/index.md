# ml4em

**Machine learning for electromagnetic light curve analysis.**

ml4em is a modular, science-case-agnostic ML pipeline library for classifying variable
astronomical sources from photometric time-series data. You supply training labels and
a model; the library handles data fetching, feature extraction, training, and inference.

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

    The Quick Start gets you from a clone to a running pipeline.

    [Quick Start →](quickstart.md)

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
| Features | `features/catalog.py` | Stub — Gaia TAP query pending |
| Features | `features/pipeline.py` | Complete |
| Models | `models/base.py` | Complete |
| Models | `models/xgboost.py` | Reference pattern (predict/save/load shells) |
| Training | `training/dataset.py` | Partial — label join complete; parquet load stub |
| Training | `training/trainer.py` | Shell — training loop pending |
| Inference | `inference/postprocess.py` | Complete |
| Inference | `inference/loader.py` | Complete |
| Inference | `inference/predictor.py` | Shell — depends on model implementation |
