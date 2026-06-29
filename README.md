# ml4em

Machine learning for electromagnetic light curve analysis.

A modular, science-case-agnostic ML pipeline library for classifying variable
astronomical sources from photometric time-series data (ZTF, Rubin/LSST, simulated).
You supply training labels and a model; the library handles data fetching, feature
extraction, training, and inference.

## Documentation

**Full documentation: https://maniacurgency42.github.io/ml4em/**

The docs cover:
- [Architecture overview](https://maniacurgency42.github.io/ml4em/architecture/overview/) — how the six layers fit together
- [Background](https://maniacurgency42.github.io/ml4em/background/) — astrophysics concepts explained for non-experts
- [Layer reference](https://maniacurgency42.github.io/ml4em/layers/foundation/) — I/O contracts for every module
- [Guides](https://maniacurgency42.github.io/ml4em/guides/add-data-source/) — adding new sources, extractors, and models

