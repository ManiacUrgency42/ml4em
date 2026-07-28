"""
Pydantic configuration schema for ml4em.

Design principles
-----------------
Each section maps to exactly one pipeline layer.  A layer receives only
its own config section — nothing else.  This enforces the strict module
separation: changing a training hyperparameter cannot affect feature
extraction, and vice versa.

Layer → Config section mapping
-------------------------------
Data layer        →  PipelineConfig.sources.ztf
Feature layer     →  PipelineConfig.features
Training layer    →  PipelineConfig.training
Inference layer   →  PipelineConfig.inference
All layers        →  PipelineConfig.storage  (shared path roots)

Defaults are set so that PipelineConfig() is a fully valid config with no
config.yaml needed.  Users only override what differs from defaults.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ml4em.constants import (
    DMDT_DM_MAX,
    DMDT_DM_MIN,
    DMDT_DT_MAX,
    DMDT_DT_MIN,
    N_DM_BINS,
    N_DT_BINS,
    XMATCH_RADIUS_ARCSEC,
    ZTF_BANDS,
    ZTF_MIN_CADENCE_DAYS,
)


# ---------------------------------------------------------------------------
# Sources  (data layer config)
# ---------------------------------------------------------------------------

class ZTFConfig(BaseModel):
    """Connection and data-selection settings for ZTF via Kowalski.

    The API token is NOT stored here.
    Load it via ml4em.config.get_ztf_token() → reads ML4EM_ZTF_TOKEN from env.
    """

    # Without this, assigning an invalid value after construction sticks
    # silently and only fails much later, inside whatever consumes it.
    model_config = {"validate_assignment": True}

    host     : str = "melman.caltech.edu"
    port     : int = 443
    protocol : str = "https"
    timeout  : int = 300   # seconds

    # Source catalog to query for light curves.
    # ZTF_sources_84525009 is the largest available collection on melman (DR20).
    collection_sources : str = "ZTF_sources_84525009"

    # Restrict to observations before this HJD (end of a specific data release).
    # None → use every epoch in collection_sources.
    #
    # Default is None rather than ZTF_DR16_MAX_HJD.  collection_sources points
    # at a DR20 collection, so pairing it with the DR16 cutoff would give DR20
    # sky coverage with DR16 time coverage — roughly two years of epochs
    # silently discarded, shortening every baseline and coarsening the
    # frequency grid.  Set this explicitly (to ZTF_DR16_MAX_HJD) only when
    # deliberately reproducing a DR16-era result.
    max_timestamp_hjd : Optional[float] = None

    # Bands to fetch. Each band produces one LightCurve per source.
    bands : list[str] = list(ZTF_BANDS)

    # Drop observations closer together than this before feature extraction.
    # Removes intra-night duplicates that bias period-finding.
    min_cadence_days : float = ZTF_MIN_CADENCE_DAYS

    # Server-side minimum epoch count for source discovery (near_ids()).
    # Kowalski indexes an `nobs` field per source, so filtering here rejects
    # sparsely-sampled sources before any light curve data crosses the network.
    # Without it a region query returns every source in the footprint and the
    # pipeline pays the full transfer cost for rows it will immediately discard
    # against FeatureConfig.min_observations.  Matches scope-ml's `minobs`
    # filter in get_quad_ids.py.  Set to 0 to disable.
    min_nobs : int = 50

    # Number of parallel Kowalski threads for batch light curve fetching.
    # scope-ml equivalent: Ncore in get_lightcurves_via_ids().
    # On MSI set to 8–16; leave at 1 for local/single-source runs.
    n_workers : int = 1

    # ZTF programid filter: 1=public, 2=ZTF partnership, 3=Caltech private.
    # Matches scope-ml's default program_id_selector=[1,2,3].
    program_ids : list[int] = Field(default_factory=lambda: [1, 2, 3])

    # Maximum source IDs per Kowalski find query.
    # Matches scope-ml's limit_per_query=1000 in get_lightcurves_via_ids().
    # fetch_batch() sends queries in a sliding window of this size.
    limit_per_query : int = 1000

    @field_validator("bands")
    @classmethod
    def _valid_bands(cls, v: list[str]) -> list[str]:
        bad = set(v) - set(ZTF_BANDS)
        if bad:
            raise ValueError(f"Unknown ZTF bands: {bad}. Valid: {set(ZTF_BANDS)}")
        return v



class SourcesConfig(BaseModel):
    """All data source configurations, grouped."""
    ztf   : ZTFConfig   = Field(default_factory=ZTFConfig)


# ---------------------------------------------------------------------------
# Features  (feature layer config)
# ---------------------------------------------------------------------------

class PeriodConfig(BaseModel):
    """Period-finding settings.

    Multiple algorithms run in parallel; results are compared via
    agreement scoring.  The period with the highest cross-algorithm
    agreement is used for Fourier decomposition.

    Algorithm identifiers (subset of periodfind library)
    ----------------------------------------------------
    CE   Conditional Entropy
    AOV  Analysis of Variance
    LS   Lomb-Scargle
    BLS  Box Least Squares  (best for flat-bottomed eclipses)
    FPW  Fast Period-finding with Wavelets
    MHF  Multi-Harmonic Fit
    """

    # Without this, assigning an invalid value after construction sticks
    # silently and only fails much later, inside whatever consumes it.
    model_config = {"validate_assignment": True}

    algorithms      : list[str]    = ["CE", "AOV", "LS", "MHF", "FPW"]
    min_period_days : float        = 0.01   # days
    max_period_days : float        = 10.0   # days
    top_n_periods   : int          = 10     # periods retained per algorithm before scoring (matches scope-ml)
    min_agreement   : int          = 2      # algorithms that must agree → "high confidence"

    # Frequency-spaced grid step: df = 1 / (samples_per_peak * baseline).
    # Matches scope-ml's default of 10. Higher values = finer grid = slower.
    samples_per_peak : float = 10.0

    # Peak extraction granularity.  A real peak occupies many adjacent grid
    # points, so naively taking the top_n_periods best bins returns
    # top_n_periods samples of the *same* peak.  The periodogram is instead
    # split into (n_chunks_multiplier * top_n_periods) contiguous chunks, the
    # best point in each chunk is taken, and those winners are ranked.
    # Matches scope-ml's extract_top_n_periods default of 3.
    n_chunks_multiplier : int = 3

    _KNOWN: ClassVar[frozenset] = frozenset({"CE", "AOV", "LS", "FPW", "BLS", "MHF"})

    @field_validator("algorithms")
    @classmethod
    def _valid_algorithms(cls, v: list[str]) -> list[str]:
        # Strip legacy "E" prefix (ECE → CE) for scope-ml backward compat
        normed = [a[1:] if a.startswith("E") and a[1:] in cls._KNOWN else a for a in v]
        bad = set(normed) - cls._KNOWN
        if bad:
            raise ValueError(f"Unknown algorithms: {bad}. Known: {cls._KNOWN}")
        return normed

    @field_validator("min_period_days", "max_period_days")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Period bound must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _ordered_period_bounds(self):
        # An inverted range yields f_min > f_max and therefore an empty
        # frequency grid, which surfaces far downstream as "no periods found"
        # rather than as a config error.
        if self.min_period_days >= self.max_period_days:
            raise ValueError(
                f"min_period_days ({self.min_period_days}) must be < "
                f"max_period_days ({self.max_period_days})"
            )
        return self


class DmdtConfig(BaseModel):
    """dm/dt histogram parameters.

    The histogram is a 2-D image (Δt vs. Δmag) over all observation pairs.
    These parameters must stay fixed within a project — changing them
    invalidates previously computed histograms and requires retraining.
    """

    # Without this, assigning an invalid value after construction sticks
    # silently and only fails much later, inside whatever consumes it.
    model_config = {"validate_assignment": True}

    n_dt_bins : int   = N_DT_BINS     # number of time-difference bins
    n_dm_bins : int   = N_DM_BINS     # number of magnitude-difference bins
    dt_min    : float = DMDT_DT_MIN   # minimum Δt  (days, log-spaced axis)
    dt_max    : float = DMDT_DT_MAX   # maximum Δt  (days)
    dm_min    : float = DMDT_DM_MIN   # minimum Δmag (mag, linear axis)
    dm_max    : float = DMDT_DM_MAX   # maximum Δmag (mag)

    @field_validator("n_dt_bins", "n_dm_bins")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Bin count must be ≥ 1, got {v}")
        return v

    @model_validator(mode="after")
    def _ordered_edges(self):
        # The Δt axis is built with np.logspace(log10(dt_min), log10(dt_max)),
        # so a non-positive dt_min gives -inf/nan edges and an inverted range
        # gives descending edges.  Either way the histogram is silently
        # meaningless, so both are rejected here.
        if self.dt_min <= 0:
            raise ValueError(f"dt_min must be positive (log axis), got {self.dt_min}")
        if self.dt_min >= self.dt_max:
            raise ValueError(
                f"dt_min ({self.dt_min}) must be < dt_max ({self.dt_max})"
            )
        if self.dm_min >= self.dm_max:
            raise ValueError(
                f"dm_min ({self.dm_min}) must be < dm_max ({self.dm_max})"
            )
        return self


class CatalogConfig(BaseModel):
    """Gaia cross-match settings for the feature layer.

    Note: Gaia is NOT a light curve source — it is a feature enrichment step.
    The CatalogExtractor queries Gaia EDR3 for each source's (ra, dec) and
    appends parallax / colour / astrometric_excess_noise to the FeatureVector.
    """

    # Without this, assigning an invalid value after construction sticks
    # silently and only fails much later, inside whatever consumes it.
    model_config = {"validate_assignment": True}

    xmatch_radius_arcsec : float = XMATCH_RADIUS_ARCSEC
    include_gaia         : bool  = True

    # Number of parallel Kowalski threads for batch Gaia cone searches.
    # scope-ml equivalent: Ncore in external_xmatch.py (splits radec dict
    # across threads for simultaneous cone_search queries).
    # On MSI set to 8–16; leave at 1 for local/single-source runs.
    n_workers : int = 1


class FeatureConfig(BaseModel):
    """All feature extraction settings, grouped."""

    # Overriding a field after load — cfg.features.device = "cuda" — is the
    # normal way scripts and benchmarks reconfigure a run.  Without this,
    # pydantic skips field validators on assignment, so the device alias
    # normalisation below would apply to YAML but not to that assignment, and
    # the unnormalised value would reach periodfind.set_device() and raise.
    model_config = {"validate_assignment": True}

    period  : PeriodConfig  = Field(default_factory=PeriodConfig)
    dmdt    : DmdtConfig    = Field(default_factory=DmdtConfig)
    catalog : CatalogConfig = Field(default_factory=CatalogConfig)

    # Minimum observations required to attempt feature extraction.
    # Sources below this are skipped and logged as insufficient data.
    min_observations : int = 50

    # Whether to compute the dm/dt histogram.
    # Set False for XGBoost-only runs to skip the O(N²) pairwise computation.
    compute_dmdt : bool = True

    # periodfind device selection.
    # 'auto' tries GPU (nvidia-smi check), falls back to CPU if unavailable.
    #
    # periodfind.set_device() itself accepts only the literals 'cpu' and 'gpu'.
    # 'auto' is an ml4em-level value resolved by the pipeline before the call;
    # 'cuda' is accepted as a convenience alias and normalised to 'gpu' here so
    # a plausible config value cannot reach set_device() and raise.
    device : str = "auto"

    @field_validator("device")
    @classmethod
    def _valid_device(cls, v: str) -> str:
        norm = {"auto": "auto", "cpu": "cpu", "gpu": "gpu", "cuda": "gpu"}
        key = v.strip().lower()
        if key not in norm:
            raise ValueError(
                f"Unknown device {v!r}. Valid: 'auto', 'cpu', 'gpu' (alias 'cuda')."
            )
        return norm[key]

    # Number of sources processed per periodfind call.
    # Controls GPU memory usage — lower this if you hit OOM on large light curves.
    feature_batch_size : int = 1000

    @field_validator("feature_batch_size")
    @classmethod
    def _positive_batch(cls, v: int) -> int:
        # Chunking uses range(0, n, batch_size): zero raises mid-run, and a
        # negative value produces no chunks at all, so the pipeline returns
        # nothing without reporting an error.
        if v < 1:
            raise ValueError(f"feature_batch_size must be ≥ 1, got {v}")
        return v

    # Directory for batch checkpoint files.
    # When set, the pipeline saves a checkpoint after every feature_batch_size
    # chunk so that a crashed MSI job can resume from where it left off.
    # None → no checkpointing (default for local/single-source runs).
    # Example MSI path: /scratch.global/jin00404/ml4em/checkpoints/run_001
    # Each run should use a unique subdirectory to avoid cross-run conflicts.
    checkpoint_dir : Optional[str] = None


# ---------------------------------------------------------------------------
# Storage  (shared across all layers)
# ---------------------------------------------------------------------------

class StorageConfig(BaseModel):
    """File paths used by every layer to read and write pipeline artifacts.

    All paths are strings to avoid platform-specific Path issues in YAML.
    Relative paths are resolved from the working directory at runtime.
    On MSI, override these in config.yaml with absolute scratch paths.

    Input files (read by the pipeline)
    ------------------------------------
    catalog_path  CSV of known target sources with ra, dec columns.
                  Used by scripts/prepare_labels.py to look up ZTF IDs
                  via cone search and produce labels_path.
                  Local default:  data/wdb_sources.csv
                  MSI example:    /scratch.global/jin00404/ml4em/data/wdb_sources.csv

    labels_path   CSV produced by prepare_labels.py.
                  Columns: source_id (ZTF integer _id as str), label (0 or 1).
                  Read by FeatureDataset._load_labels() during training.
                  Local default:  data/labels.csv
                  MSI example:    /scratch.global/jin00404/ml4em/data/labels.csv

    Output directories (written by the pipeline)
    ---------------------------------------------
    Feature layer  →  writes to features_dir
    Training layer →  reads from features_dir, writes to models_dir
    Inference layer→  reads from features_dir + models_dir,
                      writes to predictions_dir
    """

    # Without this, assigning an invalid value after construction sticks
    # silently and only fails much later, inside whatever consumes it.
    model_config = {"validate_assignment": True}

    # Input files — must exist before running the pipeline.
    # Place them in ml4em/data/ locally (gitignored).
    # Override in config.yaml on MSI with absolute scratch paths.
    catalog_path : str = "data/wdb_sources.csv"   # ra/dec catalog of target sources
    labels_path  : str = "data/labels.csv"         # source_id,label — produced by prepare_labels.py

    # Output directories — created automatically by each layer
    features_dir    : str = "features"     # parquet files, one per ZTF quadrant
    models_dir      : str = "models"       # trained model weights + feature scaler stats
    predictions_dir : str = "predictions"  # per-source WDB probability scores


# ---------------------------------------------------------------------------
# Training  (training layer config)
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    """Hyperparameters for model training.

    The training layer reads features from StorageConfig.features_dir,
    trains, and writes weights to StorageConfig.models_dir.
    These settings have no effect on feature extraction or inference.
    """

    # Without this, assigning an invalid value after construction sticks
    # silently and only fails much later, inside whatever consumes it.
    model_config = {"validate_assignment": True}

    batch_size    : int   = 64
    learning_rate : float = 3e-4
    max_epochs    : int   = 100
    patience      : int   = 20     # early-stopping: epochs without improvement

    val_fraction  : float = 0.1    # fraction held out for validation
    test_fraction : float = 0.1    # fraction held out for final evaluation
    seed          : int   = 42

    @field_validator("val_fraction", "test_fraction")
    @classmethod
    def _valid_fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Fraction must be in (0, 1), got {v}")
        return v


# ---------------------------------------------------------------------------
# Inference  (inference layer config)
# ---------------------------------------------------------------------------

class InferenceConfig(BaseModel):
    """Settings for the inference layer.

    The inference layer reads features from StorageConfig.features_dir,
    loads the model from model_path, and writes results to
    StorageConfig.predictions_dir.
    """

    # Without this, assigning an invalid value after construction sticks
    # silently and only fails much later, inside whatever consumes it.
    model_config = {"validate_assignment": True}

    # Path to the trained model weights file.
    # None means "use the latest model found in StorageConfig.models_dir".
    model_path : Optional[str] = None

    # How many feature rows to process per forward pass.
    batch_size : int = 10_000

    # Probability thresholds for the qualitative confidence label.
    # A source with probability >= high_threshold → "high",
    # >= medium_threshold → "medium", otherwise → "low".
    confidence_thresholds : dict[str, float] = Field(
        default={"high": 0.9, "medium": 0.7}
    )

    @field_validator("confidence_thresholds")
    @classmethod
    def _valid_thresholds(cls, v: dict[str, float]) -> dict[str, float]:
        required = {"high", "medium"}
        missing = required - v.keys()
        if missing:
            raise ValueError(f"confidence_thresholds must contain keys: {required}. Missing: {missing}")
        for key, val in v.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Threshold '{key}' must be in [0, 1], got {val}")
        if v["high"] <= v["medium"]:
            raise ValueError("'high' threshold must be greater than 'medium'")
        return v


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """Root configuration for the ml4em pipeline.

    Maps directly to the structure of config.yaml.
    Calling PipelineConfig() with no arguments returns a fully valid config
    using all defaults — no file required.

    This config is science-case agnostic.  The researcher's science case
    (WDB, AGN, RR Lyrae, etc.) is defined by the model trained and the
    labels used — not by this config.

    Minimal config.yaml example (override only what differs):

        sources:
          ztf:
            collection_sources: ZTF_sources_20240515
        features:
          period:
            algorithms: [CE, AOV, LS, BLS]
            min_period_days: 0.01
            max_period_days: 10.0
        storage:
          features_dir: /data/ml4em/features
    """

    sources  : SourcesConfig   = Field(default_factory=SourcesConfig)
    features : FeatureConfig   = Field(default_factory=FeatureConfig)
    storage  : StorageConfig   = Field(default_factory=StorageConfig)
    training : TrainingConfig  = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
