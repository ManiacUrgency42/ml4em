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
Data layer        →  PipelineConfig.sources.ztf / PipelineConfig.sources.rubin
Feature layer     →  PipelineConfig.features
Training layer    →  PipelineConfig.training
Inference layer   →  PipelineConfig.inference
All layers        →  PipelineConfig.storage  (shared path roots)

Defaults are set so that PipelineConfig() is a fully valid config with no
config.yaml needed.  Users only override what differs from defaults.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from ml4em.constants import (
    DMDT_DM_MAX,
    DMDT_DM_MIN,
    DMDT_DT_MAX,
    DMDT_DT_MIN,
    N_DM_BINS,
    N_DT_BINS,
    RUBIN_BANDS,
    XMATCH_RADIUS_ARCSEC,
    ZTF_BANDS,
    ZTF_DR16_MAX_HJD,
    ZTF_MIN_CADENCE_DAYS,
)


# ---------------------------------------------------------------------------
# Sources  (data layer config)
# ---------------------------------------------------------------------------

class ZTFConfig(BaseModel):
    """Connection and data-selection settings for ZTF via Kowalski.

    The API token is NOT stored here.
    Load it via ml4em.config.get_ztf_token() → reads WDB_ZTF_TOKEN from env.
    """

    host     : str = "kowalski.caltech.edu"
    port     : int = 443
    protocol : str = "https"
    timeout  : int = 300   # seconds

    # Source catalog to query for light curves
    collection_sources : str = "ZTF_sources_20240515"

    # Restrict to observations before this HJD (end of a specific data release).
    # None → use all available data.
    max_timestamp_hjd : Optional[float] = ZTF_DR16_MAX_HJD

    # Bands to fetch. Each band produces one LightCurve per source.
    bands : list[str] = list(ZTF_BANDS)

    # Drop observations closer together than this before feature extraction.
    # Removes intra-night duplicates that bias period-finding.
    min_cadence_days : float = ZTF_MIN_CADENCE_DAYS

    @field_validator("bands")
    @classmethod
    def _valid_bands(cls, v: list[str]) -> list[str]:
        bad = set(v) - set(ZTF_BANDS)
        if bad:
            raise ValueError(f"Unknown ZTF bands: {bad}. Valid: {set(ZTF_BANDS)}")
        return v


class RubinConfig(BaseModel):
    """Connection and table settings for Rubin DP1 via TAP.

    The API token is NOT stored here.
    Load it via ml4em.config.get_rubin_token() → reads WDB_RUBIN_TOKEN from env.
    """

    tap_url : str = "https://data.lsst.cloud/api/tap"
    timeout : int = 300

    table_object        : str = "dp1.Object"
    table_forced_source : str = "dp1.ForcedSource"
    table_visit         : str = "dp1.Visit"

    bands    : list[str]      = list(RUBIN_BANDS)
    band_map : dict[str, int] = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    # Optional path to local parquet cache (offline / HPC use).
    data_path : Optional[str] = None

    @field_validator("bands")
    @classmethod
    def _valid_bands(cls, v: list[str]) -> list[str]:
        bad = set(v) - set(RUBIN_BANDS)
        if bad:
            raise ValueError(f"Unknown Rubin bands: {bad}. Valid: {set(RUBIN_BANDS)}")
        return v


class SourcesConfig(BaseModel):
    """All data source configurations, grouped."""
    ztf   : ZTFConfig   = Field(default_factory=ZTFConfig)
    rubin : RubinConfig = Field(default_factory=RubinConfig)


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

    algorithms      : list[str] = ["CE", "AOV", "LS", "BLS"]
    min_period_days : float     = 0.01   # days — override for your science case
    max_period_days : float     = 10.0   # days — override for your science case
    top_n_periods   : int     = 3    # periods retained per algorithm before scoring
    min_agreement   : int     = 2    # algorithms that must agree → "high confidence"

    _KNOWN = frozenset({"CE", "AOV", "LS", "FPW", "BLS", "MHF"})

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


class DmdtConfig(BaseModel):
    """dm/dt histogram parameters.

    The histogram is a 2-D image (Δt vs. Δmag) over all observation pairs.
    These parameters must stay fixed within a project — changing them
    invalidates previously computed histograms and requires retraining.
    """

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


class CatalogConfig(BaseModel):
    """Gaia cross-match settings for the feature layer.

    Note: Gaia is NOT a light curve source — it is a feature enrichment step.
    The CatalogExtractor queries Gaia EDR3 for each source's (ra, dec) and
    appends parallax / colour / RUWE to the FeatureVector.
    """

    xmatch_radius_arcsec : float = XMATCH_RADIUS_ARCSEC
    include_gaia         : bool  = True


class FeatureConfig(BaseModel):
    """All feature extraction settings, grouped."""

    period  : PeriodConfig  = Field(default_factory=PeriodConfig)
    dmdt    : DmdtConfig    = Field(default_factory=DmdtConfig)
    catalog : CatalogConfig = Field(default_factory=CatalogConfig)

    # Minimum observations required to attempt feature extraction.
    # Sources below this are skipped and logged as insufficient data.
    min_observations : int = 50

    # Whether to compute the dm/dt histogram.
    # Set False for XGBoost-only runs to skip the O(N²) pairwise computation.
    compute_dmdt : bool = True


# ---------------------------------------------------------------------------
# Storage  (shared across all layers)
# ---------------------------------------------------------------------------

class StorageConfig(BaseModel):
    """File paths used by every layer to read and write pipeline artifacts.

    All paths are strings to avoid platform-specific Path issues in YAML.
    Relative paths are resolved from the working directory at runtime.

    Layer responsibilities
    ---------------------
    Feature layer  →  writes to features_dir
    Training layer →  reads from features_dir, writes to models_dir
    Inference layer→  reads from features_dir + models_dir,
                      writes to predictions_dir
    """

    features_dir    : str = "features"     # parquet files, one per ZTF quad / Rubin tract
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
