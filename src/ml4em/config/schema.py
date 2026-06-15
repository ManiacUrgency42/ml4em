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
Data layer        →  WDBConfig.sources.ztf / WDBConfig.sources.rubin
Feature layer     →  WDBConfig.features
Training layer    →  WDBConfig.training
Inference layer   →  WDBConfig.inference
Training+Inference→  WDBConfig.classification  (what to look for)
All layers        →  WDBConfig.storage  (shared path roots)

Defaults are set so that WDBConfig() is a fully valid config with no
config.yaml needed.  Users only override what differs from defaults.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ml4em.constants import (
    DMDT_DM_MAX,
    DMDT_DM_MIN,
    DMDT_DT_MAX,
    DMDT_DT_MIN,
    N_DM_BINS,
    N_DT_BINS,
    RUBIN_BANDS,
    WDB_PERIOD_MAX_DAYS,
    WDB_PERIOD_MIN_DAYS,
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

    algorithms    : list[str] = ["CE", "AOV", "LS", "BLS"]
    min_period_days : float   = WDB_PERIOD_MIN_DAYS
    max_period_days : float   = WDB_PERIOD_MAX_DAYS
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
# Classification  (training + inference layer config)
# ---------------------------------------------------------------------------

class StellarClassConfig(BaseModel):
    """Physics-informed descriptor for one class of variable stellar object.

    Used by the training layer to assign positive/negative labels and by
    the feature layer to optionally narrow the period search range to a
    class-specific window.

    Fields
    ------
    label : str
        String label as it appears in the training data parquet
        (e.g. "wdb", "cv", "rr_lyrae").
    description : str
        Plain-language description of the object type.
    period_min_days, period_max_days : float | None
        Expected orbital or pulsation period range for this class.
        None means "no physics-based bound" (e.g. irregular variables).
        When a run targets this class, the feature layer can use these
        to override the global PeriodConfig bounds and focus the search.
    is_periodic : bool
        True if the class shows periodic variability.
        False for irregular classes (CVs in outburst, YSOs).
    """

    label            : str
    description      : str   = ""
    period_min_days  : Optional[float] = None
    period_max_days  : Optional[float] = None
    is_periodic      : bool  = True


# Physics-informed defaults for all classes the pipeline distinguishes.
# Grouped by relationship to white dwarfs.
_DEFAULT_CLASSES: dict[str, StellarClassConfig] = {

    # ── WD binary systems (target and closely related) ────────────────────

    "wdb": StellarClassConfig(
        label="wdb",
        description=(
            "White Dwarf Binary — detached, non-accreting WD + companion. "
            "Includes post-common-envelope binaries (WD+M dwarf) and "
            "double-WD systems. Primary detection signature: deep, narrow "
            "eclipses at short orbital period."
        ),
        period_min_days=0.01,    # ~14 min; lower limit set by ZTF phase coverage
        period_max_days=10.0,
        is_periodic=True,
    ),

    "am_cvn": StellarClassConfig(
        label="am_cvn",
        description=(
            "AM CVn — ultra-compact WD + WD or WD + He-star system. "
            "Helium-dominated accretion disk. Periods 5–65 min; the shortest "
            "known EM-detected compact binaries and primary LISA verification "
            "targets. Most ZTF observations alias-fold these periods."
        ),
        period_min_days=0.003,   # ~4.3 min (HM Cnc, shortest known)
        period_max_days=0.065,   # ~94 min
        is_periodic=True,
    ),

    "hw_vir": StellarClassConfig(
        label="hw_vir",
        description=(
            "HW Vir — hot subdwarf (sdB or sdO) primary + M-dwarf secondary. "
            "Deep primary eclipse (sdB eclipsed by cool companion) and "
            "prominent reflection effect. Period typically 0.07–0.3 d."
        ),
        period_min_days=0.07,
        period_max_days=0.30,
        is_periodic=True,
    ),

    "cv": StellarClassConfig(
        label="cv",
        description=(
            "Cataclysmic Variable — WD + Roche-lobe-filling donor, "
            "actively accreting. Light curve dominated by disk flickering, "
            "outbursts (dwarf novae), or steady accretion (nova-likes). "
            "Orbital period detectable in quiescence but LC is irregular."
        ),
        period_min_days=0.01,
        period_max_days=0.50,
        is_periodic=False,   # irregular outburst morphology
    ),

    # ── Non-degenerate eclipsing binaries (contaminants) ─────────────────

    "ea": StellarClassConfig(
        label="ea",
        description=(
            "Algol-type (EA) eclipsing binary — detached, non-degenerate. "
            "Flat light curve between eclipses; primary minimum deeper than "
            "secondary. Period 0.5–100 d. Primary contaminant for WDB "
            "because of similar eclipse morphology at short periods."
        ),
        period_min_days=0.5,
        period_max_days=100.0,
        is_periodic=True,
    ),

    "eb": StellarClassConfig(
        label="eb",
        description=(
            "Beta Lyrae-type (EB) eclipsing binary — semi-detached; one "
            "component fills its Roche lobe. Continuous brightness variation "
            "with unequal minima. Period 0.3–200 d."
        ),
        period_min_days=0.3,
        period_max_days=200.0,
        is_periodic=True,
    ),

    "ew": StellarClassConfig(
        label="ew",
        description=(
            "W Ursae Majoris (EW / W UMa) — overcontact binary sharing a "
            "common envelope. Nearly equal eclipse depths; quasi-sinusoidal "
            "light curve. Period tightly clustered 0.2–0.8 d. Superficially "
            "similar to short-period WDB systems."
        ),
        period_min_days=0.2,
        period_max_days=0.8,
        is_periodic=True,
    ),

    # ── Pulsating variables (contaminants) ───────────────────────────────

    "rr_lyrae": StellarClassConfig(
        label="rr_lyrae",
        description=(
            "RR Lyrae — radial pulsator on the horizontal branch. "
            "Asymmetric (sawtooth) light curve with rapid rise and slow "
            "decline. Period 0.2–1.0 d. Clearly distinguished from WDB "
            "by light curve shape but overlaps in period space."
        ),
        period_min_days=0.2,
        period_max_days=1.0,
        is_periodic=True,
    ),

    "delta_scuti": StellarClassConfig(
        label="delta_scuti",
        description=(
            "Delta Scuti / SX Phoenicis — short-period pulsator near the "
            "instability strip. Often multi-periodic. Period 0.01–0.3 d; "
            "overlaps with compact binary period space."
        ),
        period_min_days=0.01,
        period_max_days=0.30,
        is_periodic=True,
    ),

    # ── Rotating / spotted stars (contaminants) ──────────────────────────

    "rs_cvn": StellarClassConfig(
        label="rs_cvn",
        description=(
            "RS CVn — chromospherically active binary with large stellar "
            "spots. Quasi-sinusoidal light curve; shape evolves on months "
            "timescale as spots migrate. Period 1–14 d."
        ),
        period_min_days=1.0,
        period_max_days=14.0,
        is_periodic=True,
    ),
}


class ClassificationConfig(BaseModel):
    """Defines the classification task: what to detect and what to distinguish it from.

    Consumed by both the training layer (labeling) and the inference layer
    (interpreting model output).

    Fields
    ------
    target : str
        The positive class for this training/inference run.
        Must be a key in ``classes``.  All other classes in ``classes``
        are treated as negative examples in binary classification.
    classes : dict[str, StellarClassConfig]
        All object types the model is trained to distinguish.
        Keys are the string labels used in training data parquets.
    """

    target  : str = "wdb"
    classes : dict[str, StellarClassConfig] = Field(
        default_factory=lambda: dict(_DEFAULT_CLASSES)
    )

    @model_validator(mode="after")
    def _target_must_exist(self) -> "ClassificationConfig":
        if self.target not in self.classes:
            raise ValueError(
                f"target='{self.target}' is not in classes. "
                f"Available: {list(self.classes.keys())}"
            )
        return self

    @property
    def target_config(self) -> StellarClassConfig:
        """Return the StellarClassConfig for the current target class."""
        return self.classes[self.target]

    @property
    def contaminant_labels(self) -> list[str]:
        """Return labels of all non-target classes."""
        return [k for k in self.classes if k != self.target]


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class WDBConfig(BaseModel):
    """Root configuration for the ml4em pipeline.

    Maps directly to the structure of config.yaml.
    Calling WDBConfig() with no arguments returns a fully valid config
    using all defaults — no file required.

    Minimal config.yaml example (override only what differs):

        classification:
          target: wdb
        sources:
          ztf:
            collection_sources: ZTF_sources_20240515
        features:
          period:
            algorithms: [CE, AOV, LS, BLS]
        storage:
          features_dir: /data/ml4em/features
    """

    classification : ClassificationConfig = Field(default_factory=ClassificationConfig)
    sources        : SourcesConfig        = Field(default_factory=SourcesConfig)
    features       : FeatureConfig        = Field(default_factory=FeatureConfig)
    storage        : StorageConfig        = Field(default_factory=StorageConfig)
    training       : TrainingConfig       = Field(default_factory=TrainingConfig)
    inference      : InferenceConfig      = Field(default_factory=InferenceConfig)
