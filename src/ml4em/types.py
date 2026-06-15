"""
Core data contracts for ml4em.

Three types define every boundary between pipeline layers:

    LightCurve      raw photometric time series            (Data → Feature)
    FeatureVector   extracted feature set for one source   (Feature → Training / Inference)
    WDBCandidate    inference result for one source        (Inference → Output)

These types are the only shared language between modules.
Nothing in this file computes anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Aliases used across layers
# ---------------------------------------------------------------------------

Survey     = Literal["ztf", "rubin", "simulated"]
Band       = Literal["u", "g", "r", "i", "z", "y"]
Confidence = Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Interface 1: Data layer → Feature layer
# ---------------------------------------------------------------------------

@dataclass
class LightCurve:
    """Single-band photometric time series for one source.

    Produced by every data source (ZTF, Rubin, simulated Lcurve).
    All feature extractors consume this type — no raw tuples or dicts.

    Fields
    ------
    source_id : str
        Unique identifier within the survey (ZTF source ID, Rubin objectId,
        or simulation label).
    time : ndarray, shape (N,)
        Observation times in Modified Julian Date (MJD).
    mag : ndarray, shape (N,)
        Apparent magnitude at each epoch.
    mag_err : ndarray, shape (N,)
        1-sigma magnitude uncertainty at each epoch.
    band : Band
        Photometric filter ('g', 'r', 'i', etc.).
    survey : Survey
        Originating survey.
    ra : float
        Right ascension in decimal degrees (J2000).
    dec : float
        Declination in decimal degrees (J2000).
    """

    source_id : str
    time      : np.ndarray
    mag       : np.ndarray
    mag_err   : np.ndarray
    band      : Band
    survey    : Survey
    ra        : float
    dec       : float

    def __post_init__(self) -> None:
        if not (self.time.shape == self.mag.shape == self.mag_err.shape):
            raise ValueError(
                "time, mag, and mag_err must have identical shapes. "
                f"Got {self.time.shape}, {self.mag.shape}, {self.mag_err.shape}."
            )
        if self.time.ndim != 1:
            raise ValueError(
                f"Arrays must be 1-D, got shape {self.time.shape}."
            )

    @property
    def n_obs(self) -> int:
        return int(self.time.shape[0])


# ---------------------------------------------------------------------------
# Interface 2: Feature layer → Training / Inference
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """Fully extracted feature set for one source.

    Produced by the feature layer after processing a LightCurve.
    Consumed by both training (as a labeled sample) and inference (unlabeled).

    Feature groups
    --------------
    1. Light curve statistics  — 22 scalar features
    2. Period detection        —  3 features (value, significance, algorithm)
    3. Fourier decomposition   — 14 scalar features at the detected period
    4. dm/dt histogram         — (N_DM_BINS, N_DT_BINS) image; None if not computed
    5. Gaia cross-match        —  4 features; None if no counterpart found

    All float fields default to np.nan so that partial feature extraction is
    explicit rather than absent. The feature layer sets each field it computes;
    downstream code can check for nan to detect uncomputed features.
    """

    source_id : str
    survey    : Survey

    # ── 1. Light curve statistics ────────────────────────────────────────────
    # Computed directly from (time, mag, mag_err) by StatisticsExtractor.
    n_obs               : int   = 0
    median              : float = np.nan   # median magnitude
    wmean               : float = np.nan   # error-weighted mean magnitude
    chi2red             : float = np.nan   # reduced chi-squared vs. constant model
    roms                : float = np.nan   # ratio of median scatter to sigma
    wstd                : float = np.nan   # error-weighted standard deviation
    norm_peak_to_peak_amp : float = np.nan # (max−err − min+err) / (max−err + min+err)
    norm_excess_var     : float = np.nan   # normalised excess variance
    median_abs_dev      : float = np.nan   # median absolute deviation
    iqr                 : float = np.nan   # 25th–75th percentile range
    i60r                : float = np.nan   # 20th–80th percentile range
    i70r                : float = np.nan   # 15th–85th percentile range
    i80r                : float = np.nan   # 10th–90th percentile range
    i90r                : float = np.nan   #  5th–95th percentile range
    skew                : float = np.nan   # weighted skewness
    small_kurt          : float = np.nan   # Fisher kurtosis (small-sample corrected)
    inv_von_neumann     : float = np.nan   # inverse Von Neumann ratio (time-weighted)
    stetson_i           : float = np.nan   # Welch/Stetson I index
    stetson_j           : float = np.nan   # Stetson J index
    stetson_k           : float = np.nan   # Stetson K index
    anderson_darling    : float = np.nan   # Anderson-Darling normality statistic
    shapiro_wilk        : float = np.nan   # Shapiro-Wilk normality statistic

    # ── 2. Period detection ──────────────────────────────────────────────────
    # Best period chosen by agreement scoring across all run algorithms.
    period              : float = np.nan   # orbital period, days
    period_significance : float = np.nan   # algorithm-specific confidence score
    period_algorithm    : str   = ""       # algorithm that found this period

    # ── 3. Fourier decomposition at `period` ─────────────────────────────────
    # Fit: mag(t) = f1_a·cos(2πt/P) + f1_b·sin(2πt/P) + higher harmonics + offset
    f1_power    : float = np.nan   # fractional chi2 reduction from the fit
    f1_bic      : float = np.nan   # Bayesian Information Criterion of best-order fit
    f1_a        : float = np.nan   # cosine coefficient of first harmonic
    f1_b        : float = np.nan   # sine coefficient of first harmonic
    f1_amp      : float = np.nan   # amplitude of first harmonic  sqrt(a²+b²)
    f1_phi0     : float = np.nan   # phase of first harmonic  arctan2(a, b)
    f1_relamp1  : float = np.nan   # 2nd harmonic amplitude / 1st harmonic amplitude
    f1_relphi1  : float = np.nan   # 2nd harmonic relative phase
    f1_relamp2  : float = np.nan   # 3rd harmonic relative amplitude
    f1_relphi2  : float = np.nan   # 3rd harmonic relative phase
    f1_relamp3  : float = np.nan   # 4th harmonic relative amplitude
    f1_relphi3  : float = np.nan   # 4th harmonic relative phase
    f1_relamp4  : float = np.nan   # 5th harmonic relative amplitude
    f1_relphi4  : float = np.nan   # 5th harmonic relative phase

    # ── 4. dm/dt histogram (image feature for convolutional branch) ──────────
    # Pairwise (Δt, Δmag) histogram, L2-normalised.
    # Shape: (N_DM_BINS, N_DT_BINS). Set to None when not computed
    # (e.g. XGBoost-only inference, or insufficient observations).
    dmdt : Optional[np.ndarray] = field(default=None, repr=False)

    # ── 5. Gaia EDR3 cross-match ─────────────────────────────────────────────
    # Nearest Gaia source within XMATCH_RADIUS_ARCSEC. None if no match.
    # Used to confirm WD nature: blue BP-RP + high parallax + low RUWE.
    gaia_parallax       : Optional[float] = None  # mas
    gaia_parallax_error : Optional[float] = None  # mas
    gaia_bp_rp          : Optional[float] = None  # BP − RP colour, mag
    gaia_ruwe           : Optional[float] = None  # astrometric quality (< 1.4 = clean)


# ---------------------------------------------------------------------------
# Interface 3: Inference layer → Output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WDBCandidate:
    """Inference result for a single source.

    Produced by the inference post-processing step. Immutable once created.

    Fields
    ------
    source_id : str
        Survey source identifier.
    ra, dec : float
        Sky position in decimal degrees (J2000).
    survey : Survey
        Originating survey.
    probability : float
        Model output in [0, 1] — probability the source is a WDB.
    period : float
        Detected orbital period in days.
    period_algorithm : str
        Algorithm that found the period.
    confidence : Confidence
        Qualitative tier derived from probability thresholds defined in
        InferenceConfig.confidence_thresholds.
    """

    source_id        : str
    ra               : float
    dec              : float
    survey           : Survey
    probability      : float
    period           : float
    period_algorithm : str
    confidence       : Confidence
