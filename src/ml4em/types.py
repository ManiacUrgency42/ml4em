"""
Core data contracts for ml4em.

Four types define every boundary between pipeline layers:

    LightCurve      raw photometric time series            (Data → Feature)
    FeatureVector   extracted feature set for one source   (Feature → Training / Inference)
    LabeledSample   FeatureVector + ground-truth label     (Label prep → Training)
    Candidate       inference result for one source        (Inference → Output)

These types are the only shared language between modules.
Nothing in this file computes anything.  All types are generic — no assumption
is made about the science case (WDB, AGN, RR Lyrae, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Aliases used across layers
# ---------------------------------------------------------------------------

Survey     = Literal["ztf", "simulated"]
Band       = Literal["u", "g", "r", "i", "z", "y"]
Confidence = Literal["high", "medium", "low"]

# Time scales a LightCurve may be expressed in.  'relative' means days since
# an unspecified epoch, which is what simulated light curves usually produce.
_TIME_SYSTEMS = frozenset({"hjd", "mjd", "bjd", "jd", "relative"})


# ---------------------------------------------------------------------------
# Interface 1: Data layer → Feature layer
# ---------------------------------------------------------------------------

@dataclass
class LightCurve:
    """Single-band photometric time series for one source.

    Produced by every data source (ZTF, simulated Lcurve).
    All feature extractors consume this type — no raw tuples or dicts.

    Fields
    ------
    source_id : str
        Unique identifier within the survey (ZTF source ID,
        or simulation label).
    time : ndarray, shape (N,), float64
        Observation times, in days, in the system named by `time_system`.
    mag : ndarray, shape (N,), float64
        Apparent magnitude at each epoch.
    mag_err : ndarray, shape (N,), float64
        1-sigma magnitude uncertainty at each epoch.
    band : Band
        Photometric filter ('g', 'r', 'i', etc.).
    survey : Survey
        Originating survey.
    ra : float
        Right ascension in decimal degrees (J2000).
    dec : float
        Declination in decimal degrees (J2000).
    time_system : str
        Which time scale `time` is expressed in — 'hjd', 'mjd', 'bjd', 'jd',
        or 'relative'.  ZTF publishes HJD (~2.46e6), but simulated light
        curves and any future source may not.  Nothing downstream can
        distinguish the systems from the values alone, and every feature
        here depends only on time *differences*, so an unlabelled mix would
        produce plausible-looking nonsense rather than an error.  Recording
        the system makes a merge checkable instead of silent.

    Guarantees
    ----------
    Enforced by __post_init__, so every consumer may rely on them:

    - `time`, `mag` and `mag_err` are float64 and 1-D with identical length.
      Extractors downcast to float32 for periodfind, but only after
      zero-offsetting; a float32 array arriving here would already have lost
      the sub-day timing that downcast is careful to preserve.
    - `time` is sorted ascending, with `mag` and `mag_err` permuted to match.
      Consecutive-difference statistics (von Neumann ratio, Stetson J and K)
      and the dm/dt histogram all read neighbouring elements and are silently
      wrong on unsorted input rather than failing.

    Non-finite values are *not* removed — that is a survey loader's decision,
    and dropping epochs here would hide it from the caller.
    """

    source_id : str
    time      : np.ndarray
    mag       : np.ndarray
    mag_err   : np.ndarray
    band      : Band
    survey    : Survey
    ra        : float
    dec       : float
    time_system : str = "hjd"

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
        if self.time_system not in _TIME_SYSTEMS:
            raise ValueError(
                f"Unknown time_system {self.time_system!r}. "
                f"Valid: {sorted(_TIME_SYSTEMS)}."
            )

        # Upcasting a float32 time array here would satisfy the float64
        # guarantee while carrying damage that is already done: at HJD
        # ~2.46e6 one float32 ULP is 0.25 days, so every separation under six
        # hours has already collapsed to zero and no later step can tell.
        # Reject it at the boundary instead, where the loader that produced
        # it is still identifiable.
        t_in = np.asarray(self.time)
        if t_in.dtype == np.float32:
            raise ValueError(
                f"time for source {self.source_id!r} arrived as float32. "
                "Absolute epochs need float64: one float32 ULP at HJD ~2.46e6 "
                "is 0.25 days, so sub-day sampling is already lost. Build the "
                "array in float64 in the loader."
            )

        self.time    = np.asarray(self.time,    dtype=np.float64)
        self.mag     = np.asarray(self.mag,     dtype=np.float64)
        self.mag_err = np.asarray(self.mag_err, dtype=np.float64)

        # Sort once here rather than trusting every loader and every
        # extractor to agree about it.
        if self.time.size > 1 and not np.all(np.diff(self.time) >= 0):
            order        = np.argsort(self.time, kind="stable")
            self.time    = self.time[order]
            self.mag     = self.mag[order]
            self.mag_err = self.mag_err[order]

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
    2. Period detection        — the adopted period plus its significance,
                                 the algorithm that found it, the full
                                 per-algorithm top-N candidates, and the
                                 cross-algorithm agreement counts
    3. Fourier decomposition   — 14 scalar features at the detected period
    4. dm/dt histogram         — (N_DM_BINS, N_DT_BINS) image; None if not computed
    5. Gaia cross-match        —  7 features; None if no counterpart found

    Group sizes other than the fixed 22 / 14 are deliberately not stated as
    counts here: they change whenever an algorithm or a catalogue column is
    added, and a stale count in a docstring is worse than no count.  The
    authoritative list is dataclasses.fields(FeatureVector), and the scalar
    subset a model consumes is ml4em.models.base.SCALAR_FIELDS.

    All float fields default to np.nan so that partial feature extraction is
    explicit rather than absent. The feature layer sets each field it computes;
    downstream code can check for nan to detect uncomputed features.
    """

    source_id : str
    survey    : Survey

    # Photometric band these features were computed from.  One FeatureVector is
    # emitted per (source, band): a source observed in g and r yields two rows
    # with the same source_id.  Independent per-band periods are themselves a
    # signal — a real variable repeats across bands, an artefact usually does not.
    band : str = ""

    # Sky position — copied from the primary LightCurve by the feature pipeline.
    # Required by the inference layer to populate Candidate.ra / .dec.
    ra  : float = np.nan   # right ascension, decimal degrees (J2000)
    dec : float = np.nan   # declination, decimal degrees (J2000)

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

    # Top-N candidate peaks per algorithm, ranked best-first.
    # Keys are algorithm names ('CE', 'AOV', ...); values are lists of length
    # PeriodConfig.top_n_periods.  Kept as dicts here because the number of
    # algorithms is a config choice; the training layer flattens them into
    # period_{i}_{ALGO} / significance_{i}_{ALGO} parquet columns.
    period_top       : dict[str, list[float]] = field(default_factory=dict, repr=False)
    significance_top : dict[str, list[float]] = field(default_factory=dict, repr=False)

    # Cross-algorithm agreement.  Two peaks "agree" when they match within
    # tolerance at a 1:1, 1:2, 2:1, 1:3 or 3:1 harmonic ratio.
    period_n_agree_pairs  : int   = 0        # agreeing (algorithm, algorithm) pairs
    period_n_total_pairs  : int   = 0        # pairs compared
    period_agree_score    : float = np.nan   # n_agree_pairs / n_total_pairs
    period_agree_strict   : float = np.nan   # same, counting only exact 1:1 matches
    period_agree_weighted : float = np.nan   # agreement weighted by peak rank
    period_best_agree     : float = np.nan   # period from the best agreeing pair, days
    period_best_consensus : float = np.nan   # period agreed on by the most algorithms, days

    # Sidereal alias families.  All top-N peaks from all algorithms are nodes;
    # peaks linked by a sidereal-day alias relation are merged into families.
    # The family spanning the most distinct algorithms wins — this survives the
    # case where every algorithm found the same star but landed on a different
    # alias of its period.
    period_family_n_algos    : int   = 0        # distinct algorithms in the winning family
    period_family_rank_score : float = np.nan   # rank-weighted strength of that family
    period_family_n_members  : int   = 0        # peaks in the winning family
    period_family_n_total    : int   = 0        # families found
    period_family_best       : float = np.nan   # representative period of the family, days
    period_family_algorithm  : str   = ""       # algorithm contributing that representative

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
    # Used to confirm WD nature: blue BP-RP + high parallax + clean astrometry.
    # Field names and choice of astrometric quality indicator match scope-ml's
    # external_xmatch.py projection (Gaia_EDR3 catalog on Kowalski).
    #
    # The raw G/BP/RP magnitudes are kept alongside the derived colour.  They
    # cost nothing extra (the cross-match already returns them) and a colour
    # alone is not recoverable back into them, so dropping them would decide
    # on the model's behalf that apparent brightness is irrelevant.  G with
    # parallax gives absolute magnitude, which is what places a source on the
    # HR diagram — the standard way to separate white dwarfs from main
    # sequence stars of the same colour.
    gaia_parallax                  : Optional[float] = None  # mas
    gaia_parallax_error            : Optional[float] = None  # mas
    gaia_g_mean_mag                : Optional[float] = None  # G  apparent mag
    gaia_bp_mean_mag               : Optional[float] = None  # BP apparent mag
    gaia_rp_mean_mag               : Optional[float] = None  # RP apparent mag
    gaia_bp_rp                     : Optional[float] = None  # BP − RP colour, mag
    gaia_astrometric_excess_noise  : Optional[float] = None  # astrometric residual noise; lower = cleaner single-source fit


# ---------------------------------------------------------------------------
# Interface 2b: Label preparation → Training layer
# ---------------------------------------------------------------------------

@dataclass
class LabeledSample:
    """A FeatureVector paired with a ground-truth label for training.

    Data contract between upstream label preparation (e.g. Gaia cross-match
    in wdb-ml) and the training layer.  The training layer never produces
    labels — it only consumes them.

    Fields
    ------
    feature : FeatureVector
        Fully extracted feature set for this source.
    label : int
        Ground-truth class index, assigned upstream.  Binary use is the
        common case (0 = negative, 1 = positive), but any non-negative
        index is valid so a multi-class head can consume the same store.
    """

    feature : FeatureVector
    label   : int


# ---------------------------------------------------------------------------
# Interface 3: Inference layer → Output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """Inference result for a single source.  Generic — not science-case specific.

    Produced by the inference post-processing step. Immutable once created.

    The researcher's science case (WDB, AGN, eclipsing binary, etc.) is
    encoded in the trained model and the labels used during training —
    not in this type.

    Fields
    ------
    source_id : str
        Survey source identifier.
    ra, dec : float
        Sky position in decimal degrees (J2000).
    survey : Survey
        Originating survey.
    probability : float
        Model output in [0, 1] — P(positive class) as defined by training labels.
    period : float
        Detected dominant period in days.  np.nan if not computed.
    period_algorithm : str
        Algorithm that found the period.  Empty string if not computed.
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
