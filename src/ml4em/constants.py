"""
Survey and pipeline constants for ml4em.

Every fixed value used anywhere in the codebase is defined here with its
unit and source.  Magic numbers must not appear in any other module.

Sections
--------
dm/dt histogram parameters — fixed binning (changing invalidates saved features)
ZTF survey parameters
Cross-match parameters
"""

import numpy as np

# ---------------------------------------------------------------------------
# dm/dt histogram parameters
#
# The dm/dt histogram is a 2-D representation of all pairwise (Δt, Δmag)
# values in a light curve.  It serves as an image input to the
# convolutional branch of the neural network.
#
# Convention (matches scope-ml, required for model compatibility):
#   axis 0 → Δmag dimension  (N_DM_BINS rows)
#   axis 1 → Δt   dimension  (N_DT_BINS columns)
#   final ndarray shape: (N_DM_BINS, N_DT_BINS)
#
# The histogram2d output (shape N_DT_BINS × N_DM_BINS) is transposed
# before storage to give (N_DM_BINS, N_DT_BINS).
# ---------------------------------------------------------------------------

N_DT_BINS : int = 26   # time-difference bins
N_DM_BINS : int = 26   # magnitude-difference bins

# Scalar range parameters (bin edges are derived from these in dmdt_edges()).
DMDT_DT_MIN : float =  1e-3   # days  (~1.4 min)
DMDT_DT_MAX : float =  1e3    # days  (~2.7 yr)
DMDT_DM_MIN : float = -3.0    # mag   (source brightened by 3 mag)
DMDT_DM_MAX : float =  3.0    # mag   (source faded by 3 mag)


def dmdt_edges() -> tuple[np.ndarray, np.ndarray]:
    """Return (dt_edges, dm_edges) bin-edge arrays for the dm/dt histogram.

    dt_edges : shape (N_DT_BINS + 1,)  — log-spaced over DMDT_DT_MIN..DMDT_DT_MAX
    dm_edges : shape (N_DM_BINS + 1,)  — linearly spaced over DMDT_DM_MIN..DMDT_DM_MAX

    Time differences are log-spaced because relevant timescales span six
    orders of magnitude (minutes to years).  Magnitude differences are
    linear because eclipses and variability are typically bounded within
    a few magnitudes and we want uniform resolution there.
    """
    dt_edges = np.logspace(
        np.log10(DMDT_DT_MIN), np.log10(DMDT_DT_MAX), N_DT_BINS + 1
    )
    dm_edges = np.linspace(DMDT_DM_MIN, DMDT_DM_MAX, N_DM_BINS + 1)
    return dt_edges, dm_edges


# ---------------------------------------------------------------------------
# ZTF survey parameters
# ---------------------------------------------------------------------------

ZTF_BANDS : tuple[str, ...] = ("g", "r", "i")

# Sidereal day in solar days.
# Periods near integer multiples of this value are cadence aliases —
# the cadence_alias module uses this to flag and reject them.
ZTF_SIDEREAL_DAY : float = 0.997_269_57   # days

# High-cadence filter threshold.
# ZTF sometimes takes back-to-back exposures separated by ~10–20 min.
# Observations closer together than this are dropped before feature
# extraction to avoid biasing period-finding toward very short periods.
#
# 5 minutes matches the value used in the reference WDB production run
# (--min-cadence-minutes 5.0).  scope-ml's *library* default is 30 minutes,
# but 30 min would discard every pair of epochs closer together than half
# an hour — which is exactly the regime short-period WDBs live in, and is
# incompatible with searching down to min_period_days = 0.01 d (14.4 min).
ZTF_MIN_CADENCE_DAYS : float = 5.0 / 1440.0   # 5 minutes in days

# Maximum HJD for ZTF DR16.  Set when restricting to a specific data release.
ZTF_DR16_MAX_HJD : float = 2_459_951.5

# Sidereal frequency in cycles per solar day (1 / 0.99727 sidereal days).
# ZTF observes from the ground, so the observing window repeats on the
# sidereal day.  A true frequency f and any alias f ± n·f_sidereal are
# indistinguishable in the periodogram.  Used to group aliases into
# families during consensus scoring.
SIDEREAL_FREQ_PER_DAY : float = 24.0 / 23.9345   # ≈ 1.00274 cycles/day



# ---------------------------------------------------------------------------
# External catalog cross-match
# ---------------------------------------------------------------------------

# Cone-search radius used when matching against Gaia EDR3.
# ZTF astrometric precision is ~0.1–0.5 arcsec; 2 arcsec provides margin
# for proper-motion offsets while keeping false-match rates low.
XMATCH_RADIUS_ARCSEC : float = 2.0

# Gaia RUWE threshold below which the astrometric solution is considered
# reliable (Lindegren et al. 2021).  RUWE > 1.4 often indicates an
# unresolved binary or a poorly-fit single-star solution.
GAIA_RUWE_CLEAN : float = 1.4
