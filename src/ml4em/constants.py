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
# The bin edges below are the hand-tuned, non-uniform edges used by the
# reference WDB production run (scope-ml config.defaults.yaml → dmdt_ints).
# They are *not* derivable from a min/max plus a spacing law, and the exact
# values are part of the feature definition: a histogram computed on
# different edges is not comparable to one computed on these, even at the
# same 26×26 shape.  Two properties matter and are lost under uniform
# spacing:
#
#   Δmag  spans ±8 mag so deep eclipses land in a real bin instead of
#         saturating the outermost one, while the edges tighten to 0.05 mag
#         either side of zero to resolve low-amplitude variability.
#   Δt    starts at exactly 0 and places four edges inside the first
#         0.12 d, so same-night pairs — the regime short-period binaries
#         live in — are resolved rather than collapsed into one bin.
# ---------------------------------------------------------------------------

DMDT_DM_EDGES : tuple[float, ...] = (
    -8.0, -4.5, -3.0, -2.5, -2.0, -1.5, -1.25, -0.75, -0.5, -0.3,
    -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75,
    1.25, 1.5, 2.0, 2.5, 3.0, 4.5, 8.0,
)

DMDT_DT_EDGES : tuple[float, ...] = (
    0.0, 0.02759, 0.04, 0.08, 0.12, 0.3, 0.75, 1.0, 1.5, 2.5,
    3.5, 4.5, 5.5, 7.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0,
    120.0, 180.0, 240.0, 360.0, 500.0, 650.0, 2000.0,
)

N_DT_BINS : int = len(DMDT_DT_EDGES) - 1   # time-difference bins      → 26
N_DM_BINS : int = len(DMDT_DM_EDGES) - 1   # magnitude-difference bins → 26


def dmdt_edges() -> tuple[np.ndarray, np.ndarray]:
    """Return (dt_edges, dm_edges) as float32 arrays for periodfind.DmDt.

    dt_edges : shape (N_DT_BINS + 1,)
    dm_edges : shape (N_DM_BINS + 1,)

    periodfind.DmDt.calc requires float32 edges; returning them already
    cast keeps that requirement in one place.
    """
    return (
        np.asarray(DMDT_DT_EDGES, dtype=np.float32),
        np.asarray(DMDT_DM_EDGES, dtype=np.float32),
    )


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
# incompatible with searching down to min_period_days = 0.003472 d (5 min).
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
