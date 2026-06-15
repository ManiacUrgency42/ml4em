"""
dm/dt histogram extractor.

Computes the 2-D Δmagnitude / Δtime histogram used as an image input for
the convolutional branch of the classifier.

Background
----------
For a light curve with N observations, there are N*(N-1)/2 unique pairs.
For each pair (i, j) with tⱼ > tᵢ:
    Δt  = tⱼ − tᵢ   (log-spaced axis, captures intra-night to multi-year)
    Δmag = mⱼ − mᵢ   (linear axis, captures dimming/brightening)

The resulting histogram reveals characteristic patterns:
- A WDB eclipse appears as a narrow cluster at the orbital period.
- RR Lyrae show an asymmetric sawtooth arc from the rapid rise.
- Noise sources produce broad, diffuse distributions.

Output shape: (N_DM_BINS, N_DT_BINS) = (26, 26)
Matches the shape used by scope-ml's CNN branch.

Post-processing
---------------
L2-normalise the histogram so that ||H||₂ = 1.
This makes the histogram scale-invariant across different baseline lengths.

Preprocessing
-------------
Select the primary band (most observations).
No outlier rejection — all observations contribute to the dm/dt image.

Requires: numpy, fast-histogram (optional; falls back to numpy.histogram2d)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml4em.config.schema import DmdtConfig
from ml4em.constants import dmdt_edges
from ml4em.types import LightCurve


class DmdtExtractor:
    """Compute the Δmag / Δt pairwise histogram.

    Parameters
    ----------
    config:
        DmdtConfig from WDBConfig.features.dmdt.
        Controls bin counts and axis ranges.
    normalize:
        If True (default), L2-normalise the histogram.
    """

    def __init__(self, config: DmdtConfig, *, normalize: bool = True) -> None:
        self._cfg = config
        self._normalize = normalize
        # Pre-compute bin edges once; reused for every extract() call.
        self._dt_edges, self._dm_edges = self._build_edges()

    # ------------------------------------------------------------------
    # Bin edges
    # ------------------------------------------------------------------

    def _build_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Build histogram edges from config (or defaults from constants)."""
        dt_edges = np.logspace(
            np.log10(self._cfg.dt_min),
            np.log10(self._cfg.dt_max),
            self._cfg.n_dt_bins + 1,
        )
        dm_edges = np.linspace(
            self._cfg.dm_min,
            self._cfg.dm_max,
            self._cfg.n_dm_bins + 1,
        )
        return dt_edges, dm_edges

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, lcs: list[LightCurve]) -> tuple[np.ndarray, np.ndarray]:
        """Return (time, mag) for the band with the most observations."""
        primary = max(lcs, key=lambda lc: lc.n_obs)
        return primary.time, primary.mag

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_pairs(
        self, t: np.ndarray, m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute all N*(N-1)/2 pairwise (Δt, Δmag) combinations.

        Only forward-in-time pairs are kept (tⱼ > tᵢ), so Δt > 0 always.

        Returns
        -------
        dt : ndarray, shape (N_pairs,)
            Time differences in days.
        dm : ndarray, shape (N_pairs,)
            Magnitude differences (signed: positive = dimming).
        """
        n = len(t)
        # Upper-triangular indices (j > i guarantees tⱼ > tᵢ since t is sorted)
        i_idx, j_idx = np.triu_indices(n, k=1)
        dt = t[j_idx] - t[i_idx]   # always positive
        dm = m[j_idx] - m[i_idx]   # signed
        return dt, dm

    def _histogram(
        self, dt: np.ndarray, dm: np.ndarray
    ) -> np.ndarray:
        """Fill the 2-D histogram (dm × dt) and optionally L2-normalise.

        Returns array of shape (N_DM_BINS, N_DT_BINS).
        """
        try:
            # fast-histogram is ~5× faster than numpy for this use case
            from fast_histogram import histogram2d as fh2d
            h = fh2d(
                dm, dt,
                range=[[self._cfg.dm_min, self._cfg.dm_max],
                        [self._cfg.dt_min, self._cfg.dt_max]],
                bins=[self._cfg.n_dm_bins, self._cfg.n_dt_bins],
            ).astype(np.float32)
        except ImportError:
            # Fall back to numpy; log-spaced dt bins require explicit edges
            h, _, _ = np.histogram2d(
                dm, dt,
                bins=[self._dm_edges, self._dt_edges],
            )
            h = h.astype(np.float32)

        if self._normalize:
            norm = float(np.linalg.norm(h))
            if norm > 0:
                h /= norm

        return h  # shape: (N_DM_BINS, N_DT_BINS)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, lcs: list[LightCurve]) -> dict[str, Any]:
        """Compute the dm/dt histogram for one source.

        Parameters
        ----------
        lcs:
            All bands for the source.  Primary band is used.

        Returns
        -------
        dict[str, Any]
            ``{"dmdt": ndarray of shape (N_DM_BINS, N_DT_BINS)}``
            Empty dict if fewer than 2 clean observations remain.
        """
        if not lcs:
            return {}

        try:
            t, m = self._preprocess(lcs)
            if len(t) < 2:
                return {}

            dt, dm = self._compute_pairs(t, m)

            # Clip to configured range before histogramming
            # (pairs outside the range would be silently dropped by histogram2d)
            in_range = (
                (dt >= self._cfg.dt_min) & (dt <= self._cfg.dt_max) &
                (dm >= self._cfg.dm_min) & (dm <= self._cfg.dm_max)
            )
            dt, dm = dt[in_range], dm[in_range]

            if len(dt) == 0:
                return {}

            h = self._histogram(dt, dm)
            return {"dmdt": h}

        except Exception:
            return {}
