"""
dm/dt histogram extractor.

Computes the 2-D Δmagnitude / Δtime histogram used as an image input for
the convolutional branch of the classifier, delegating to periodfind.DmDt —
a Rust-backed batched implementation.

Background
----------
For a light curve with N observations, there are N*(N-1)/2 unique pairs.
For each pair (i, j) with tⱼ > tᵢ:
    Δt  = tⱼ − tᵢ   (log-spaced axis, captures intra-night to multi-year)
    Δmag = mⱼ − mᵢ   (linear axis, captures dimming/brightening)

Output shape: (N_DM_BINS, N_DT_BINS) = (26, 26)
Matches the shape used by scope-ml's CNN branch.

Post-processing
---------------
L2-normalised by periodfind.DmDt internally.

Requires: periodfind (hard dependency, built via Dockerfile)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml4em.config.schema import DmdtConfig
from ml4em.constants import dmdt_edges
from ml4em.types import LightCurve


class DmdtExtractor:
    """Compute the Δmag / Δt pairwise histogram via periodfind.DmDt.

    Parameters
    ----------
    config:
        DmdtConfig from FeatureConfig.dmdt.
    """

    def __init__(self, config: DmdtConfig) -> None:
        self._cfg = config
        self._dt_edges, self._dm_edges = self._build_edges()

    def _build_edges(self) -> tuple[np.ndarray, np.ndarray]:
        dt_edges = np.logspace(
            np.log10(self._cfg.dt_min),
            np.log10(self._cfg.dt_max),
            self._cfg.n_dt_bins + 1,
        ).astype(np.float32)
        dm_edges = np.linspace(
            self._cfg.dm_min,
            self._cfg.dm_max,
            self._cfg.n_dm_bins + 1,
        ).astype(np.float32)
        return dt_edges, dm_edges

    def extract(
        self, sources: list[list[LightCurve]]
    ) -> list[dict[str, Any]]:
        """Compute dm/dt histograms for a batch of sources.

        Parameters
        ----------
        sources:
            Each element is all bands for one source.  The band with the most
            observations is used.

        Returns
        -------
        list[dict[str, Any]]
            One dict per source with key "dmdt" → ndarray of shape
            (N_DM_BINS, N_DT_BINS).  Empty dict for any source that fails.
        """
        if not sources:
            return []

        import periodfind

        times, mags, valid_idx = [], [], []

        for i, lcs in enumerate(sources):
            if not lcs:
                continue
            primary = max(lcs, key=lambda lc: lc.n_obs)
            if primary.n_obs < 2:
                continue
            times.append(primary.time.astype(np.float32))
            mags.append(primary.mag.astype(np.float32))
            valid_idx.append(i)

        results: list[dict[str, Any]] = [{} for _ in sources]

        if not valid_idx:
            return results

        try:
            raw = periodfind.DmDt().calc(
                times, mags, self._dt_edges, self._dm_edges
            )  # (M, n_dm_bins, n_dt_bins)
        except Exception:
            return results

        for batch_pos, src_idx in enumerate(valid_idx):
            results[src_idx] = {"dmdt": raw[batch_pos]}

        return results
