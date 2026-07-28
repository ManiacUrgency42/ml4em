"""
dm/dt histogram extractor.

Computes the 2-D Δmagnitude / Δtime histogram used as an image input for
the convolutional branch of the classifier, delegating to periodfind.DmDt —
a Rust-backed batched implementation.

Background
----------
For a light curve with N observations, there are N*(N-1)/2 unique pairs.
For each pair (i, j) with tⱼ > tᵢ:
    Δt  = tⱼ − tᵢ   (captures intra-night to multi-year)
    Δmag = mⱼ − mᵢ   (captures dimming/brightening)

Both axes use the non-uniform bin edges defined in constants.py — see
DMDT_DT_EDGES / DMDT_DM_EDGES for why they are not derived from a
min/max plus a spacing law.

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
from ml4em.features.base import to_float32_time
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
        # periodfind.DmDt.calc requires float32 edges.
        self._dt_edges = np.asarray(config.dt_edges, dtype=np.float32)
        self._dm_edges = np.asarray(config.dm_edges, dtype=np.float32)

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
            (N_DM_BINS, N_DT_BINS).  Empty dict for a source that is too
            short.  If the kernel itself raises, every source in the call
            gets an empty dict — the call is batched and there is no way to
            attribute the failure to one entry.
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
            times.append(to_float32_time(primary.time))
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
