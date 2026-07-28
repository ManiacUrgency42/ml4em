"""
Light curve statistics extractor.

Computes 22 scalar variability features from a single-band light curve
by delegating to periodfind.BasicStats — a Rust-backed batched implementation
that matches the feature set used by scope-ml's production pipeline.

Preprocessing
-------------
Select the band with the most observations (primary band) per source.
No sigma-clipping — consistent with scope-ml's periodfind-based pipeline.

Feature generation
------------------
One batched call to periodfind.BasicStats().calc(times, mags, errs) processes
all N sources at once and returns an (N, 22) array.  Column order is defined
by BasicStats.STAT_NAMES; names are remapped to FeatureVector field names.

Requires: periodfind (hard dependency, built via Dockerfile)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml4em.features.base import to_float32_time
from ml4em.types import LightCurve

# Column order returned by periodfind.BasicStats().calc() — matches
# BasicStats.STAT_NAMES in periodfind/periodfind/cpu/__init__.py.
# Listed here as a constant to avoid an import-time dependency on periodfind.
_PF_STAT_NAMES: list[str] = [
    "N", "median", "wmean", "chi2red", "RoMS", "wstd",
    "NormPeaktoPeakamp", "NormExcessVar", "medianAbsDev",
    "iqr", "i60r", "i70r", "i80r", "i90r",
    "skew", "smallkurt", "invNeumann",
    "WelchI", "StetsonJ", "StetsonK", "AD", "SW",
]

# Mapping from periodfind BasicStats column names to FeatureVector field names.
# Identical names are still listed explicitly for clarity.
_STAT_NAME_MAP: dict[str, str] = {
    "N"                 : "n_obs",
    "median"            : "median",
    "wmean"             : "wmean",
    "chi2red"           : "chi2red",
    "RoMS"              : "roms",
    "wstd"              : "wstd",
    "NormPeaktoPeakamp" : "norm_peak_to_peak_amp",
    "NormExcessVar"     : "norm_excess_var",
    "medianAbsDev"      : "median_abs_dev",
    "iqr"               : "iqr",
    "i60r"              : "i60r",
    "i70r"              : "i70r",
    "i80r"              : "i80r",
    "i90r"              : "i90r",
    "skew"              : "skew",
    "smallkurt"         : "small_kurt",
    "invNeumann"        : "inv_von_neumann",
    "WelchI"            : "stetson_i",
    "StetsonJ"          : "stetson_j",
    "StetsonK"          : "stetson_k",
    "AD"                : "anderson_darling",
    "SW"                : "shapiro_wilk",
}

# Column holding "N".  Used to spot an all-NaN row before int() sees it.
_N_OBS_COL: int = _PF_STAT_NAMES.index("N")

# periodfind's Rust kernel returns an all-NaN row for fewer than four points
# (rust/src/basicstats.rs).  Matching the guard here keeps the NaN out of the
# int-typed n_obs field rather than discovering it downstream.
_MIN_OBS_FOR_STATS: int = 4


class StatisticsExtractor:
    """Compute 22 scalar light curve statistics via periodfind.BasicStats."""

    def extract(
        self, sources: list[list[LightCurve]]
    ) -> list[dict[str, Any]]:
        """Compute statistics for a batch of sources.

        Parameters
        ----------
        sources:
            Each element is all bands for one source.  The band with the most
            observations is used.

        Returns
        -------
        list[dict[str, Any]]
            One dict per source with FeatureVector field names as keys.
            Empty dict for a source that is too short, or whose statistics
            come back non-finite.  If the kernel itself raises, every source
            in the call gets an empty dict — the call is batched and there is
            no way to attribute the failure to one entry.
        """
        if not sources:
            return []

        import periodfind

        times, mags, errs, valid_idx = [], [], [], []

        for i, lcs in enumerate(sources):
            if not lcs:
                continue
            primary = max(lcs, key=lambda lc: lc.n_obs)
            # periodfind's BasicStats returns an all-NaN row below four
            # points (rust/src/basicstats.rs).  Admitting a shorter curve
            # would put a NaN into n_obs, which is an int field.
            if primary.n_obs < _MIN_OBS_FOR_STATS:
                continue
            times.append(to_float32_time(primary.time))
            mags.append(primary.mag.astype(np.float32))
            errs.append(primary.mag_err.astype(np.float32))
            valid_idx.append(i)

        results: list[dict[str, Any]] = [{} for _ in sources]

        if not valid_idx:
            return results

        try:
            raw = periodfind.BasicStats().calc(times, mags, errs)  # (M, 22)
        except Exception:
            return results

        for batch_pos, src_idx in enumerate(valid_idx):
            row = raw[batch_pos]
            # A row can still come back all-NaN (non-finite input, for
            # instance).  n_obs is an int field, so int(nan) would raise and
            # take the whole chunk's statistics down with it; an empty dict
            # is the documented signal that this one source has no stats.
            if not np.isfinite(row[_N_OBS_COL]):
                continue
            out: dict[str, Any] = {}
            for col_idx, pf_name in enumerate(_PF_STAT_NAMES):
                fv_name = _STAT_NAME_MAP.get(pf_name)
                if fv_name is None:
                    continue
                val = float(row[col_idx])
                out[fv_name] = int(val) if fv_name == "n_obs" else val
            results[src_idx] = out

        return results
