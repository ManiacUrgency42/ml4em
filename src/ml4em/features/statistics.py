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
            Empty dict for any source that fails.
        """
        if not sources:
            return []

        import periodfind

        times, mags, errs, valid_idx = [], [], [], []

        for i, lcs in enumerate(sources):
            if not lcs:
                continue
            primary = max(lcs, key=lambda lc: lc.n_obs)
            if primary.n_obs < 2:
                continue
            times.append(primary.time.astype(np.float32))
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
            out: dict[str, Any] = {}
            for col_idx, pf_name in enumerate(_PF_STAT_NAMES):
                fv_name = _STAT_NAME_MAP.get(pf_name)
                if fv_name is None:
                    continue
                val = float(row[col_idx])
                out[fv_name] = int(val) if fv_name == "n_obs" else val
            results[src_idx] = out

        return results
