"""
Light curve statistics extractor.

Computes 22 scalar variability features from a single-band light curve.
These are the classical features used by variability classifiers such as
scope-ml, SIMBAD classifiers, and the original Stetson (1996) papers.

Preprocessing
-------------
1. Select the band with the most observations (primary band).
2. Iterative sigma-clip: remove epochs more than 3 MAD-sigmas from the median.
   Uses MAD-scaled sigma (1.4826 * MAD) for robustness against outliers.

Feature generation
------------------
Weighted statistics use w = 1/σ² to down-weight noisy epochs.
All features default to np.nan if computation fails (e.g. < 2 points).

Requires: numpy, scipy
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml4em.types import LightCurve


class StatisticsExtractor:
    """Compute 22 scalar light curve statistics.

    Parameters
    ----------
    n_sigma_clip:
        Sigma threshold for iterative outlier rejection.
        Default 3.0 follows the scope-ml convention.
    max_clip_iterations:
        Maximum passes of sigma-clipping before stopping.
    """

    def __init__(
        self,
        n_sigma_clip: float = 3.0,
        max_clip_iterations: int = 5,
    ) -> None:
        self._n_sigma = n_sigma_clip
        self._max_iter = max_clip_iterations

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _select_primary(self, lcs: list[LightCurve]) -> LightCurve:
        """Return the band with the most observations."""
        return max(lcs, key=lambda lc: lc.n_obs)

    def _sigma_clip(
        self,
        mag: np.ndarray,
        mag_err: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Iterative sigma-clip using MAD-scaled sigma.

        Returns clipped (mag, mag_err) with outlier epochs removed.
        """
        mask = np.ones(len(mag), dtype=bool)
        for _ in range(self._max_iter):
            m = mag[mask]
            median = np.median(m)
            mad = np.median(np.abs(m - median))
            sigma = 1.4826 * mad  # MAD → equivalent std for a Gaussian
            if sigma == 0:
                break
            new_mask = np.abs(mag - median) <= self._n_sigma * sigma
            if np.sum(new_mask) == np.sum(mask):
                break
            mask = new_mask
        return mag[mask], mag_err[mask]

    def _preprocess(
        self, lcs: list[LightCurve]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select primary band and apply sigma-clipping.

        Returns (time, mag, mag_err) after preprocessing.
        """
        primary = self._select_primary(lcs)
        m_clip, e_clip = self._sigma_clip(primary.mag, primary.mag_err)
        # Re-align time to the surviving mask
        mask = np.isin(primary.mag, m_clip)  # positional match after clip
        # Safer: rebuild mask by tracking which indices survived
        mask = np.zeros(len(primary.mag), dtype=bool)
        clip_mag, clip_err = self._sigma_clip(primary.mag, primary.mag_err)
        # Rebuild index mask (iterative clip operates in-place on a boolean mask)
        mask2 = np.ones(len(primary.mag), dtype=bool)
        for _ in range(self._max_iter):
            m = primary.mag[mask2]
            median = np.median(m)
            mad = np.median(np.abs(m - median))
            sigma = 1.4826 * mad
            if sigma == 0:
                break
            new_mask2 = np.abs(primary.mag - median) <= self._n_sigma * sigma
            if np.sum(new_mask2) == np.sum(mask2):
                break
            mask2 = new_mask2
        return primary.time[mask2], primary.mag[mask2], primary.mag_err[mask2]

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute(
        self, t: np.ndarray, m: np.ndarray, e: np.ndarray
    ) -> dict[str, Any]:
        """Compute all 22 statistics from pre-cleaned arrays."""
        n = len(m)
        if n < 2:
            return {"n_obs": n}

        out: dict[str, Any] = {"n_obs": n}

        # Weights
        w = 1.0 / e**2
        w_sum = w.sum()

        # ── Weighted moments ────────────────────────────────────────────
        wmean = float(np.dot(w, m) / w_sum)
        wvar  = float(np.dot(w, (m - wmean)**2) / w_sum)
        wstd  = float(np.sqrt(wvar)) if wvar >= 0 else np.nan

        out["wmean"] = wmean
        out["wstd"]  = wstd

        # ── Median and MAD ──────────────────────────────────────────────
        median = float(np.median(m))
        mad    = float(np.median(np.abs(m - median)))

        out["median"]          = median
        out["median_abs_dev"]  = mad

        # ── Reduced chi-squared vs constant (wmean) model ───────────────
        chi2red = float(np.sum((m - wmean)**2 / e**2) / (n - 1))
        out["chi2red"] = chi2red

        # ── Ratio of median scatter to median error (ROMS) ──────────────
        # Robust variability: median |Δm| / median σ
        med_err = float(np.median(e))
        out["roms"] = float(np.median(np.abs(m - wmean)) / med_err) if med_err > 0 else np.nan

        # ── Normalized peak-to-peak amplitude ───────────────────────────
        # (max_mag − err_at_max − min_mag − err_at_min) / (max + min + combined errors)
        # Equivalent to eclipse depth normalized by mean brightness.
        i_max, i_min = int(np.argmax(m)), int(np.argmin(m))
        bright = m[i_min] + e[i_min]   # faintest "bright edge" in mag
        faint  = m[i_max] - e[i_max]   # brightest "faint edge" in mag
        denom  = bright + faint
        out["norm_peak_to_peak_amp"] = float((faint - bright) / denom) if denom != 0 else np.nan

        # ── Normalized excess variance ──────────────────────────────────
        # From Vaughan+2003: (std² − mean(err²)) / mean(mag)²
        var     = float(np.var(m, ddof=1))
        mean_e2 = float(np.mean(e**2))
        mean_m2 = float(np.mean(m))**2
        out["norm_excess_var"] = float((var - mean_e2) / mean_m2) if mean_m2 > 0 else np.nan

        # ── Percentile ranges ───────────────────────────────────────────
        out["iqr"] = float(np.percentile(m, 75) - np.percentile(m, 25))
        out["i60r"] = float(np.percentile(m, 80) - np.percentile(m, 20))
        out["i70r"] = float(np.percentile(m, 85) - np.percentile(m, 15))
        out["i80r"] = float(np.percentile(m, 90) - np.percentile(m, 10))
        out["i90r"] = float(np.percentile(m, 95) - np.percentile(m,  5))

        # ── Weighted skewness ───────────────────────────────────────────
        if wstd > 0:
            out["skew"] = float(np.dot(w, (m - wmean)**3) / (w_sum * wstd**3))
        else:
            out["skew"] = np.nan

        # ── Fisher kurtosis (excess, small-sample corrected) ────────────
        if n >= 4 and wstd > 0:
            # Fisher's definition: kurtosis − 3 (0 for a Gaussian)
            raw_kurt = float(np.dot(w, (m - wmean)**4) / (w_sum * wstd**4))
            # Small-sample correction (excess kurtosis)
            out["small_kurt"] = raw_kurt - 3.0
        else:
            out["small_kurt"] = np.nan

        # ── Inverse Von Neumann ratio ───────────────────────────────────
        # Von Neumann η = mean(Δmᵢ²) / var(m)
        # inv_von_neumann = 1/η — high value → autocorrelated (periodic/variable)
        if n >= 3:
            diffs = np.diff(m)
            var_m = float(np.var(m, ddof=1))
            eta   = float(np.mean(diffs**2) / var_m) if var_m > 0 else np.nan
            out["inv_von_neumann"] = float(1.0 / eta) if (eta and eta > 0) else np.nan
        else:
            out["inv_von_neumann"] = np.nan

        # ── Stetson indices ─────────────────────────────────────────────
        # δᵢ = √(n/(n-1)) * (mᵢ − w̄m) / σᵢ
        delta = np.sqrt(n / (n - 1)) * (m - wmean) / e

        # Stetson K — kurtosis of residual distribution
        out["stetson_k"] = float(
            np.mean(np.abs(delta)) / np.sqrt(np.mean(delta**2))
        ) if np.mean(delta**2) > 0 else np.nan

        # Stetson J — using consecutive observation pairs (single-band approx.)
        if n >= 2:
            products = delta[:-1] * delta[1:]
            out["stetson_j"] = float(
                np.sum(np.sign(products) * np.sqrt(np.abs(products))) / (n - 1)
            )
        else:
            out["stetson_j"] = np.nan

        # Stetson I — requires two simultaneous bands; filled by pipeline
        # if a second band is provided.  Default NaN here.
        out["stetson_i"] = np.nan

        # ── Normality tests ─────────────────────────────────────────────
        try:
            from scipy.stats import anderson as _ad
            ad_result = _ad(m, dist="norm")
            out["anderson_darling"] = float(ad_result.statistic)
        except Exception:
            out["anderson_darling"] = np.nan

        if n >= 3:
            try:
                from scipy.stats import shapiro as _sw
                sw_stat, _ = _sw(m)
                out["shapiro_wilk"] = float(sw_stat)
            except Exception:
                out["shapiro_wilk"] = np.nan
        else:
            out["shapiro_wilk"] = np.nan

        return out

    # ------------------------------------------------------------------
    # Stetson I (two-band)
    # ------------------------------------------------------------------

    def _stetson_i_two_band(
        self,
        lc1: LightCurve,
        lc2: LightCurve,
        same_night_threshold_days: float = 0.02,
    ) -> float:
        """Compute Stetson I using simultaneous two-band observations.

        Pairs observations from lc1 and lc2 that occurred within
        same_night_threshold_days of each other.

        Returns np.nan if no valid pairs are found.
        """
        n1, n2 = len(lc1.time), len(lc2.time)
        if n1 < 2 or n2 < 2:
            return np.nan

        w1 = 1.0 / lc1.mag_err**2
        w2 = 1.0 / lc2.mag_err**2
        wm1 = float(np.dot(w1, lc1.mag) / w1.sum())
        wm2 = float(np.dot(w2, lc2.mag) / w2.sum())

        delta1 = np.sqrt(n1 / (n1 - 1)) * (lc1.mag - wm1) / lc1.mag_err
        delta2 = np.sqrt(n2 / (n2 - 1)) * (lc2.mag - wm2) / lc2.mag_err

        products: list[float] = []
        for i, t1 in enumerate(lc1.time):
            diffs = np.abs(lc2.time - t1)
            j = int(np.argmin(diffs))
            if diffs[j] <= same_night_threshold_days:
                products.append(float(delta1[i] * delta2[j]))

        if not products:
            return np.nan

        p = np.array(products)
        n_pairs = len(p)
        return float(np.sum(np.sign(p) * np.sqrt(np.abs(p))) / np.sqrt(n_pairs))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, lcs: list[LightCurve]) -> dict[str, Any]:
        """Compute all 22 scalar statistics for one source.

        Parameters
        ----------
        lcs:
            All bands for the source.  The band with the most observations
            is used as the primary band for statistics.

        Returns
        -------
        dict[str, Any]
            FeatureVector field names → computed values.
            Empty dict if lcs is empty.
        """
        if not lcs:
            return {}

        try:
            t, m, e = self._preprocess(lcs)
            out = self._compute(t, m, e)

            # Attempt Stetson I if a second band is present
            if len(lcs) >= 2:
                primary = self._select_primary(lcs)
                others  = [lc for lc in lcs if lc.band != primary.band]
                if others:
                    secondary = max(others, key=lambda lc: lc.n_obs)
                    out["stetson_i"] = self._stetson_i_two_band(primary, secondary)

            return out

        except Exception:
            return {}
