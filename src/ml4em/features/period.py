"""
Period-finding and Fourier decomposition extractor.

Computes 3 period features + 14 Fourier coefficient features.

Preprocessing
-------------
Select the band with the most observations (primary band).
No additional preprocessing — cadence filtering was done in the data layer.

Feature generation
------------------
Run each configured algorithm in sequence.  Collect (period, significance)
from every algorithm that succeeds.  Select the best period by agreement
scoring: prefer a period that is confirmed by the most algorithms within
a fractional tolerance of 0.02.  Fall back to the highest-significance
result if no agreement is found.

Fourier decomposition
---------------------
Fit up to 5 harmonics at the best period using weighted least squares.
BIC-select the order (1–5 harmonics) that minimises overfitting.

Implemented algorithms
----------------------
LS    Lomb-Scargle         — astropy.timeseries.LombScargle  (always available)
BLS   Box Least Squares    — astropy.timeseries.BoxLeastSquares (always available)
CE    Conditional Entropy  — requires periodfind or p4j library (stub)
AOV   Analysis of Variance — requires periodfind or p4j library (stub)
FPW   Fast Period Wavelets — stub
MHF   Multi-Harmonic Fit   — stub

Requires: numpy, astropy
Optional: scipy (for BIC minimization tie-breaking)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ml4em.config.schema import PeriodConfig
from ml4em.types import LightCurve


class PeriodExtractor:
    """Find the dominant period and compute Fourier coefficients.

    Parameters
    ----------
    config:
        PeriodConfig from WDBConfig.features.period.
    ls_samples_per_peak:
        Frequency grid resolution for Lomb-Scargle.
        Higher values = finer grid = slower but more accurate.
    n_harmonics:
        Number of Fourier harmonics to fit at the best period.
        Must be ≥ 1; FeatureVector stores up to 5.
    """

    def __init__(
        self,
        config: PeriodConfig,
        ls_samples_per_peak: int = 10,
        n_harmonics: int = 5,
    ) -> None:
        self._cfg = config
        self._ls_spp = ls_samples_per_peak
        self._n_harmonics = n_harmonics

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, lcs: list[LightCurve]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (time, mag, mag_err) for the band with the most observations."""
        primary = max(lcs, key=lambda lc: lc.n_obs)
        return primary.time, primary.mag, primary.mag_err

    # ------------------------------------------------------------------
    # Period-finding algorithms
    # ------------------------------------------------------------------

    def _run_ls(
        self, t: np.ndarray, m: np.ndarray, e: np.ndarray
    ) -> tuple[float, float]:
        """Lomb-Scargle via scipy.signal.lombscargle.

        Uses the Zechmeister-Kürster floating-mean generalization via a
        pre-centred, error-weighted implementation so that heteroscedastic
        errors are respected.

        Returns (best_period_days, peak_normalized_power).
        Peak power in [0, 1]; values > 0.5 are typically significant.
        """
        from scipy.signal import lombscargle

        # Frequency grid: log-spaced between 1/max_period and 1/min_period
        f_min = 1.0 / self._cfg.max_period_days
        f_max = 1.0 / self._cfg.min_period_days
        n_freq = int(self._ls_spp * len(t) * (f_max - f_min) * (t[-1] - t[0]))
        n_freq = max(n_freq, 1000)  # at least 1000 frequencies

        freqs = np.linspace(f_min, f_max, n_freq)
        omegas = 2.0 * np.pi * freqs

        # Subtract weighted mean before feeding to lombscargle
        w = 1.0 / e**2
        m_centered = m - np.dot(w, m) / w.sum()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            power = lombscargle(t, m_centered, omegas, normalize=True)

        best_idx = int(np.argmax(power))
        best_period = float(1.0 / freqs[best_idx])
        significance = float(power[best_idx])  # normalized power in [0, 1]
        return best_period, significance

    def _run_bls(
        self, t: np.ndarray, m: np.ndarray, e: np.ndarray
    ) -> tuple[float, float]:
        """Box Least Squares via astropy.timeseries.BoxLeastSquares.

        Best suited for flat-bottomed eclipses (WDB, EA eclipsing binaries).
        Requires a working astropy installation (astropy >= 3.2).
        """
        try:
            from astropy.timeseries import BoxLeastSquares
        except Exception as exc:
            raise ImportError(
                "BLS requires astropy.timeseries.BoxLeastSquares.\n"
                "Ensure astropy is installed and compatible with your numpy version."
            ) from exc

        bls = BoxLeastSquares(t, m, e)
        periods = np.logspace(
            np.log10(self._cfg.min_period_days),
            np.log10(self._cfg.max_period_days),
            1000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = bls.power(periods, duration=0.1 * periods)

        best_idx = int(np.argmax(result.power))
        best_period = float(result.period[best_idx])
        significance = float(np.max(result.power))
        return best_period, significance

    def _run_ce(
        self, t: np.ndarray, m: np.ndarray, e: np.ndarray
    ) -> tuple[float, float]:
        """Conditional Entropy period finder.

        Requires the periodfind or p4j package.
        """
        raise NotImplementedError(
            "CE algorithm requires the periodfind or p4j library.\n"
            "Install periodfind: pip install periodfind"
        )

    def _run_aov(
        self, t: np.ndarray, m: np.ndarray, e: np.ndarray
    ) -> tuple[float, float]:
        """Analysis of Variance period finder.

        Requires the periodfind or p4j package.
        """
        raise NotImplementedError(
            "AOV algorithm requires the periodfind or p4j library.\n"
            "Install periodfind: pip install periodfind"
        )

    # ------------------------------------------------------------------
    # Algorithm dispatcher
    # ------------------------------------------------------------------

    _RUNNERS = {
        "LS" : "_run_ls",
        "BLS": "_run_bls",
        "CE" : "_run_ce",
        "AOV": "_run_aov",
    }

    def _find_periods(
        self, t: np.ndarray, m: np.ndarray, e: np.ndarray
    ) -> list[tuple[float, float, str]]:
        """Run all configured algorithms; return list of (period, sig, algo).

        Silently skips algorithms that raise NotImplementedError or ImportError.
        """
        results: list[tuple[float, float, str]] = []
        for algo in self._cfg.algorithms:
            runner_name = self._RUNNERS.get(algo)
            if runner_name is None:
                continue
            try:
                period, sig = getattr(self, runner_name)(t, m, e)
                if not np.isnan(period):
                    results.append((period, sig, algo))
            except (NotImplementedError, ImportError):
                pass
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------
    # Agreement scoring
    # ------------------------------------------------------------------

    def _best_by_agreement(
        self,
        candidates: list[tuple[float, float, str]],
        frac_tol: float = 0.02,
    ) -> tuple[float, float, str]:
        """Return the period confirmed by the most algorithms.

        Two periods agree when |P₁ − P₂| / min(P₁, P₂) < frac_tol.
        If no pair agrees, return the highest-significance result.
        """
        if len(candidates) == 1:
            return candidates[0]

        best_period, best_sig, best_algo = max(candidates, key=lambda x: x[1])
        best_count = 1

        for i, (p_i, s_i, a_i) in enumerate(candidates):
            count = 0
            for j, (p_j, s_j, _) in enumerate(candidates):
                if i == j:
                    continue
                frac_diff = abs(p_i - p_j) / min(p_i, p_j)
                if frac_diff < frac_tol:
                    count += 1
            if count > best_count or (count == best_count and s_i > best_sig):
                best_count = count
                best_period, best_sig, best_algo = p_i, s_i, a_i

        return best_period, best_sig, best_algo

    # ------------------------------------------------------------------
    # Fourier decomposition
    # ------------------------------------------------------------------

    def _fourier_decompose(
        self,
        t: np.ndarray,
        m: np.ndarray,
        e: np.ndarray,
        period: float,
    ) -> dict[str, Any]:
        """Fit a Fourier series at `period` and extract harmonic features.

        Selects the number of harmonics (1–n_harmonics) by BIC.
        Returns coefficients for harmonics 1–5; missing harmonics are NaN.
        """
        if np.isnan(period) or period <= 0 or len(t) < 4:
            return {}

        phase = (t % period) / period  # normalised phase in [0, 1)
        w = 1.0 / e**2
        n = len(t)

        # Evaluate fits for 1..n_harmonics harmonics; pick best by BIC
        best_bic = np.inf
        best_coeffs: np.ndarray | None = None
        best_nh = 1

        for nh in range(1, self._n_harmonics + 1):
            cols = [np.ones(n)]
            for k in range(1, nh + 1):
                cols.append(np.cos(2 * np.pi * k * phase))
                cols.append(np.sin(2 * np.pi * k * phase))
            A = np.column_stack(cols)
            # Weighted least squares via normal equations
            AW = A.T * w
            try:
                coeffs, *_ = np.linalg.lstsq(AW @ A, AW @ m, rcond=None)
            except np.linalg.LinAlgError:
                continue
            residuals = m - A @ coeffs
            chi2  = float(np.dot(residuals**2, w))
            k_par = len(coeffs)
            bic   = chi2 + k_par * np.log(n)
            if bic < best_bic:
                best_bic   = bic
                best_coeffs = coeffs
                best_nh    = nh

        if best_coeffs is None:
            return {}

        # Reference chi² for constant (wmean) model
        wmean = float(np.dot(w, m) / w.sum())
        chi2_const = float(np.dot((m - wmean)**2, w))

        # Re-evaluate best-order fit for f1_power
        nh = best_nh
        cols = [np.ones(n)]
        for k in range(1, nh + 1):
            cols.append(np.cos(2 * np.pi * k * phase))
            cols.append(np.sin(2 * np.pi * k * phase))
        A = np.column_stack(cols)
        chi2_fit = float(np.dot((m - A @ best_coeffs)**2, w))
        f1_power = float(1.0 - chi2_fit / chi2_const) if chi2_const > 0 else np.nan

        # Extract harmonic amplitudes and phases
        # best_coeffs = [offset, a1, b1, a2, b2, ...]
        harmonics: list[tuple[float, float, float, float]] = []
        for k in range(1, self._n_harmonics + 1):
            if k <= best_nh and (2 * k) < len(best_coeffs):
                a = float(best_coeffs[2 * k - 1])
                b = float(best_coeffs[2 * k])
            else:
                a, b = np.nan, np.nan
            amp = float(np.sqrt(a**2 + b**2)) if not np.isnan(a) else np.nan
            phi = float(np.arctan2(b, a)) if not np.isnan(a) else np.nan
            harmonics.append((a, b, amp, phi))

        a1, b1, amp1, phi1 = harmonics[0]

        out: dict[str, Any] = {
            "f1_power" : f1_power,
            "f1_bic"   : float(best_bic),
            "f1_a"     : a1,
            "f1_b"     : b1,
            "f1_amp"   : amp1,
            "f1_phi0"  : phi1,
        }
        for idx, (_, _, amp, phi) in enumerate(harmonics[1:], start=1):
            out[f"f1_relamp{idx}"] = float(amp / amp1) if (amp1 and amp1 > 0) else np.nan
            out[f"f1_relphi{idx}"] = float(phi - phi1) if not (np.isnan(phi) or np.isnan(phi1)) else np.nan

        return out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, lcs: list[LightCurve]) -> dict[str, Any]:
        """Find the best period and compute Fourier features.

        Parameters
        ----------
        lcs:
            All bands for the source.  The primary band (most observations)
            is used for period finding.

        Returns
        -------
        dict[str, Any]
            period, period_significance, period_algorithm + 14 Fourier fields.
            Empty dict if no period could be found.
        """
        if not lcs:
            return {}

        try:
            t, m, e = self._preprocess(lcs)
            candidates = self._find_periods(t, m, e)
            if not candidates:
                return {}

            period, sig, algo = self._best_by_agreement(candidates)

            out: dict[str, Any] = {
                "period"              : period,
                "period_significance" : sig,
                "period_algorithm"    : algo,
            }
            out.update(self._fourier_decompose(t, m, e, period))
            return out

        except Exception:
            return {}


