"""
Period-finding and Fourier decomposition extractor.

Computes 3 period features + 14 Fourier coefficient features by delegating
to periodfind — a Rust/CUDA-backed batched implementation used by scope-ml's
production pipeline.

Pipeline per batch
------------------
1. Preprocess: select primary band per source, cast to float32.
   Period-finding step: zero times (subtract t_min) and normalise mags to
   [0, 1] to match scope-ml's _prepare_lightcurves convention.
   Fourier step: use original (un-normalised) times and mags.
2. Run each configured algorithm across all N sources in a single batched
   call with output='peaks' (memory-efficient — no full periodogram stored).
3. Agreement scoring (pure Python): find the period confirmed by the most
   algorithms within _AGREE_TOL, with harmonic-aware matching and a
   minimum-period guard against sub-cadence spurious peaks.
4. Batch Fourier decomposition: one FourierDecomposition call for all sources
   that have a valid period.
5. Unpack the 14 Fourier columns into FeatureVector field names using
   scope-ml's phase convention: phi = arctan2(A, B), relative phases
   normalised as (phi_k/k − phi_1) / (2π/k) % 1.

Algorithms (scope-ml production set)
-------------------------------------
CE   Conditional Entropy       — periodfind.ConditionalEntropy
AOV  Analysis of Variance      — periodfind.AOV
LS   Lomb-Scargle              — periodfind.LombScargle
MHF  Multi-Harmonic Fourier    — periodfind.MultiHarmonicFourier

FourierDecomposition returns 14 features per source (CPU-only, Rust):
    [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]

Requires: periodfind (hard dependency, built via Dockerfile)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml4em.config.schema import PeriodConfig
from ml4em.types import LightCurve

# Fractional period tolerance for cross-algorithm agreement scoring.
# 5% matches scope-ml's default tolerance.
_AGREE_TOL: float = 0.05

# Harmonic ratios checked during agreement scoring.
# A period pair (P_a, P_b) agrees if P_a ≈ h * P_b for any h in this list.
# Catches the common aliases: half-period, double-period, 1/3, 3x period.
_AGREE_HARMONICS: list[float] = [1.0, 0.5, 2.0, 1.0 / 3, 3.0]

# Minimum period (days) considered physically meaningful.
# Periods shorter than this are treated as spurious during agreement scoring.
_MIN_AGREE_PERIOD: float = 0.007   # ~10 minutes


def _period_match(p_a: float, p_b: float) -> bool:
    """True if p_a and p_b agree within _AGREE_TOL, allowing harmonics.

    Checks all ratios in _AGREE_HARMONICS: a pair agrees if
    |p_a / (p_b * h) - 1| < _AGREE_TOL for any harmonic h.
    """
    if np.isnan(p_a) or np.isnan(p_b) or p_a <= _MIN_AGREE_PERIOD or p_b <= _MIN_AGREE_PERIOD:
        return False
    for h in _AGREE_HARMONICS:
        if abs(p_a / (p_b * h) - 1.0) < _AGREE_TOL:
            return True
    return False


class PeriodExtractor:
    """Find the dominant period and compute Fourier coefficients.

    Algorithm objects and the period grid are built once in __init__ and
    reused across all extract() calls.

    Parameters
    ----------
    config:
        PeriodConfig from FeatureConfig.period.
    """

    def __init__(self, config: PeriodConfig) -> None:
        self._cfg = config
        if config.samples_per_peak is None:
            # Period-spaced grid, built once and reused across all batches.
            self._static_periods: np.ndarray | None = np.linspace(
                config.min_period_days,
                config.max_period_days,
                config.n_freq_grid,
                dtype=np.float32,
            )
        else:
            # Frequency-spaced grid, computed per-batch from actual baseline.
            self._static_periods = None
        self._period_dts = np.zeros(1, dtype=np.float32)  # no chirp
        self._algos = self._build_algos()

    def _build_freq_grid(self, times: list[np.ndarray]) -> np.ndarray:
        """Build a frequency-spaced period grid matching scope-ml's convention.

        fmin = 2 / baseline   — scope-ml convention: require at least 2 full
                                cycles in the data baseline.
        fmax = 1 / min_period_days
        df   = 1 / (samples_per_peak * baseline)
        """
        baseline = max(float(t.max() - t.min()) for t in times)
        if baseline <= 0:
            baseline = 1.0
        f_min = 2.0 / baseline                              # scope-ml: 2 cycles minimum
        f_max = 1.0 / self._cfg.min_period_days
        df = 1.0 / (self._cfg.samples_per_peak * baseline)  # type: ignore[operator]
        freqs = np.arange(f_min, f_max, df, dtype=np.float64)
        freqs = freqs[(freqs >= f_min) & (freqs <= f_max)]
        if len(freqs) == 0:
            freqs = np.array([f_min], dtype=np.float64)
        # Convert to periods in ascending order to match linspace(min, max) convention.
        return (1.0 / freqs[::-1]).astype(np.float32).copy()

    def _build_algos(self) -> dict[str, Any]:
        import periodfind

        factories: dict[str, Any] = {
            "CE" : periodfind.ConditionalEntropy(n_phase=20, n_mag=10),
            "AOV": periodfind.AOV(n_phase=20),
            "LS" : periodfind.LombScargle(),
            "MHF": periodfind.MultiHarmonicFourier(max_harmonics=3),
            "FPW": periodfind.FPW(n_bins=10),
            "BLS": periodfind.BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.5),
        }
        return {
            name: factories[name]
            for name in self._cfg.algorithms
            if name in factories
        }

    # ------------------------------------------------------------------
    # Agreement scoring
    # ------------------------------------------------------------------

    def _agree(
        self,
        all_peaks: dict[str, list[list[Any]]],
        src_idx: int,
    ) -> tuple[float, float, str]:
        """Return (best_period, significance, algo_name) for one source.

        Collects the top peak from each algorithm, finds the period confirmed
        by the most algorithms within _AGREE_TOL (harmonic-aware), falls back
        to highest significance if no pair agrees.  Periods below
        _MIN_AGREE_PERIOD are excluded as sub-cadence artifacts.
        """
        candidates: list[tuple[float, float, str]] = []
        for algo_name, peaks_per_source in all_peaks.items():
            src_peaks = peaks_per_source[src_idx]
            if not src_peaks:
                continue
            top = src_peaks[0]
            period = float(top.params[0])
            sig = float(top.significance)
            if period > _MIN_AGREE_PERIOD and not np.isnan(period):
                candidates.append((period, sig, algo_name))

        if not candidates:
            return np.nan, np.nan, ""

        if len(candidates) == 1:
            return candidates[0]

        best_period, best_sig, best_algo = max(candidates, key=lambda x: x[1])
        best_count = 1

        for i, (p_i, s_i, a_i) in enumerate(candidates):
            count = sum(
                1 for j, (p_j, _, _) in enumerate(candidates)
                if i != j and _period_match(p_i, p_j)
            )
            if count > best_count or (count == best_count and s_i > best_sig):
                best_count = count
                best_period, best_sig, best_algo = p_i, s_i, a_i

        return best_period, best_sig, best_algo

    # ------------------------------------------------------------------
    # Fourier unpacking
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_fourier(row: np.ndarray) -> dict[str, Any]:
        """Map a 14-element FourierDecomposition row to FeatureVector fields.

        Column order: [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3,
                       A4, B4, A5, B5]

        Phase convention matches scope-ml (_ab_to_amp_phi in periodsearch.py):
            phi    = arctan2(A, B)                          (not arctan2(B, A))
            relphi = (phi_k / k − phi_1) / (2π/k) % 1     (normalised to [0, 1])
        """
        out: dict[str, Any] = {
            "f1_power": float(row[0]),
            "f1_bic"  : float(row[1]),
            # offset (col 2) and slope (col 3) have no FeatureVector fields
            "f1_a"    : float(row[4]),
            "f1_b"    : float(row[5]),
        }

        a1, b1 = float(row[4]), float(row[5])
        amp1 = float(np.sqrt(a1**2 + b1**2))
        phi1 = float(np.arctan2(a1, b1))        # scope-ml convention: arctan2(A, B)
        out["f1_amp"]  = amp1
        out["f1_phi0"] = phi1

        for k in range(1, 5):          # harmonics 2–5 → relative indices 1–4
            n = k + 1                  # harmonic number (2, 3, 4, 5)
            a_k = float(row[4 + 2 * k])
            b_k = float(row[5 + 2 * k])
            if a_k == 0.0 and b_k == 0.0:
                out[f"f1_relamp{k}"] = np.nan
                out[f"f1_relphi{k}"] = np.nan
            else:
                amp_k = float(np.sqrt(a_k**2 + b_k**2))
                phi_k = float(np.arctan2(a_k, b_k))   # scope-ml convention: arctan2(A, B)
                out[f"f1_relamp{k}"] = amp_k / amp1 if amp1 > 0 else np.nan
                # scope-ml normalization: (phi_k / k − phi_1) / (2π/k) % 1
                out[f"f1_relphi{k}"] = float(
                    (phi_k / n - phi1) / (2.0 * np.pi / n) % 1
                )

        return out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self, sources: list[list[LightCurve]]
    ) -> list[dict[str, Any]]:
        """Find the best period and compute Fourier features for a batch.

        Parameters
        ----------
        sources:
            Each element is all bands for one source.  The primary band
            (most observations) is used.

        Returns
        -------
        list[dict[str, Any]]
            One dict per source with period, period_significance,
            period_algorithm, and 14 Fourier fields.
            Period and Fourier fields are np.nan for sources where no period
            could be found.
        """
        if not sources:
            return []

        import periodfind

        nan_result: dict[str, Any] = {
            "period"              : np.nan,
            "period_significance" : np.nan,
            "period_algorithm"    : "",
            "f1_power"  : np.nan, "f1_bic"    : np.nan,
            "f1_a"      : np.nan, "f1_b"      : np.nan,
            "f1_amp"    : np.nan, "f1_phi0"   : np.nan,
            "f1_relamp1": np.nan, "f1_relphi1": np.nan,
            "f1_relamp2": np.nan, "f1_relphi2": np.nan,
            "f1_relamp3": np.nan, "f1_relphi3": np.nan,
            "f1_relamp4": np.nan, "f1_relphi4": np.nan,
        }

        # ── 1. Preprocess ────────────────────────────────────────────────
        # times / mags / errs : original arrays, used for Fourier decomposition.
        # times_pf / mags_pf  : zeroed times + [0,1]-normalised mags, used for
        #                        period finding (matches scope-ml's
        #                        _prepare_lightcurves convention).
        times, mags, errs, valid_idx = [], [], [], []
        times_pf, mags_pf = [], []
        for i, lcs in enumerate(sources):
            if not lcs:
                continue
            primary = max(lcs, key=lambda lc: lc.n_obs)
            if primary.n_obs < 4:
                continue
            t = primary.time.astype(np.float32)
            m = primary.mag.astype(np.float32)
            e = primary.mag_err.astype(np.float32)
            times.append(t)
            mags.append(m)
            errs.append(e)
            # Zero times and normalise mags to [0, 1] for period finding.
            t_zero = t - t.min()
            m_range = m.max() - m.min()
            m_norm = (m - m.min()) / m_range if m_range > 0 else np.zeros_like(m)
            times_pf.append(t_zero)
            mags_pf.append(m_norm)
            valid_idx.append(i)

        results: list[dict[str, Any]] = [dict(nan_result) for _ in sources]

        if not valid_idx or not self._algos:
            return results

        # ── 2. Run period-finding algorithms ─────────────────────────────
        periods = (
            self._static_periods
            if self._static_periods is not None
            else self._build_freq_grid(times_pf)
        )

        all_peaks: dict[str, list[list[Any]]] = {}
        n_peaks = self._cfg.top_n_periods

        for algo_name, algo in self._algos.items():
            try:
                needs_errs = algo_name in ("MHF", "FPW", "BLS")
                kwargs: dict[str, Any] = {
                    "output"  : "peaks",
                    "n_peaks" : n_peaks,
                }
                if needs_errs:
                    kwargs["errs"] = errs
                peaks = algo.calc(
                    times_pf, mags_pf, periods, self._period_dts, **kwargs
                )
                all_peaks[algo_name] = peaks
            except Exception:
                pass

        if not all_peaks:
            return results

        # ── 3. Agreement scoring (per source) ────────────────────────────
        best_periods = np.full(len(valid_idx), np.nan, dtype=np.float64)
        best_sigs    = np.full(len(valid_idx), np.nan, dtype=np.float64)
        best_algos   = [""] * len(valid_idx)

        for batch_pos in range(len(valid_idx)):
            p, s, a = self._agree(all_peaks, batch_pos)
            best_periods[batch_pos] = p
            best_sigs[batch_pos]    = s
            best_algos[batch_pos]   = a

        # ── 4. Batch Fourier decomposition ───────────────────────────────
        # Uses original (un-normalised) times and mags to match scope-ml's
        # compute_fourier_features, which does not apply [0,1] normalisation.
        fourier_mask = ~np.isnan(best_periods)
        fourier_raw: np.ndarray | None = None

        if fourier_mask.any():
            f_times  = [times[i]  for i in range(len(valid_idx)) if fourier_mask[i]]
            f_mags   = [mags[i]   for i in range(len(valid_idx)) if fourier_mask[i]]
            f_errs   = [errs[i]   for i in range(len(valid_idx)) if fourier_mask[i]]
            f_periods = best_periods[fourier_mask].astype(np.float32)

            try:
                fourier_raw = periodfind.FourierDecomposition().calc(
                    f_times, f_mags, f_errs, f_periods
                )  # (M_valid, 14)
            except Exception:
                fourier_raw = None

        # ── 5. Assemble results ──────────────────────────────────────────
        fourier_cursor = 0
        for batch_pos, src_idx in enumerate(valid_idx):
            period = best_periods[batch_pos]
            out = dict(nan_result)
            out["period"]               = float(period)
            out["period_significance"]  = float(best_sigs[batch_pos])
            out["period_algorithm"]     = best_algos[batch_pos]

            if not np.isnan(period) and fourier_raw is not None:
                out.update(self._unpack_fourier(fourier_raw[fourier_cursor]))
                fourier_cursor += 1

            results[src_idx] = out

        return results
