"""
Period-finding and Fourier decomposition extractor.

Computes 3 period features + 14 Fourier coefficient features by delegating
to periodfind — a Rust/CUDA-backed batched implementation used by scope-ml's
production pipeline.

Pipeline per batch
------------------
1. Preprocess: select primary band per source, cast to float32.
2. Run each configured algorithm across all N sources in a single batched
   call with output='peaks' (memory-efficient — no full periodogram stored).
3. Agreement scoring (pure Python): find the period confirmed by the most
   algorithms within a 2% fractional tolerance per source.
4. Batch Fourier decomposition: one FourierDecomposition call for all sources
   that have a valid period.
5. Unpack the 14 Fourier columns into FeatureVector field names.

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
_AGREE_TOL = 0.02


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
        """Build a frequency-spaced period grid from the batch time baseline.

        Matches scope-ml's grid construction: evenly spaced in frequency with
        step df = 1 / (samples_per_peak * baseline), where baseline is the
        longest time span across all sources in the batch.
        """
        baseline = max(float(t.max() - t.min()) for t in times)
        if baseline <= 0:
            baseline = 1.0
        f_min = 1.0 / self._cfg.max_period_days
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
            "MHF": periodfind.MultiHarmonicFourier(max_harmonics=5),
            "FPW": periodfind.FPW(n_bins=10),
            "BLS": periodfind.BoxLeastSquares(n_bins=50),
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
        by the most algorithms within _AGREE_TOL, falls back to highest
        significance if no pair agrees.
        """
        candidates: list[tuple[float, float, str]] = []
        for algo_name, peaks_per_source in all_peaks.items():
            src_peaks = peaks_per_source[src_idx]
            if not src_peaks:
                continue
            top = src_peaks[0]
            period = float(top.params[0])
            sig = float(top.significance)
            if period > 0 and not np.isnan(period):
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
                if i != j and abs(p_i - p_j) / min(p_i, p_j) < _AGREE_TOL
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
        phi1 = float(np.arctan2(b1, a1))
        out["f1_amp"]  = amp1
        out["f1_phi0"] = phi1

        for k in range(1, 5):          # harmonics 2–5 → relative indices 1–4
            a_k = float(row[4 + 2 * k])
            b_k = float(row[5 + 2 * k])
            if a_k == 0.0 and b_k == 0.0:
                out[f"f1_relamp{k}"] = np.nan
                out[f"f1_relphi{k}"] = np.nan
            else:
                amp_k = float(np.sqrt(a_k**2 + b_k**2))
                phi_k = float(np.arctan2(b_k, a_k))
                out[f"f1_relamp{k}"] = amp_k / amp1 if amp1 > 0 else np.nan
                out[f"f1_relphi{k}"] = phi_k - phi1

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
        times, mags, errs, valid_idx = [], [], [], []
        for i, lcs in enumerate(sources):
            if not lcs:
                continue
            primary = max(lcs, key=lambda lc: lc.n_obs)
            if primary.n_obs < 4:
                continue
            times.append(primary.time.astype(np.float32))
            mags.append(primary.mag.astype(np.float32))
            errs.append(primary.mag_err.astype(np.float32))
            valid_idx.append(i)

        results: list[dict[str, Any]] = [dict(nan_result) for _ in sources]

        if not valid_idx or not self._algos:
            return results

        # ── 2. Run period-finding algorithms ─────────────────────────────
        periods = (
            self._static_periods
            if self._static_periods is not None
            else self._build_freq_grid(times)
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
                    times, mags, periods, self._period_dts, **kwargs
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
