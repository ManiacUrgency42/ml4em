"""
Period-finding, consensus scoring and Fourier decomposition.

Delegates the expensive periodogram evaluation to periodfind (Rust + CUDA)
and does the peak selection, cross-algorithm consensus and Fourier unpacking
here.  The algorithms and the consensus logic follow scope-ml's production
pipeline; the organisation differs.

Why a consensus step exists at all
----------------------------------
A periodogram's tallest peak is frequently not the true period.  Two
mechanisms conspire against it:

  Harmonics.  An eclipsing binary with two similar eclipses per orbit folds
  equally cleanly at P and at P/2, so the ranking between them is decided by
  noise.

  Sidereal aliases.  Ground-based sampling repeats on the sidereal day, so a
  true frequency f is indistinguishable from f ± n·f_sidereal.  The
  periodogram shows a comb of near-equal peaks.

Neither is fixed by a better algorithm — the information simply is not in a
single periodogram.  What does help is asking several algorithms with
different statistics and checking which *family* of aliases they collectively
favour.  That is what the agreement and sidereal-family scores below measure,
and those scores are themselves useful ML features.

Pipeline per batch
------------------
1. Preprocess.  Sort by time, subtract t_min in float64, then cast to
   float32.  The order matters — see the note in _prepare().
2. For each algorithm, evaluate the full periodogram and reduce it to the
   top-N distinct peaks with real z-score significances.
3. Fourier-decompose at each algorithm's top period to get f1_power, which
   the family scorer uses to pick between algorithms.
4. Score consensus: pairwise agreement and sidereal-alias families.
5. Fourier-decompose once more at the chosen consensus period and unpack
   the 14 coefficients into FeatureVector fields.

Algorithms
----------
CE   Conditional Entropy    minimised — a correctly folded curve is low-entropy
AOV  Analysis of Variance   maximised
LS   Lomb-Scargle           maximised
MHF  Multi-Harmonic Fourier maximised
FPW  Fast Periodic Wavelet  maximised
BLS  Box Least Squares      maximised

The minimise/maximise direction is read off each Periodogram object rather
than hardcoded, so adding an algorithm does not require touching this file.

FourierDecomposition returns 14 values per source (CPU-only, Rust):
    [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]

Requires: periodfind (hard dependency, built via Dockerfile)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ml4em.config.schema import PeriodConfig
from ml4em.constants import SIDEREAL_FREQ_PER_DAY
from ml4em.types import LightCurve

log = logging.getLogger(__name__)

# Fractional period tolerance for cross-algorithm agreement scoring.
# 5% matches scope-ml's default tolerance.
_AGREE_TOL: float = 0.05

# Harmonic ratios checked during agreement scoring.
# A period pair (P_a, P_b) agrees if P_a ≈ h * P_b for any h here.
# Catches the common aliases: half-period, double-period, 1/3, 3x.
_AGREE_HARMONICS: tuple[float, ...] = (1.0, 0.5, 2.0, 1.0 / 3, 3.0)

# Sidereal-family grouping uses a narrower harmonic set: the alias search
# already covers a wide frequency span, so admitting 1/3 and 3x as well
# merges families that are not physically related.
_FAMILY_HARMONICS: tuple[float, ...] = (1.0, 0.5, 2.0)

# Maximum sidereal alias order searched, and the fractional tolerance on the
# integer residual.  n up to 15 covers the aliases that actually appear in
# ZTF periodograms; beyond that the comb teeth are below the noise.
_SIDEREAL_MAX_N: int = 15
_SIDEREAL_TOL: float = 0.03

# Minimum period (days) considered physically meaningful.
# Shorter periods are treated as spurious during consensus scoring.
_MIN_AGREE_PERIOD: float = 0.007   # ~10 minutes

# The periodogram grid runs to ~10^6 points, so a full periodogram is several
# MB per source.  Requesting a 1000-source batch at once would return multiple
# GB to host memory.  Sources are therefore fed to periodfind in sub-batches
# sized to stay under this budget, and each sub-batch is reduced to its top-N
# peaks and discarded before the next one is requested.
_PERIODOGRAM_BUDGET_BYTES: int = 2 * 1024**3   # 2 GiB


# ---------------------------------------------------------------------------
# Period matching helpers
# ---------------------------------------------------------------------------

def _signed_z(
    values: np.ndarray,
    mean_val: float,
    std_val: float,
    use_minima: bool,
) -> np.ndarray:
    """Z-score oriented so that larger always means a stronger peak.

    Algorithms disagree about direction: conditional entropy is minimised while
    Lomb-Scargle, AOV, BLS and the multi-harmonic fit are maximised.  Scoring a
    minimised statistic with |value - mean| makes distance from the mean the
    criterion, which rates a strongly *bad* trial exactly as highly as a
    strongly good one.  Chunk winners are picked by argmin first, so this
    usually stays hidden — but a chunk lying entirely in a high-entropy region
    contributes a winner above the mean, and abs() then promotes it to the top
    of the peak list.

    Flipping the sign for minimised statistics also makes `significance`
    comparable between algorithms, which the consensus scorer relies on when it
    falls back to picking the highest-significance peak.
    """
    delta = (mean_val - values) if use_minima else (values - mean_val)
    return delta / std_val


def _period_match(
    p_a: float,
    p_b: float,
    tol: float = _AGREE_TOL,
    harmonics: tuple[float, ...] = _AGREE_HARMONICS,
) -> bool:
    """True if p_a and p_b agree within `tol`, allowing harmonic ratios.

    A pair agrees if |p_a / (p_b * h) - 1| < tol for any h in `harmonics`.
    """
    if np.isnan(p_a) or np.isnan(p_b):
        return False
    if p_a <= _MIN_AGREE_PERIOD or p_b <= _MIN_AGREE_PERIOD:
        return False
    for h in harmonics:
        if abs(p_a / (p_b * h) - 1.0) < tol:
            return True
    return False


def _are_sidereal_aliases(
    p_a: float,
    p_b: float,
    harmonics: tuple[float, ...] = _FAMILY_HARMONICS,
    max_n: int = _SIDEREAL_MAX_N,
    tol: float = _SIDEREAL_TOL,
) -> bool:
    """True if p_a and p_b differ by an integer number of sidereal aliases.

    Two frequencies separated by n · f_sidereal produce nearly identical
    folded light curves under ground-based sampling, so the periodogram
    cannot distinguish them.  Testing

        (f_a - h/p_b) / f_sidereal  ≈  integer

    identifies pairs that are the same physical signal seen at different
    teeth of the alias comb.
    """
    if np.isnan(p_a) or np.isnan(p_b) or p_a <= 0 or p_b <= 0:
        return False
    f_a = 1.0 / p_a
    for h in harmonics:
        n_float = (f_a - h / p_b) / SIDEREAL_FREQ_PER_DAY
        n_int = round(n_float)
        if abs(n_int) <= max_n and abs(n_float - n_int) < tol:
            return True
    return False


class PeriodExtractor:
    """Find candidate periods, score cross-algorithm consensus, fit Fourier terms.

    Parameters
    ----------
    config:
        PeriodConfig from FeatureConfig.period.

    Notes
    -----
    Algorithm objects are constructed lazily on first use, not in __init__.
    periodfind resolves the CPU/GPU backend at construction time, so building
    them eagerly would bind the backend before FeaturePipeline has had a
    chance to call set_device().
    """

    def __init__(self, config: PeriodConfig) -> None:
        self._cfg = config
        self._period_dts = np.zeros(1, dtype=np.float32)  # no chirp
        self._algos: dict[str, Any] | None = None
        self._grid: np.ndarray | None = None        # periods, ascending
        self._baseline: float | None = None

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def prepare(self, sources: list[list[LightCurve]]) -> None:
        """Build the frequency grid once from the full-field baseline.

        Called by FeaturePipeline before chunking.  The grid must not depend
        on which sources share a chunk: df is 1/(samples_per_peak · baseline),
        so a per-chunk baseline would give the same star a different grid —
        and therefore a different period — depending on its neighbours.  That
        breaks reproducibility and makes checkpoint/resume non-deterministic.
        """
        baseline = 0.0
        for lcs in sources:
            for lc in lcs:
                if lc.n_obs >= 2:
                    span = float(lc.time.max() - lc.time.min())
                    baseline = max(baseline, span)

        if baseline <= 0:
            log.warning("Could not determine a baseline from %d sources; "
                        "falling back to 1 day", len(sources))
            baseline = 1.0

        self._baseline = baseline
        self._grid = self._build_freq_grid(baseline)
        log.info("Period grid: baseline %.1f d, %d trial periods "
                 "(%.3f–%.3f d)", baseline, len(self._grid),
                 float(self._grid[0]), float(self._grid[-1]))

    def _build_freq_grid(self, baseline: float) -> np.ndarray:
        """Frequency-spaced period grid matching scope-ml's convention.

        f_min = max(2 / baseline, 1 / max_period_days)
        f_max = 1 / min_period_days
        df    = 1 / (samples_per_peak * baseline)

        df is the intrinsic frequency resolution 1/baseline oversampled by
        samples_per_peak, so a real peak cannot fall between grid points.

        The 2/baseline floor requires at least two full cycles inside the
        observing window — one cycle is not a detection, it is a trend.  On a
        multi-year ZTF baseline that floor alone would admit periods of many
        hundreds of days, so max_period_days is applied on top of it; without
        that the config knob would have no effect at all.
        """
        f_min = max(2.0 / baseline, 1.0 / self._cfg.max_period_days)
        f_max = 1.0 / self._cfg.min_period_days
        df = 1.0 / (self._cfg.samples_per_peak * baseline)

        if f_max <= f_min:
            raise ValueError(
                f"No periods are searchable: min_period_days="
                f"{self._cfg.min_period_days} is not below the longest "
                f"searchable period "
                f"{min(self._cfg.max_period_days, baseline / 2.0):.3f} d "
                f"(baseline {baseline:.2f} d, "
                f"max_period_days={self._cfg.max_period_days})."
            )

        freqs = np.arange(f_min, f_max, df, dtype=np.float64)
        if len(freqs) == 0:
            freqs = np.array([f_min], dtype=np.float64)
        # Ascending periods (descending frequencies).
        return (1.0 / freqs[::-1]).astype(np.float32).copy()

    # ------------------------------------------------------------------
    # Algorithm construction
    # ------------------------------------------------------------------

    def _build_algos(self) -> dict[str, Any]:
        import periodfind

        factories: dict[str, Any] = {
            # n_phase / n_mag / n_bins match scope-ml's generate_features.py
            # invocation, not the periodfind library defaults.
            "CE" : lambda: periodfind.ConditionalEntropy(n_phase=20, n_mag=10),
            "AOV": lambda: periodfind.AOV(n_phase=20),
            "LS" : lambda: periodfind.LombScargle(),
            "MHF": lambda: periodfind.MultiHarmonicFourier(max_harmonics=3),
            "FPW": lambda: periodfind.FPW(n_bins=20),
            "BLS": lambda: periodfind.BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.5),
        }
        return {
            name: factories[name]()
            for name in self._cfg.algorithms
            if name in factories
        }

    @property
    def algos(self) -> dict[str, Any]:
        if self._algos is None:
            self._algos = self._build_algos()
        return self._algos

    # ------------------------------------------------------------------
    # Peak extraction
    # ------------------------------------------------------------------

    def _extract_top_n(
        self,
        data: np.ndarray,
        use_minima: bool,
        n_top: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce one periodogram to its top-N *distinct* peaks.

        A real peak spans many adjacent frequency bins, so sorting the raw
        periodogram and taking the N largest values returns N samples of the
        same peak.  Instead the grid is split into
        n_chunks_multiplier * n_top equal chunks, the single best point in
        each chunk is taken, and those chunk winners are ranked.  That
        guarantees the returned peaks come from distinct regions of frequency
        space.  Ported from scope-ml's extract_top_n_periods.

        Returns (periods, significances), both length n_top, NaN-padded.
        """
        grid = self._grid
        assert grid is not None

        # periodfind returns shape (n_periods, n_period_dts).  We search no
        # chirp, so n_period_dts == 1; collapse the trailing axis by taking the
        # best trial along it so `flat` stays index-aligned with `grid`.
        arr = np.asarray(data).reshape(len(grid), -1)
        flat = arr.min(axis=1) if use_minima else arr.max(axis=1)

        periods = np.full(n_top, np.nan, dtype=np.float64)
        sigs = np.full(n_top, np.nan, dtype=np.float64)

        n_freqs = flat.size
        if n_freqs == 0:
            return periods, sigs

        mean_val = float(np.mean(flat))
        std_val = float(np.std(flat))
        if std_val == 0.0 or not np.isfinite(std_val):
            return periods, sigs

        n_chunks = self._cfg.n_chunks_multiplier * n_top
        chunk_size = n_freqs // n_chunks

        if chunk_size < 1:
            # Fewer grid points than chunks — fall back to a plain sort.
            order = np.argsort(flat)
            if not use_minima:
                order = order[::-1]
            n_fill = min(n_top, order.size)
            sel = order[:n_fill]
            periods[:n_fill] = grid[sel]
            sigs[:n_fill] = _signed_z(flat[sel], mean_val, std_val, use_minima)
            return periods, sigs

        # Trim to a whole number of chunks so the reduction can be vectorised;
        # the remainder is folded into the final chunk below.
        n_trim = chunk_size * n_chunks
        block = flat[:n_trim].reshape(n_chunks, chunk_size)
        local = block.argmin(axis=1) if use_minima else block.argmax(axis=1)
        best_idx = local + np.arange(n_chunks) * chunk_size

        if n_trim < n_freqs:
            tail = flat[n_trim:]
            tail_idx = n_trim + (tail.argmin() if use_minima else tail.argmax())
            cur = best_idx[-1]
            better = flat[tail_idx] < flat[cur] if use_minima else flat[tail_idx] > flat[cur]
            if better:
                best_idx[-1] = tail_idx

        chunk_sigs = _signed_z(flat[best_idx], mean_val, std_val, use_minima)
        order = np.argsort(chunk_sigs)[::-1]
        n_fill = min(n_top, order.size)
        sel = best_idx[order[:n_fill]]
        periods[:n_fill] = grid[sel]
        sigs[:n_fill] = chunk_sigs[order[:n_fill]]
        return periods, sigs

    def _sub_batch_size(self, n_sources: int) -> int:
        """Sources per periodfind call, bounded by the periodogram memory budget."""
        grid_len = len(self._grid) if self._grid is not None else 1
        per_source = max(grid_len * 4, 1)          # float32 periodogram
        n = max(1, _PERIODOGRAM_BUDGET_BYTES // per_source)
        return int(min(n, n_sources))

    def _run_algorithm(
        self,
        algo_name: str,
        algo: Any,
        times: list[np.ndarray],
        mags: list[np.ndarray],
        errs: list[np.ndarray],
        n_top: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Evaluate one algorithm over all sources, returning top-N peaks.

        Periodograms are requested in sub-batches and reduced immediately so
        peak host memory stays bounded regardless of the source count.
        Returns (periods, sigs) each of shape (n_sources, n_top), or None if
        the algorithm failed.
        """
        n_sources = len(times)
        top_periods = np.full((n_sources, n_top), np.nan, dtype=np.float64)
        top_sigs = np.full((n_sources, n_top), np.nan, dtype=np.float64)

        needs_errs = algo_name in ("MHF", "FPW", "BLS")
        step = self._sub_batch_size(n_sources)

        for lo in range(0, n_sources, step):
            hi = min(lo + step, n_sources)
            kwargs: dict[str, Any] = {"output": "periodogram"}
            if needs_errs:
                kwargs["errs"] = errs[lo:hi]

            try:
                pgrams = algo.calc(
                    times[lo:hi], mags[lo:hi], self._grid,
                    self._period_dts, **kwargs
                )
            except Exception:
                log.exception(
                    "Period algorithm %s failed on sources %d–%d of %d — "
                    "dropping its contribution for this batch",
                    algo_name, lo, hi, n_sources,
                )
                return None

            for k, pgram in enumerate(pgrams):
                # use_max is carried on the Periodogram object, so the
                # minimise/maximise direction never has to be hardcoded here.
                use_minima = not getattr(pgram, "use_max", True)
                p, s = self._extract_top_n(
                    np.asarray(pgram.data), use_minima, n_top
                )
                top_periods[lo + k] = p
                top_sigs[lo + k] = s

            del pgrams

        return top_periods, top_sigs

    # ------------------------------------------------------------------
    # Consensus scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _agreement_scores(
        top_periods: dict[str, np.ndarray],
        idx: int,
    ) -> dict[str, Any]:
        """Pairwise cross-algorithm agreement for one source.

        Ported from scope-ml's compute_agreement_scores.  Three tiers:

        agree_strict    fraction of algorithm pairs whose *top* periods match
        agree_score     fraction of pairs matching anywhere in their top-N
        agree_weighted  same, weighted by 1/(rank_i · rank_j) so a match
                        between two first-ranked peaks counts far more than
                        one between two tenth-ranked peaks

        Also returns best_consensus_period: the period attracting the most
        rank-weighted votes from other algorithms.
        """
        names = list(top_periods.keys())
        n = len(names)
        total_pairs = n * (n - 1) // 2

        empty = {
            "period_n_agree_pairs": 0,
            "period_n_total_pairs": total_pairs,
            "period_agree_score": 0.0,
            "period_agree_strict": 0.0,
            "period_agree_weighted": 0.0,
            "period_best_agree": np.nan,
            "period_best_consensus": np.nan,
        }
        if total_pairs == 0:
            return empty

        rows = {a: top_periods[a][idx] for a in names}

        n_strict = 0
        n_agree = 0
        best_agree = np.nan
        weighted_sum = 0.0

        for ii in range(n):
            for jj in range(ii + 1, n):
                ps_a, ps_b = rows[names[ii]], rows[names[jj]]

                if _period_match(float(ps_a[0]), float(ps_b[0])):
                    n_strict += 1

                matched = False
                pair_weight = 0.0
                for ri, pa in enumerate(ps_a):
                    if np.isnan(pa) or pa < _MIN_AGREE_PERIOD:
                        continue
                    for rj, pb in enumerate(ps_b):
                        if np.isnan(pb) or pb < _MIN_AGREE_PERIOD:
                            continue
                        if _period_match(float(pa), float(pb)):
                            if not matched:
                                matched = True
                                if np.isnan(best_agree):
                                    best_agree = float(pa)
                            w = 1.0 / ((ri + 1) * (rj + 1))
                            if w > pair_weight:
                                pair_weight = w
                if matched:
                    n_agree += 1
                weighted_sum += pair_weight

        # Rank-weighted votes: which period do the other algorithms support?
        votes: dict[float, float] = {}
        for aname in names:
            for rank, p in enumerate(rows[aname]):
                if np.isnan(p) or p < _MIN_AGREE_PERIOD:
                    continue
                p = float(p)
                weight = 0.0
                for other in names:
                    if other == aname:
                        continue
                    for rj, p_other in enumerate(rows[other]):
                        if np.isnan(p_other) or p_other < _MIN_AGREE_PERIOD:
                            continue
                        if _period_match(p, float(p_other)):
                            weight += 1.0 / ((rank + 1) * (rj + 1))
                            break
                key = next((k for k in votes if _period_match(p, k)), None)
                if key is None:
                    votes[p] = weight
                else:
                    votes[key] += weight

        best_consensus = np.nan
        if votes:
            cand = max(votes, key=lambda k: votes[k])
            if votes[cand] > 0:
                best_consensus = cand

        return {
            "period_n_agree_pairs": n_agree,
            "period_n_total_pairs": total_pairs,
            "period_agree_score": n_agree / total_pairs,
            "period_agree_strict": n_strict / total_pairs,
            "period_agree_weighted": weighted_sum / total_pairs,
            "period_best_agree": best_agree,
            "period_best_consensus": best_consensus,
        }

    @staticmethod
    def _family_scores(
        top_periods: dict[str, np.ndarray],
        top_sigs: dict[str, np.ndarray],
        f1_power: dict[str, np.ndarray],
        idx: int,
    ) -> dict[str, Any]:
        """Sidereal-alias family scoring for one source.

        Ported from scope-ml's compute_sidereal_family_scores.  Every top-N
        peak from every algorithm becomes a node; nodes linked by a sidereal
        alias relation are merged with union-find.  The family spanning the
        most distinct algorithms wins.

        This is more robust than pairwise agreement because algorithms
        routinely land on *different teeth of the same alias comb* — pairwise
        matching calls that a disagreement, family grouping correctly calls
        it a detection.
        """
        names = list(top_periods.keys())

        entries: list[tuple[str, int, float, float]] = []
        for aname in names:
            prow, srow = top_periods[aname][idx], top_sigs[aname][idx]
            for rank, (p, s) in enumerate(zip(prow, srow)):
                if not np.isnan(p) and p > 0:
                    entries.append((aname, rank, float(p), float(s)))

        n_total = len(entries)
        if n_total == 0:
            return {
                "period_family_n_algos": 0,
                "period_family_rank_score": 0.0,
                "period_family_n_members": 0,
                "period_family_n_total": 0,
                "period_family_best": np.nan,
                "period_family_algorithm": "",
            }

        parent = list(range(n_total))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n_total):
            for j in range(i + 1, n_total):
                if _are_sidereal_aliases(entries[i][2], entries[j][2]):
                    union(i, j)

        families: dict[int, list[tuple[str, int, float, float]]] = {}
        for i, entry in enumerate(entries):
            families.setdefault(find(i), []).append(entry)

        best_members: list[tuple[str, int, float, float]] = []
        best_score = (-1, -1.0)
        for members in families.values():
            algo_set = {m[0] for m in members}
            best_rank = {}
            for aname, rank, _p, _s in members:
                if aname not in best_rank or rank < best_rank[aname]:
                    best_rank[aname] = rank
            # Reward breadth first (how many algorithms), then how highly
            # each of them ranked its member of this family.
            rank_score = sum(1.0 / (r + 1) for r in best_rank.values())
            score = (len(algo_set), rank_score)
            if score > best_score:
                best_score = score
                best_members = members

        n_algos, rank_score = best_score
        family_algos = {m[0] for m in best_members}

        # Prefer the algorithm whose top period gives the strongest Fourier
        # fit, among those represented in the winning family.
        best_period = np.nan
        best_algo = ""
        best_f1 = np.nan
        for aname in names:
            if aname not in family_algos:
                continue
            f1 = float(f1_power[aname][idx])
            if np.isnan(f1):
                continue
            if np.isnan(best_f1) or f1 > best_f1:
                best_f1 = f1
                best_algo = aname
                best_period = float(top_periods[aname][idx][0])

        # Fallback: highest-significance top-ranked member of the family.
        if np.isnan(best_period):
            best_sig = -np.inf
            for aname, rank, p, s in best_members:
                if rank == 0 and s > best_sig:
                    best_sig, best_period, best_algo = s, p, aname

        return {
            "period_family_n_algos": n_algos,
            "period_family_rank_score": rank_score,
            "period_family_n_members": len(best_members),
            "period_family_n_total": n_total,
            "period_family_best": best_period,
            "period_family_algorithm": best_algo,
        }

    # ------------------------------------------------------------------
    # Fourier
    # ------------------------------------------------------------------

    @staticmethod
    def _fourier(
        times: list[np.ndarray],
        mags: list[np.ndarray],
        errs: list[np.ndarray],
        periods: np.ndarray,
    ) -> np.ndarray | None:
        """Batched Fourier decomposition at the supplied per-source periods.

        Only sources with a finite period are submitted; the caller is
        responsible for mapping rows back.  Returns None on failure.
        """
        import periodfind

        if not times:
            return None
        try:
            return periodfind.FourierDecomposition().calc(
                times, mags, errs, periods.astype(np.float32)
            )
        except Exception:
            log.exception("FourierDecomposition failed for %d sources", len(times))
            return None

    @staticmethod
    def _unpack_fourier(row: np.ndarray) -> dict[str, Any]:
        """Map a 14-element FourierDecomposition row to FeatureVector fields.

        Column order: [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3,
                       A4, B4, A5, B5]

        Phase convention matches scope-ml (_ab_to_amp_phi in periodsearch.py):
            phi    = arctan2(A, B)                       (not arctan2(B, A))
            relphi = (phi_k / k − phi_1) / (2π/k) % 1
        """
        out: dict[str, Any] = {
            "f1_power": float(row[0]),
            "f1_bic": float(row[1]),
            # offset (col 2) and slope (col 3) have no FeatureVector fields
            "f1_a": float(row[4]),
            "f1_b": float(row[5]),
        }

        a1, b1 = float(row[4]), float(row[5])
        amp1 = float(np.sqrt(a1**2 + b1**2))
        phi1 = float(np.arctan2(a1, b1))
        out["f1_amp"] = amp1
        out["f1_phi0"] = phi1

        for k in range(1, 5):          # harmonics 2–5 → relative indices 1–4
            n = k + 1
            a_k = float(row[4 + 2 * k])
            b_k = float(row[5 + 2 * k])
            if a_k == 0.0 and b_k == 0.0:
                out[f"f1_relamp{k}"] = np.nan
                out[f"f1_relphi{k}"] = np.nan
            else:
                amp_k = float(np.sqrt(a_k**2 + b_k**2))
                phi_k = float(np.arctan2(a_k, b_k))
                out[f"f1_relamp{k}"] = amp_k / amp1 if amp1 > 0 else np.nan
                out[f"f1_relphi{k}"] = float(
                    (phi_k / n - phi1) / (2.0 * np.pi / n) % 1
                )

        return out

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(
        lc: LightCurve,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sort, zero-offset and normalise one light curve.

        Returns (t_raw, m_raw, e_raw, t_pf, m_pf) where the _pf arrays are
        what period finding consumes and the _raw arrays feed Fourier.

        The float64 subtraction before the float32 cast is not cosmetic.
        ZTF timestamps are HJD ≈ 2.458e6, where one float32 ULP is 0.25 days
        — six hours.  Casting first snaps every epoch onto a six-hour grid and
        destroys all sub-day timing, which is the entire signal for a short
        period binary.  Subtracting t_min first brings the values into
        [0, ~2000] where a float32 ULP is well under a millisecond.
        Same ordering as scope-ml's _prepare_lightcurves.
        """
        order = np.argsort(lc.time)
        t64 = np.asarray(lc.time, dtype=np.float64)[order]
        m64 = np.asarray(lc.mag, dtype=np.float64)[order]
        e64 = np.asarray(lc.mag_err, dtype=np.float64)[order]

        # min()/max() propagate a single NaN across the whole array, so one
        # bad epoch would blank the entire light curve.  LightCurve does not
        # drop non-finite values (see types.py), so they can reach here; the
        # nan-aware reductions keep the damage at the offending index, which
        # is what to_float32_time does for the other extractors.
        finite_t = np.isfinite(t64)
        t_zero = (
            (t64 - t64[finite_t].min()).astype(np.float32)
            if finite_t.any() else t64.astype(np.float32)
        )

        m_min   = np.nanmin(m64) if np.isfinite(m64).any() else np.nan
        m_range = (np.nanmax(m64) - m_min) if np.isfinite(m64).any() else np.nan
        m_norm = ((m64 - m_min) / m_range) if m_range > 0 else np.zeros_like(m64)

        return (
            t_zero,
            m64.astype(np.float32),
            e64.astype(np.float32),
            t_zero,
            m_norm.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, sources: list[list[LightCurve]]) -> list[dict[str, Any]]:
        """Find periods and Fourier features for a batch of sources.

        Parameters
        ----------
        sources:
            Each element is the light curves for one output row.  The pipeline
            passes single-band groups when per-band rows are enabled; if a
            group holds several bands the one with the most epochs is used.

        Returns
        -------
        list[dict[str, Any]]
            One dict per input element, with period, consensus scores,
            per-algorithm top-N peaks and the 14 Fourier fields.  Fields are
            np.nan / empty where nothing could be computed.
        """
        n_top = self._cfg.top_n_periods

        nan_result: dict[str, Any] = {
            "period": np.nan,
            "period_significance": np.nan,
            "period_algorithm": "",
            "period_top": {},
            "significance_top": {},
            "period_n_agree_pairs": 0,
            "period_n_total_pairs": 0,
            "period_agree_score": np.nan,
            "period_agree_strict": np.nan,
            "period_agree_weighted": np.nan,
            "period_best_agree": np.nan,
            "period_best_consensus": np.nan,
            "period_family_n_algos": 0,
            "period_family_rank_score": np.nan,
            "period_family_n_members": 0,
            "period_family_n_total": 0,
            "period_family_best": np.nan,
            "period_family_algorithm": "",
            "f1_power": np.nan, "f1_bic": np.nan,
            "f1_a": np.nan, "f1_b": np.nan,
            "f1_amp": np.nan, "f1_phi0": np.nan,
            "f1_relamp1": np.nan, "f1_relphi1": np.nan,
            "f1_relamp2": np.nan, "f1_relphi2": np.nan,
            "f1_relamp3": np.nan, "f1_relphi3": np.nan,
            "f1_relamp4": np.nan, "f1_relphi4": np.nan,
        }

        results: list[dict[str, Any]] = [dict(nan_result) for _ in sources]
        if not sources:
            return results

        # ── 1. Preprocess ────────────────────────────────────────────────
        times, mags, errs = [], [], []
        times_pf, mags_pf = [], []
        valid_idx: list[int] = []

        for i, lcs in enumerate(sources):
            if not lcs:
                continue
            lc = max(lcs, key=lambda x: x.n_obs)
            if lc.n_obs < 4:
                continue
            t_raw, m_raw, e_raw, t_pf, m_pf = self._prepare(lc)
            times.append(t_raw)
            mags.append(m_raw)
            errs.append(e_raw)
            times_pf.append(t_pf)
            mags_pf.append(m_pf)
            valid_idx.append(i)

        if not valid_idx or not self.algos:
            return results

        # The grid should have been built by prepare() over the whole field.
        # Falling back to this batch's baseline keeps single-batch callers
        # (tests, single_latency.py) working, but is flagged because it makes
        # results depend on batch composition.
        if self._grid is None:
            log.warning(
                "prepare() was not called — building the period grid from this "
                "batch's baseline only.  Periods will not be comparable across "
                "batches."
            )
            self.prepare(sources)

        # ── 2. Periodograms → top-N distinct peaks per algorithm ─────────
        n_valid = len(valid_idx)
        top_periods: dict[str, np.ndarray] = {}
        top_sigs: dict[str, np.ndarray] = {}

        for algo_name, algo in self.algos.items():
            out = self._run_algorithm(
                algo_name, algo, times_pf, mags_pf, errs, n_top
            )
            if out is None:
                continue
            top_periods[algo_name], top_sigs[algo_name] = out

        if not top_periods:
            log.error("All %d period algorithms failed for a batch of %d sources",
                      len(self.algos), n_valid)
            return results

        # ── 3. Fourier at each algorithm's top period (for family scoring) ─
        f1_power: dict[str, np.ndarray] = {}
        for algo_name, periods in top_periods.items():
            best = periods[:, 0]
            mask = np.isfinite(best) & (best > 0)
            powers = np.full(n_valid, np.nan, dtype=np.float64)
            if mask.any():
                sel = np.flatnonzero(mask)
                raw = self._fourier(
                    [times[i] for i in sel],
                    [mags[i] for i in sel],
                    [errs[i] for i in sel],
                    best[sel],
                )
                if raw is not None:
                    powers[sel] = np.asarray(raw)[:, 0]
            f1_power[algo_name] = powers

        # ── 4. Consensus scoring ─────────────────────────────────────────
        per_source: list[dict[str, Any]] = []
        chosen = np.full(n_valid, np.nan, dtype=np.float64)

        for k in range(n_valid):
            scores = self._agreement_scores(top_periods, k)
            scores.update(self._family_scores(top_periods, top_sigs, f1_power, k))

            # Preference order: the sidereal family is the most robust
            # discriminator, then the rank-weighted consensus, then simply
            # the most significant top peak across algorithms.
            period = scores["period_family_best"]
            algo = scores["period_family_algorithm"]
            if np.isnan(period):
                period = scores["period_best_consensus"]
                algo = "consensus"
            if np.isnan(period):
                best_sig = -np.inf
                for name in top_periods:
                    s = float(top_sigs[name][k][0])
                    p = float(top_periods[name][k][0])
                    if np.isfinite(p) and s > best_sig:
                        best_sig, period, algo = s, p, name

            sig = np.nan
            if algo in top_sigs:
                sig = float(top_sigs[algo][k][0])

            scores["period"] = float(period) if period is not None else np.nan
            scores["period_significance"] = sig
            scores["period_algorithm"] = algo
            scores["period_top"] = {
                a: top_periods[a][k].tolist() for a in top_periods
            }
            scores["significance_top"] = {
                a: top_sigs[a][k].tolist() for a in top_sigs
            }
            per_source.append(scores)
            chosen[k] = scores["period"]

        # ── 5. Fourier at the chosen period ──────────────────────────────
        mask = np.isfinite(chosen) & (chosen > 0)
        fourier_rows: dict[int, np.ndarray] = {}
        if mask.any():
            sel = np.flatnonzero(mask)
            raw = self._fourier(
                [times[i] for i in sel],
                [mags[i] for i in sel],
                [errs[i] for i in sel],
                chosen[sel],
            )
            if raw is not None:
                raw = np.asarray(raw)
                for row_pos, k in enumerate(sel):
                    fourier_rows[int(k)] = raw[row_pos]

        # ── 6. Assemble ──────────────────────────────────────────────────
        for k, src_idx in enumerate(valid_idx):
            out = dict(nan_result)
            out.update(per_source[k])
            if k in fourier_rows:
                out.update(self._unpack_fourier(fourier_rows[k]))
            results[src_idx] = out

        return results
