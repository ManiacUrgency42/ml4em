"""Tests for ml4em.features.period — pure Python logic and mocked batch paths.

_unpack_fourier, _agreement_scores and _family_scores are pure Python and
tested without any mock.  extract() tests use patch.dict(sys.modules) to
satisfy `import periodfind` inside the extractor without requiring the
compiled library.

Consensus model
---------------
Two independent scorers run over the same top-N peak table:

_agreement_scores  pairwise.  For each (algorithm, algorithm) pair, do any of
                   their top-N peaks match under _period_match?  Reported as
                   strict (top peak only), plain (anywhere in the top-N) and
                   rank-weighted fractions, plus the period attracting the most
                   rank-weighted votes.

_family_scores     union-find.  Every peak from every algorithm is a node;
                   nodes related by a sidereal-day alias are merged.  The
                   family spanning the most distinct algorithms wins.  This
                   catches the common case where algorithms found the same star
                   but landed on different teeth of its alias comb — pairwise
                   matching scores that as disagreement.

Phase convention (scope-ml aligned)
------------------------------------
phi  = arctan2(A, B)   — A (cosine coefficient) is the FIRST argument.
relphi = (phi_k / k - phi_1) / (2π/k) % 1  — normalised to [0, 1].
"""

import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml4em.config.schema import PeriodConfig
from ml4em.constants import SIDEREAL_FREQ_PER_DAY
from ml4em.features.period import (
    PeriodExtractor,
    _MIN_AGREE_PERIOD,
    _are_sidereal_aliases,
    _period_match,
)
from ml4em.types import LightCurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lc(n: int = 20, source_id: str = "src_001") -> LightCurve:
    rng = np.random.default_rng(0)
    return LightCurve(
        source_id=source_id,
        time=np.linspace(2459000.0, 2459200.0, n),
        mag=rng.normal(18.5, 0.2, n),
        mag_err=np.full(n, 0.05),
        band="r",
        survey="ztf",
        ra=180.0,
        dec=30.0,
    )


def _rows(**by_algo: list[float]) -> dict[str, np.ndarray]:
    """Build the {algorithm: (n_sources, n_top) period table} the scorers take.

    Each keyword is one algorithm's ranked peak list for a single source.
    Lists are NaN-padded to a common width, matching what _run_algorithm emits.
    """
    width = max(len(v) for v in by_algo.values())
    out = {}
    for name, values in by_algo.items():
        row = np.full((1, width), np.nan, dtype=np.float64)
        row[0, : len(values)] = values
        out[name] = row
    return out


def _make_periodogram(data: np.ndarray, use_max: bool) -> MagicMock:
    """Minimal stand-in for a periodfind Periodogram object."""
    pgram = MagicMock()
    pgram.data = data
    pgram.use_max = use_max
    return pgram


@pytest.fixture
def period_extractor():
    """PeriodExtractor constructed with periodfind mocked out.

    Suitable for calling pure-Python methods (_agreement_scores,
    _family_scores, _unpack_fourier) without needing periodfind present.
    """
    mock_pf = MagicMock()
    with patch.dict(sys.modules, {"periodfind": mock_pf}):
        return PeriodExtractor(PeriodConfig())


# ---------------------------------------------------------------------------
# _period_match — module-level helper for harmonic-aware period comparison
# ---------------------------------------------------------------------------

class TestPeriodMatch:

    def test_exact_match(self):
        """Identical periods always match."""
        assert _period_match(1.0, 1.0) is True

    def test_within_tolerance_direct(self):
        """Periods within 5% at harmonic h=1 match."""
        assert _period_match(1.0, 1.04) is True   # 3.8% apart
        # |1.0/(1.05*1) - 1| = |0.9524 - 1| = 0.0476 < 0.05 → True
        assert _period_match(1.0, 1.05) is True

    def test_outside_tolerance_all_harmonics(self):
        """Periods 20% apart (no harmonic relationship) do not match."""
        assert _period_match(1.0, 1.2) is False

    def test_half_period_harmonic(self):
        """P_a = 0.5 * P_b matches at harmonic h=0.5."""
        assert _period_match(1.0, 2.0) is True

    def test_double_period_harmonic(self):
        """P_a = 2 * P_b matches at harmonic h=2."""
        assert _period_match(2.0, 1.0) is True

    def test_third_period_harmonic(self):
        """P_a ≈ P_b / 3 matches at harmonic h=1/3."""
        assert _period_match(1.0, 3.0) is True

    def test_triple_period_harmonic(self):
        """P_a ≈ 3 * P_b matches at harmonic h=3."""
        assert _period_match(3.0, 1.0) is True

    def test_sub_minimum_period_rejected(self):
        """Periods below _MIN_AGREE_PERIOD are always rejected."""
        p_tiny = _MIN_AGREE_PERIOD * 0.5
        assert _period_match(p_tiny, p_tiny) is False

    def test_nan_periods_rejected(self):
        assert _period_match(np.nan, 1.0) is False
        assert _period_match(1.0, np.nan) is False


# ---------------------------------------------------------------------------
# _unpack_fourier — pure Python static method, no mock required
# ---------------------------------------------------------------------------

def _fourier_row(
    power: float = 0.8, bic: float = -50.0,
    a1: float = 1.0, b1: float = 0.0,
    a2: float = 0.5, b2: float = 0.0,
) -> np.ndarray:
    """Build a 14-element FourierDecomposition row.

    Column layout: [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
    """
    row = np.zeros(14, dtype=np.float64)
    row[0] = power
    row[1] = bic
    # cols 2 (offset) and 3 (slope) have no FeatureVector fields
    row[4] = a1; row[5] = b1
    row[6] = a2; row[7] = b2
    return row


class TestUnpackFourier:

    def test_first_harmonic_amplitude(self):
        """f1_amp = sqrt(a1² + b1²)."""
        row = _fourier_row(a1=3.0, b1=4.0)
        out = PeriodExtractor._unpack_fourier(row)
        assert abs(out["f1_amp"] - 5.0) < 1e-6

    def test_first_harmonic_phase_scope_ml_convention(self):
        """f1_phi0 = arctan2(a1, b1) — scope-ml uses A as the first argument.

        arctan2(A=1, B=0) = π/2  (not 0, which the old arctan2(B, A) returned).
        """
        row = _fourier_row(a1=1.0, b1=0.0)
        out = PeriodExtractor._unpack_fourier(row)
        assert abs(out["f1_phi0"] - math.pi / 2) < 1e-6

    def test_first_harmonic_phase_pure_sine(self):
        """arctan2(A=0, B=1) = 0 for a pure-sine component."""
        row = _fourier_row(a1=0.0, b1=1.0)
        out = PeriodExtractor._unpack_fourier(row)
        assert abs(out["f1_phi0"] - 0.0) < 1e-6

    def test_zero_higher_harmonic_returns_nan(self):
        """When a2=b2=0 the relative amplitude and phase must be NaN, not 0 or inf."""
        row = _fourier_row(a1=1.0, b1=0.0, a2=0.0, b2=0.0)
        out = PeriodExtractor._unpack_fourier(row)
        assert np.isnan(out["f1_relamp1"])
        assert np.isnan(out["f1_relphi1"])

    def test_relative_amplitude_ratio(self):
        """f1_relamp1 = amplitude of 2nd harmonic / amplitude of 1st harmonic."""
        row = _fourier_row(a1=4.0, b1=0.0, a2=2.0, b2=0.0)
        out = PeriodExtractor._unpack_fourier(row)
        assert abs(out["f1_relamp1"] - 0.5) < 1e-6

    def test_relative_phase_normalization(self):
        """f1_relphi1 = (phi_2 / 2 - phi_1) / (π) % 1  — scope-ml formula.

        With A1=1, B1=0: phi_1 = arctan2(1, 0) = π/2
        With A2=1, B2=0: phi_2 = arctan2(1, 0) = π/2
        relphi1 = (π/2 / 2 - π/2) / π % 1
                = (-π/4) / π % 1
                = -0.25 % 1 = 0.75
        """
        row = _fourier_row(a1=1.0, b1=0.0, a2=1.0, b2=0.0)
        out = PeriodExtractor._unpack_fourier(row)
        assert abs(out["f1_relphi1"] - 0.75) < 1e-6

    def test_power_and_bic_fields_populated(self):
        row = _fourier_row(power=0.75, bic=-120.0, a1=1.0, b1=0.0)
        out = PeriodExtractor._unpack_fourier(row)
        assert abs(out["f1_power"] - 0.75) < 1e-6
        assert abs(out["f1_bic"] - (-120.0)) < 1e-6


# ---------------------------------------------------------------------------
# _are_sidereal_aliases — module-level helper
# ---------------------------------------------------------------------------

class TestSiderealAliases:

    def test_identical_periods_are_aliases(self):
        assert _are_sidereal_aliases(0.4, 0.4) is True

    def test_one_sidereal_step_apart(self):
        """f and f + f_sidereal are indistinguishable to a ground-based survey."""
        f = 1.0 / 0.4
        assert _are_sidereal_aliases(0.4, 1.0 / (f + SIDEREAL_FREQ_PER_DAY)) is True

    def test_negative_sidereal_step(self):
        f = 1.0 / 0.2
        assert _are_sidereal_aliases(0.2, 1.0 / (f - SIDEREAL_FREQ_PER_DAY)) is True

    def test_unrelated_periods_are_not_aliases(self):
        # 0.4 d and 0.63 d differ by 1.09 c/d — not near any integer multiple
        # of the 1.0027 c/d sidereal frequency at the 3% tolerance.
        assert _are_sidereal_aliases(0.4, 0.63) is False

    def test_nan_rejected(self):
        assert _are_sidereal_aliases(np.nan, 0.4) is False


# ---------------------------------------------------------------------------
# _agreement_scores — pairwise cross-algorithm agreement
# ---------------------------------------------------------------------------

class TestAgreementScores:

    def test_single_algorithm_has_no_pairs(self, period_extractor):
        """One algorithm forms zero pairs, so every fraction is undefined → 0."""
        out = period_extractor._agreement_scores(_rows(CE=[1.23]), idx=0)
        assert out["period_n_total_pairs"] == 0
        assert out["period_n_agree_pairs"] == 0
        assert np.isnan(out["period_best_consensus"])

    def test_all_nan_periods_score_zero(self, period_extractor):
        out = period_extractor._agreement_scores(
            _rows(CE=[np.nan], AOV=[np.nan]), idx=0
        )
        assert out["period_n_total_pairs"] == 1
        assert out["period_agree_score"] == 0.0
        assert np.isnan(out["period_best_consensus"])

    def test_two_agreeing_algorithms(self, period_extractor):
        """1.00 and 1.01 are 1% apart — inside the 5% tolerance."""
        out = period_extractor._agreement_scores(
            _rows(CE=[1.00], AOV=[1.01]), idx=0
        )
        assert out["period_n_agree_pairs"] == 1
        assert out["period_agree_score"] == 1.0
        assert out["period_agree_strict"] == 1.0
        assert abs(out["period_best_consensus"] - 1.00) < 1e-9

    def test_disagreeing_algorithms_score_zero(self, period_extractor):
        """1.0 and 1.2 are 20% apart with no harmonic relationship."""
        out = period_extractor._agreement_scores(
            _rows(CE=[1.0], LS=[1.2]), idx=0
        )
        assert out["period_n_agree_pairs"] == 0
        assert out["period_agree_score"] == 0.0
        assert np.isnan(out["period_best_consensus"])

    def test_harmonic_counts_as_agreement(self, period_extractor):
        """An algorithm at 2P agrees with one at P via the harmonic check."""
        out = period_extractor._agreement_scores(
            _rows(CE=[1.0], AOV=[2.0]), idx=0
        )
        assert out["period_n_agree_pairs"] == 1

    def test_strict_ignores_lower_ranked_matches(self, period_extractor):
        """Top peaks disagree but rank-2 peaks match: agree_score 1, strict 0.

        This is the case top-N retention exists for — the correct period is
        present in both algorithms, just not ranked first by either.
        """
        out = period_extractor._agreement_scores(
            _rows(CE=[5.0, 1.00], AOV=[9.0, 1.01]), idx=0
        )
        assert out["period_agree_score"] == 1.0
        assert out["period_agree_strict"] == 0.0

    def test_weighted_penalises_low_rank_matches(self, period_extractor):
        """A match between two rank-1 peaks outweighs one between rank-2 peaks."""
        top = period_extractor._agreement_scores(
            _rows(CE=[1.00], AOV=[1.01]), idx=0
        )
        deep = period_extractor._agreement_scores(
            _rows(CE=[5.0, 1.00], AOV=[9.0, 1.01]), idx=0
        )
        assert top["period_agree_weighted"] > deep["period_agree_weighted"]

    def test_sub_cadence_periods_excluded(self, period_extractor):
        """Peaks below _MIN_AGREE_PERIOD cannot contribute agreement."""
        tiny = _MIN_AGREE_PERIOD * 0.5
        out = period_extractor._agreement_scores(
            _rows(CE=[tiny], AOV=[tiny]), idx=0
        )
        assert out["period_n_agree_pairs"] == 0
        assert np.isnan(out["period_best_consensus"])

    def test_consensus_prefers_the_widely_supported_period(self, period_extractor):
        """Three algorithms back 1.0 d; a fourth is alone at 5.0 d."""
        out = period_extractor._agreement_scores(
            _rows(CE=[1.000], AOV=[1.010], LS=[1.015], MHF=[5.0]), idx=0
        )
        assert abs(out["period_best_consensus"] - 1.0) < 0.02
        assert out["period_n_total_pairs"] == 6
        assert out["period_n_agree_pairs"] == 3   # the three mutual pairs only


# ---------------------------------------------------------------------------
# _family_scores — sidereal alias family grouping
# ---------------------------------------------------------------------------

def _sig_rows(like: dict[str, np.ndarray], value: float = 5.0) -> dict[str, np.ndarray]:
    """Significance table shaped like `like`, constant where the period is set."""
    return {k: np.where(np.isnan(v), np.nan, value) for k, v in like.items()}


class TestFamilyScores:

    def test_no_peaks_returns_zeros(self, period_extractor):
        periods = _rows(CE=[np.nan], AOV=[np.nan])
        out = period_extractor._family_scores(
            periods, _sig_rows(periods), {"CE": np.array([np.nan]), "AOV": np.array([np.nan])}, idx=0
        )
        assert out["period_family_n_total"] == 0
        assert np.isnan(out["period_family_best"])

    def test_aliased_algorithms_form_one_family(self, period_extractor):
        """Two algorithms one sidereal step apart are the same detection."""
        f = 1.0 / 0.4
        aliased = 1.0 / (f + SIDEREAL_FREQ_PER_DAY)
        periods = _rows(CE=[0.4], AOV=[aliased])
        f1 = {"CE": np.array([0.9]), "AOV": np.array([0.3])}
        out = period_extractor._family_scores(periods, _sig_rows(periods), f1, idx=0)
        assert out["period_family_n_algos"] == 2
        assert out["period_family_n_members"] == 2
        # CE has the stronger Fourier fit, so it supplies the representative.
        assert out["period_family_algorithm"] == "CE"
        assert abs(out["period_family_best"] - 0.4) < 1e-9

    def test_unrelated_periods_split_into_separate_families(self, period_extractor):
        periods = _rows(CE=[0.4], AOV=[0.63])
        f1 = {"CE": np.array([0.9]), "AOV": np.array([0.3])}
        out = period_extractor._family_scores(periods, _sig_rows(periods), f1, idx=0)
        assert out["period_family_n_total"] == 2
        assert out["period_family_n_algos"] == 1
        assert out["period_family_n_members"] == 1

    def test_broad_family_beats_a_narrow_one(self, period_extractor):
        """A 3-algorithm family wins over a 1-algorithm family regardless of rank."""
        f = 1.0 / 0.4
        periods = _rows(
            CE =[0.4],
            AOV=[1.0 / (f + SIDEREAL_FREQ_PER_DAY)],
            LS =[1.0 / (f + 2 * SIDEREAL_FREQ_PER_DAY)],
            MHF=[7.0],
        )
        f1 = {
            "CE": np.array([0.5]), "AOV": np.array([0.4]),
            "LS": np.array([0.3]), "MHF": np.array([0.99]),
        }
        out = period_extractor._family_scores(periods, _sig_rows(periods), f1, idx=0)
        assert out["period_family_n_algos"] == 3
        # MHF has by far the best Fourier power but is not in the winning
        # family, so it must not supply the representative period.
        assert out["period_family_algorithm"] != "MHF"
        assert abs(out["period_family_best"] - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# extract() — short-circuit paths that avoid calling algo.calc
# ---------------------------------------------------------------------------

def test_extract_empty_input_returns_empty_list(period_extractor):
    # extract([]) returns before `import periodfind` — no sys.modules mock needed.
    assert period_extractor.extract([]) == []


def test_extract_insufficient_obs_returns_nan_result_per_source():
    """Sources with < 4 observations receive a fully nan-filled result dict."""
    mock_pf = MagicMock()
    with patch.dict(sys.modules, {"periodfind": mock_pf}):
        ext = PeriodExtractor(PeriodConfig())
        lc_short = _make_lc(n=3)  # n_obs=3 < 4 threshold
        results = ext.extract([[lc_short]])

    assert len(results) == 1
    assert np.isnan(results[0]["period"])
    assert np.isnan(results[0]["period_significance"])
    assert results[0]["period_algorithm"] == ""


def test_extract_batch_length_preserved_with_short_sources():
    """Output list length matches input even when every source is skipped."""
    mock_pf = MagicMock()
    with patch.dict(sys.modules, {"periodfind": mock_pf}):
        ext = PeriodExtractor(PeriodConfig())
        sources = [[_make_lc(n=2)], [_make_lc(n=3, source_id="s2")]]
        results = ext.extract(sources)

    assert len(results) == 2
    for r in results:
        assert np.isnan(r["period"])


def _run_extract_with_mock_ce(lc: LightCurve, spike_at: int | None = None):
    """Run extract() on one light curve with CE and Fourier mocked.

    Returns (extractor, result_dict, calc_call_args).  When `spike_at` is given
    the periodogram is flat except for one sharp minimum at that grid index, so
    the peak the extractor picks is known exactly.
    """
    mock_pf = MagicMock()
    mock_algo = MagicMock()
    mock_pf.ConditionalEntropy.return_value = mock_algo

    with patch.dict(sys.modules, {"periodfind": mock_pf}):
        ext = PeriodExtractor(PeriodConfig(algorithms=["CE"]))
        # prepare() builds the grid; the mocked periodogram must match its
        # length or _extract_top_n's reshape is meaningless.
        ext.prepare([[lc]])
        n_per = len(ext._grid)

        if spike_at is None:
            data = np.random.default_rng(0).normal(size=(n_per, 1)).astype(np.float32)
        else:
            data = np.ones((n_per, 1), dtype=np.float32)
            data[spike_at, 0] = -50.0   # CE minimises, so this bin must win

        mock_algo.calc.return_value = [_make_periodogram(data, use_max=False)]
        # FourierDecomposition.calc returns one 14-column row per source.
        mock_pf.FourierDecomposition.return_value.calc.return_value = np.zeros(
            (1, 14), dtype=np.float64
        )
        result = ext.extract([[lc]])[0]

    assert mock_algo.calc.called
    _, call_args, _ = mock_algo.calc.mock_calls[0]
    return ext, result, call_args


def test_extract_period_finding_uses_normalised_mags():
    """algo.calc must receive mags in [0, 1], not raw magnitudes."""
    _, _, call_args = _run_extract_with_mock_ce(_make_lc(n=20))
    mags_passed = call_args[1]   # second positional arg is mags_pf
    for m in mags_passed:
        assert m.min() >= 0.0 - 1e-6
        assert m.max() <= 1.0 + 1e-6


def test_extract_period_finding_uses_zeroed_times():
    """algo.calc must receive times starting near 0, not raw HJD (~2459000).

    float32 has ~7 significant digits, so its spacing at HJD 2.459e6 is exactly
    0.25 d — six hours.  Passing raw HJD would quantise every timestamp onto a
    6-hour lattice and make sub-day period finding meaningless.  Zeroing must
    happen in float64 before the cast.
    """
    _, _, call_args = _run_extract_with_mock_ce(_make_lc(n=20))
    times_passed = call_args[0]   # first positional arg is times_pf
    for t in times_passed:
        assert t.dtype == np.float32
        assert t.min() < 1.0


def test_extract_picks_the_periodogram_extremum():
    """The reported peak must be the grid point at the periodogram's minimum."""
    lc = _make_lc(n=20)
    ext, result, _ = _run_extract_with_mock_ce(lc, spike_at=0)
    spike = len(ext._grid) // 3
    ext, result, _ = _run_extract_with_mock_ce(lc, spike_at=spike)

    assert result["period_top"]["CE"][0] == pytest.approx(ext._grid[spike])
    assert result["period"] == pytest.approx(ext._grid[spike])
    assert result["period_algorithm"] != ""


def test_extract_pads_top_n_to_configured_length():
    """Each algorithm reports exactly top_n_periods ranked peaks."""
    cfg_n = PeriodConfig().top_n_periods
    _, result, _ = _run_extract_with_mock_ce(_make_lc(n=20), spike_at=100)
    assert len(result["period_top"]["CE"]) == cfg_n
    assert len(result["significance_top"]["CE"]) == cfg_n
