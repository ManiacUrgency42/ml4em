"""Tests for ml4em.features.period — pure Python logic and mocked batch paths.

_unpack_fourier and _agree are pure Python and tested without any mock.
extract() tests use patch.dict(sys.modules) to satisfy `import periodfind`
inside the extractor without requiring the compiled library.

Note on _agree semantics
------------------------
best_count is initialised to 1 (the highest-significance lone algorithm).
An algorithm's count is the number of OTHER algorithms whose peak agrees
with its period via _period_match (harmonic-aware, _AGREE_TOL = 5%).
An algorithm only replaces the current best when count > best_count, or
count == best_count with higher significance.
Consequently, three mutually-agreeing algorithms (count=2 each) can beat a
lone high-significance outlier (effective count=1).

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
from ml4em.features.period import (
    PeriodExtractor,
    _MIN_AGREE_PERIOD,
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


def _make_peak(period: float, sig: float) -> MagicMock:
    """Minimal mock of a periodfind Peak object."""
    peak = MagicMock()
    peak.params = [period]
    peak.significance = sig
    return peak


@pytest.fixture
def period_extractor():
    """PeriodExtractor constructed with periodfind mocked out.

    Suitable for calling pure-Python methods (_agree, _unpack_fourier) without
    needing periodfind present after construction.
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
# _agree — pure Python, uses mock peak objects
# ---------------------------------------------------------------------------

class TestAgree:

    def test_no_candidates_returns_nan(self, period_extractor):
        """Empty peak lists for every algorithm → all-NaN result."""
        all_peaks = {"CE": [[]], "AOV": [[]]}
        p, s, a = period_extractor._agree(all_peaks, src_idx=0)
        assert np.isnan(p)
        assert np.isnan(s)
        assert a == ""

    def test_single_algorithm_returns_that_period(self, period_extractor):
        peak = _make_peak(period=1.23, sig=0.9)
        all_peaks = {"CE": [[peak]]}
        p, s, a = period_extractor._agree(all_peaks, src_idx=0)
        assert abs(p - 1.23) < 1e-9
        assert a == "CE"

    def test_two_agreeing_algorithms_higher_sig_wins(self, period_extractor):
        """When two algorithms agree within 5%, the one with higher significance wins."""
        peak_ce  = _make_peak(period=1.00, sig=0.7)
        peak_aov = _make_peak(period=1.01, sig=0.6)  # 1% apart, within 5% tolerance
        all_peaks = {"CE": [[peak_ce]], "AOV": [[peak_aov]]}
        p, _, a = period_extractor._agree(all_peaks, src_idx=0)
        assert a == "CE"
        assert abs(p - 1.0) < 1e-9

    def test_algorithms_outside_tolerance_fall_back_to_highest_sig(self, period_extractor):
        """Periods with no harmonic relationship do not agree; highest sig wins."""
        peak_ce = _make_peak(period=1.0,  sig=0.7)
        peak_ls = _make_peak(period=1.2,  sig=0.6)  # 20% apart, no harmonic match
        all_peaks = {"CE": [[peak_ce]], "LS": [[peak_ls]]}
        _, _, a = period_extractor._agree(all_peaks, src_idx=0)
        assert a == "CE"  # CE has higher significance

    def test_harmonic_alias_counts_as_agreement(self, period_extractor):
        """An algorithm at 2P agrees with one at P via the h=2 harmonic check."""
        peak_ce  = _make_peak(period=1.0, sig=0.6)
        peak_aov = _make_peak(period=2.0, sig=0.5)  # exactly 2× — h=0.5 matches
        all_peaks = {"CE": [[peak_ce]], "AOV": [[peak_aov]]}
        p, _, a = period_extractor._agree(all_peaks, src_idx=0)
        # Both algorithms agree (harmonic), CE has higher sig
        assert a == "CE"

    def test_sub_cadence_period_rejected(self, period_extractor):
        """Periods below _MIN_AGREE_PERIOD are excluded even if they agree."""
        tiny = _MIN_AGREE_PERIOD * 0.5
        peak_ce  = _make_peak(period=tiny, sig=0.99)
        peak_aov = _make_peak(period=1.0,  sig=0.5)
        all_peaks = {"CE": [[peak_ce]], "AOV": [[peak_aov]]}
        p, _, a = period_extractor._agree(all_peaks, src_idx=0)
        # tiny period filtered → only AOV candidate survives
        assert a == "AOV"
        assert abs(p - 1.0) < 1e-9

    def test_triple_agreement_beats_lone_high_significance(self, period_extractor):
        """Three mutually-agreeing algorithms (count=2 each) beat a lone high-sig outlier.

        A lone algorithm starts at effective best_count=1.  Any algorithm with
        count > 1 (agrees with 2+ others) replaces it regardless of significance.
        """
        # Three algorithms within 5% of each other
        peak_ce  = _make_peak(period=1.000, sig=0.5)
        peak_aov = _make_peak(period=1.010, sig=0.4)  # 1% apart < 5%
        peak_ls  = _make_peak(period=1.015, sig=0.45) # 1.5% apart < 5%
        # Lone high-significance outlier with no harmonic relationship
        peak_mhf = _make_peak(period=5.0,   sig=0.9)

        all_peaks = {
            "CE" : [[peak_ce]],
            "AOV": [[peak_aov]],
            "LS" : [[peak_ls]],
            "MHF": [[peak_mhf]],
        }
        p, _, a = period_extractor._agree(all_peaks, src_idx=0)
        # CE is first in iteration order and has count=2 > best_count=1 → wins
        assert a == "CE"
        assert abs(p - 1.0) < 1e-9


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


def test_extract_period_finding_uses_normalised_mags():
    """algo.calc must receive mags in [0, 1], not raw magnitudes."""
    mock_pf = MagicMock()
    mock_algo = MagicMock()
    mock_algo.calc.return_value = [[]]   # no peaks → nan result
    mock_pf.ConditionalEntropy.return_value = mock_algo

    with patch.dict(sys.modules, {"periodfind": mock_pf}):
        cfg = PeriodConfig(algorithms=["CE"])
        ext = PeriodExtractor(cfg)

        lc = _make_lc(n=20)
        ext.extract([[lc]])

    # Inspect the mags array passed to algo.calc
    assert mock_algo.calc.called
    _, call_args, _ = mock_algo.calc.mock_calls[0]
    mags_passed = call_args[1]   # second positional arg is mags_pf
    for m in mags_passed:
        assert m.min() >= 0.0 - 1e-6
        assert m.max() <= 1.0 + 1e-6


def test_extract_period_finding_uses_zeroed_times():
    """algo.calc must receive times starting near 0, not raw HJD (~2459000)."""
    mock_pf = MagicMock()
    mock_algo = MagicMock()
    mock_algo.calc.return_value = [[]]
    mock_pf.ConditionalEntropy.return_value = mock_algo

    with patch.dict(sys.modules, {"periodfind": mock_pf}):
        cfg = PeriodConfig(algorithms=["CE"])
        ext = PeriodExtractor(cfg)

        lc = _make_lc(n=20)
        ext.extract([[lc]])

    _, call_args, _ = mock_algo.calc.mock_calls[0]
    times_passed = call_args[0]   # first positional arg is times_pf
    for t in times_passed:
        assert t.min() < 1.0   # zeroed: should start at 0
