"""Tests for ml4em.features.period — pure Python logic and mocked batch paths.

_unpack_fourier and _agree are pure Python and tested without any mock.
extract() tests use patch.dict(sys.modules) to satisfy `import periodfind`
inside the extractor without requiring the compiled library.

Note on _agree semantics
------------------------
best_count is initialised to 1 (the highest-significance lone algorithm).
An algorithm's count is the number of OTHER algorithms whose peak is within
_AGREE_TOL (2%) of its period.  An algorithm only replaces the current best
when count > best_count, or count == best_count with higher significance.
Consequently, three mutually-agreeing algorithms (count=2 each) can beat a
lone high-significance outlier (effective count=1).
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml4em.config.schema import PeriodConfig
from ml4em.features.period import PeriodExtractor
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

    def test_first_harmonic_phase(self):
        """f1_phi0 = arctan2(b1, a1)."""
        row = _fourier_row(a1=1.0, b1=0.0)
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
        """When two algorithms agree within 2%, the one with higher significance wins."""
        peak_ce  = _make_peak(period=1.00, sig=0.7)
        peak_aov = _make_peak(period=1.01, sig=0.6)  # |1.01-1.00|/1.00 = 1% < 2%
        all_peaks = {"CE": [[peak_ce]], "AOV": [[peak_aov]]}
        p, _, a = period_extractor._agree(all_peaks, src_idx=0)
        assert a == "CE"
        assert abs(p - 1.0) < 1e-9

    def test_algorithms_outside_tolerance_fall_back_to_highest_sig(self, period_extractor):
        """Periods differing by > 2% do not agree; highest significance wins."""
        peak_ce = _make_peak(period=1.0,  sig=0.7)
        peak_ls = _make_peak(period=1.05, sig=0.6)  # 5% apart — outside 2% tolerance
        all_peaks = {"CE": [[peak_ce]], "LS": [[peak_ls]]}
        _, _, a = period_extractor._agree(all_peaks, src_idx=0)
        assert a == "CE"  # CE has higher significance

    def test_triple_agreement_beats_lone_high_significance(self, period_extractor):
        """Three mutually-agreeing algorithms (count=2 each) beat a lone high-sig outlier.

        A lone algorithm starts at effective best_count=1.  Any algorithm with
        count > 1 (agrees with 2+ others) replaces it regardless of significance.
        """
        # Three algorithms within 2% of each other
        peak_ce  = _make_peak(period=1.000, sig=0.5)
        peak_aov = _make_peak(period=1.010, sig=0.4)  # |0.010|/1.0 = 1% < 2%
        peak_ls  = _make_peak(period=1.015, sig=0.45) # |0.015|/1.0 = 1.5% < 2%
        # Lone high-significance outlier
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
