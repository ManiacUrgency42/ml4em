"""Tests for ml4em.features.statistics — pure logic, periodfind mocked.

periodfind is a compiled Rust/CUDA extension not present in the test
environment.  All tests intercept the in-function `import periodfind`
via patch.dict(sys.modules) and substitute a MagicMock.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml4em.features.statistics import StatisticsExtractor
from ml4em.types import LightCurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lc(n: int = 20, source_id: str = "src_001", band: str = "r") -> LightCurve:
    rng = np.random.default_rng(42)
    return LightCurve(
        source_id=source_id,
        time=np.linspace(2459000.0, 2459200.0, n),
        mag=rng.normal(18.5, 0.1, n),
        mag_err=np.full(n, 0.05),
        band=band,
        survey="ztf",
        ra=180.0,
        dec=30.0,
    )


def _mock_periodfind(n_valid: int) -> MagicMock:
    """periodfind mock whose BasicStats().calc() returns an (n_valid, 22) array."""
    mock_pf = MagicMock()
    arr = np.ones((n_valid, 22), dtype=np.float32)
    arr[:, 0] = 20.0  # N column — will be cast to int by the extractor
    mock_pf.BasicStats.return_value.calc.return_value = arr
    return mock_pf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_list():
    # extract([]) short-circuits before `import periodfind` — no mock needed.
    assert StatisticsExtractor().extract([]) == []


def test_batch_length_matches_input():
    """Output list length always equals input length."""
    sources = [[_make_lc(n=20)], [_make_lc(n=30, source_id="s2")]]
    with patch.dict(sys.modules, {"periodfind": _mock_periodfind(2)}):
        result = StatisticsExtractor().extract(sources)
    assert len(result) == 2


def test_source_below_min_obs_returns_empty_dict():
    """Source with < 2 observations → empty dict; valid source → feature dict."""
    lc_short = _make_lc(n=1, source_id="short")
    lc_ok    = _make_lc(n=20, source_id="ok")
    sources  = [[lc_short], [lc_ok]]

    with patch.dict(sys.modules, {"periodfind": _mock_periodfind(1)}):
        result = StatisticsExtractor().extract(sources)

    assert result[0] == {}
    assert result[1] != {}


def test_primary_band_uses_most_observed_band():
    """When a source has multiple bands, the band with most observations is used."""
    lc_r = _make_lc(n=30, band="r")
    lc_g = _make_lc(n=5,  band="g")
    sources = [[lc_r, lc_g]]  # r-band has more obs → should be primary

    mock_pf = _mock_periodfind(1)
    with patch.dict(sys.modules, {"periodfind": mock_pf}):
        StatisticsExtractor().extract(sources)

    # First positional arg to calc() is times; verify it matches r-band length.
    call_times = mock_pf.BasicStats.return_value.calc.call_args[0][0]
    assert len(call_times[0]) == 30


def test_output_keys_are_valid_feature_vector_fields():
    """Every key in the returned dict must be a defined FeatureVector field."""
    import dataclasses
    from ml4em.types import FeatureVector
    valid_fields = {f.name for f in dataclasses.fields(FeatureVector)}

    sources = [[_make_lc(n=20)]]
    with patch.dict(sys.modules, {"periodfind": _mock_periodfind(1)}):
        result = StatisticsExtractor().extract(sources)

    for key in result[0]:
        assert key in valid_fields, f"Key '{key}' is not a FeatureVector field"


def test_n_obs_is_integer():
    """n_obs must be returned as int, not float."""
    sources = [[_make_lc(n=20)]]
    with patch.dict(sys.modules, {"periodfind": _mock_periodfind(1)}):
        result = StatisticsExtractor().extract(sources)
    assert isinstance(result[0]["n_obs"], int)


def test_periodfind_exception_does_not_raise():
    """If periodfind.BasicStats raises, all dicts remain empty — no crash."""
    mock_pf = MagicMock()
    mock_pf.BasicStats.return_value.calc.side_effect = RuntimeError("GPU OOM")

    sources = [[_make_lc(n=20)], [_make_lc(n=30, source_id="s2")]]
    with patch.dict(sys.modules, {"periodfind": mock_pf}):
        result = StatisticsExtractor().extract(sources)

    assert result == [{}, {}]


def test_batch_order_preserved_with_skipped_sources():
    """Valid sources at non-contiguous positions map to the correct output index.

    Layout: [skip, ok, skip, ok] → positions 1 and 3 should have feature dicts.
    """
    lc_short = _make_lc(n=1, source_id="skip")
    lc_ok1   = _make_lc(n=20, source_id="ok1")
    lc_ok2   = _make_lc(n=25, source_id="ok2")
    sources  = [[lc_short], [lc_ok1], [lc_short], [lc_ok2]]

    with patch.dict(sys.modules, {"periodfind": _mock_periodfind(2)}):
        result = StatisticsExtractor().extract(sources)

    assert result[0] == {}
    assert result[2] == {}
    assert result[1] != {}
    assert result[3] != {}
