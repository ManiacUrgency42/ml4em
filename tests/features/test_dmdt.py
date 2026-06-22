"""Tests for ml4em.features.dmdt — pure logic, periodfind mocked.

periodfind.DmDt is the only external call; it is replaced by a MagicMock
whose .calc() returns a fixed-shape ndarray.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml4em.config.schema import DmdtConfig
from ml4em.constants import N_DM_BINS, N_DT_BINS
from ml4em.features.dmdt import DmdtExtractor
from ml4em.types import LightCurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lc(n: int = 20, source_id: str = "src_001") -> LightCurve:
    return LightCurve(
        source_id=source_id,
        time=np.linspace(2459000.0, 2459200.0, n),
        mag=np.full(n, 18.5),
        mag_err=np.full(n, 0.05),
        band="r",
        survey="ztf",
        ra=180.0,
        dec=30.0,
    )


def _mock_periodfind(n_valid: int) -> MagicMock:
    """periodfind mock whose DmDt().calc() returns (n_valid, N_DM_BINS, N_DT_BINS)."""
    mock_pf = MagicMock()
    mock_pf.DmDt.return_value.calc.return_value = np.zeros(
        (n_valid, N_DM_BINS, N_DT_BINS), dtype=np.float32
    )
    return mock_pf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_list():
    # extract([]) short-circuits before `import periodfind`.
    assert DmdtExtractor(DmdtConfig()).extract([]) == []


def test_batch_length_matches_input():
    """Output list length always equals input length."""
    sources = [[_make_lc(n=20)], [_make_lc(n=30, source_id="s2")]]
    with patch.dict(sys.modules, {"periodfind": _mock_periodfind(2)}):
        result = DmdtExtractor(DmdtConfig()).extract(sources)
    assert len(result) == 2


def test_source_below_min_obs_returns_empty_dict():
    """Source with < 2 observations returns {} (no 'dmdt' key)."""
    lc_short = _make_lc(n=1, source_id="short")
    lc_ok    = _make_lc(n=20, source_id="ok")
    sources  = [[lc_short], [lc_ok]]

    with patch.dict(sys.modules, {"periodfind": _mock_periodfind(1)}):
        result = DmdtExtractor(DmdtConfig()).extract(sources)

    assert result[0] == {}
    assert "dmdt" in result[1]


def test_dmdt_output_shape():
    """Output histogram must have shape (N_DM_BINS, N_DT_BINS) = (26, 26)."""
    sources = [[_make_lc(n=20)]]
    with patch.dict(sys.modules, {"periodfind": _mock_periodfind(1)}):
        result = DmdtExtractor(DmdtConfig()).extract(sources)
    assert result[0]["dmdt"].shape == (N_DM_BINS, N_DT_BINS)


def test_periodfind_exception_does_not_raise():
    """If periodfind.DmDt raises, all dicts remain empty — no crash."""
    mock_pf = MagicMock()
    mock_pf.DmDt.return_value.calc.side_effect = RuntimeError("OOM")

    sources = [[_make_lc(n=20)], [_make_lc(n=30, source_id="s2")]]
    with patch.dict(sys.modules, {"periodfind": mock_pf}):
        result = DmdtExtractor(DmdtConfig()).extract(sources)

    assert result == [{}, {}]


def test_batch_order_preserved_with_skipped_sources():
    """Valid/skipped positions map correctly in the returned list.

    Layout: [skip, ok, skip] → index 1 has dmdt, 0 and 2 are empty.
    """
    lc_short = _make_lc(n=1, source_id="skip")
    lc_ok    = _make_lc(n=20, source_id="ok")
    sources  = [[lc_short], [lc_ok], [lc_short]]

    with patch.dict(sys.modules, {"periodfind": _mock_periodfind(1)}):
        result = DmdtExtractor(DmdtConfig()).extract(sources)

    assert result[0] == {}
    assert "dmdt" in result[1]
    assert result[2] == {}
