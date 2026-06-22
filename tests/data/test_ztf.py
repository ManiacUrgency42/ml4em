"""Tests for ml4em.data.ztf — pure parsing logic, no network required."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml4em.config.schema import ZTFConfig
from ml4em.data.ztf import ZTFSource, _remove_high_cadence
from ml4em.types import LightCurve


# ---------------------------------------------------------------------------
# Fixture: a ZTFSource with the Kowalski connection mocked out
# ---------------------------------------------------------------------------

@pytest.fixture
def ztf_source() -> ZTFSource:
    """ZTFSource with _connect patched — no real Kowalski connection."""
    with patch.object(ZTFSource, "_connect", return_value=MagicMock()):
        return ZTFSource(ZTFConfig(), token="fake_token")


# ---------------------------------------------------------------------------
# _remove_high_cadence
# ---------------------------------------------------------------------------

def test_cadence_filter_drops_close_observations():
    """Points closer than min_cadence_days to the previous kept point are dropped."""
    t = np.array([0.0, 0.1, 1.0, 1.05, 2.0])
    m = np.ones(5)
    e = np.full(5, 0.01)

    t_out, m_out, e_out = _remove_high_cadence(t, m, e, min_cadence_days=0.5)

    np.testing.assert_array_equal(t_out, [0.0, 1.0, 2.0])
    assert len(m_out) == len(t_out)
    assert len(e_out) == len(t_out)


def test_cadence_filter_keeps_all_when_well_spaced():
    """No points dropped when all gaps exceed min_cadence_days."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    m = np.ones(4)
    e = np.full(4, 0.01)

    t_out, _, _ = _remove_high_cadence(t, m, e, min_cadence_days=0.5)

    np.testing.assert_array_equal(t_out, t)


def test_cadence_filter_empty_input():
    """Empty arrays return empty arrays without error."""
    t = np.array([])
    m = np.array([])
    e = np.array([])

    t_out, m_out, e_out = _remove_high_cadence(t, m, e, min_cadence_days=1.0)

    assert len(t_out) == 0


# ---------------------------------------------------------------------------
# ZTFSource._doc_to_lightcurve  (pure parsing — no network)
# ---------------------------------------------------------------------------

def _make_doc(filter_id=2, ra=180.5, dec=30.2, data=None):
    """Build a minimal Kowalski source document."""
    if data is None:
        data = [
            {"hjd": 2459000.0, "mag": 18.5, "magerr": 0.05, "catflags": 0},
            {"hjd": 2459001.0, "mag": 18.6, "magerr": 0.05, "catflags": 0},
            {"hjd": 2459002.0, "mag": 18.4, "magerr": 0.06, "catflags": 0},
        ]
    return {"_id": 1234567890, "filter": filter_id, "ra": ra, "dec": dec, "data": data}


def test_doc_to_lightcurve_valid(ztf_source):
    """A well-formed document with clean data produces a LightCurve."""
    lc = ztf_source._doc_to_lightcurve(_make_doc())

    assert isinstance(lc, LightCurve)
    assert lc.source_id == "1234567890"
    assert lc.band == "r"        # filter_id=2 maps to r
    assert lc.survey == "ztf"
    assert lc.n_obs == 3


def test_doc_to_lightcurve_drops_flagged_epochs(ztf_source):
    """Epochs with catflags != 0 are excluded from the returned LightCurve."""
    data = [
        {"hjd": 2459000.0, "mag": 18.5, "magerr": 0.05, "catflags": 0},
        {"hjd": 2459001.0, "mag": 18.6, "magerr": 0.05, "catflags": 1},  # flagged
        {"hjd": 2459002.0, "mag": 18.4, "magerr": 0.06, "catflags": 0},
    ]
    lc = ztf_source._doc_to_lightcurve(_make_doc(data=data))

    assert lc is not None
    assert lc.n_obs == 2


def test_doc_to_lightcurve_all_flagged_returns_none(ztf_source):
    """A document where every epoch is flagged should return None."""
    data = [
        {"hjd": 2459000.0, "mag": 18.5, "magerr": 0.05, "catflags": 4},
        {"hjd": 2459001.0, "mag": 18.6, "magerr": 0.05, "catflags": 1},
    ]
    lc = ztf_source._doc_to_lightcurve(_make_doc(data=data))

    assert lc is None


def test_doc_to_lightcurve_unknown_band_returns_none(ztf_source):
    """An unrecognised filter_id (not 1/2/3) returns None."""
    lc = ztf_source._doc_to_lightcurve(_make_doc(filter_id=99))
    assert lc is None


# ---------------------------------------------------------------------------
# ZTFSource._parse_responses  (pure parsing — no network)
# ---------------------------------------------------------------------------

def test_parse_responses_returns_lightcurves(ztf_source):
    """A successful Kowalski response is parsed into LightCurve objects."""
    responses = {
        "kowalski": [
            {
                "status": "success",
                "data": [_make_doc(filter_id=1), _make_doc(filter_id=2)],
            }
        ]
    }
    lcs = ztf_source._parse_responses(responses)

    assert len(lcs) == 2
    bands = {lc.band for lc in lcs}
    assert bands == {"g", "r"}


def test_parse_responses_skips_failed_status(ztf_source):
    """Response entries with status != 'success' are silently skipped."""
    responses = {
        "kowalski": [
            {"status": "error", "data": [_make_doc()]},
            {"status": "success", "data": [_make_doc(filter_id=3)]},
        ]
    }
    lcs = ztf_source._parse_responses(responses)

    assert len(lcs) == 1
    assert lcs[0].band == "i"


# ---------------------------------------------------------------------------
# Integration test — hits real Kowalski API, requires ML4EM_ZTF_TOKEN
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ztf_fetch_real_lightcurves():
    """Fetch a real ZTF source from Kowalski and confirm we get LightCurves back."""
    token = os.environ.get("ML4EM_ZTF_TOKEN", "")
    if not token:
        pytest.skip("ML4EM_ZTF_TOKEN not set")

    # Real source ID confirmed present in ZTF_sources_84525009 on melman
    source_id = "10269362000000"

    source = ZTFSource(ZTFConfig(), token=token)
    lcs = source.fetch_batch([source_id])

    assert len(lcs) > 0, "Expected at least one LightCurve back from Kowalski"
    for lc in lcs:
        assert isinstance(lc, LightCurve)
        assert lc.source_id == source_id
        assert lc.survey == "ztf"
        assert lc.n_obs > 0
        assert lc.band in {"g", "r", "i"}
