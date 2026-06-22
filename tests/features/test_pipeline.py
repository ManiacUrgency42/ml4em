"""Tests for ml4em.features.pipeline — orchestration logic.

Strategy: mock the individual extractors (not periodfind internals) to keep
tests focused on pipeline behaviour: ordering, chunking, min_observations
filtering, and FeatureVector construction.  periodfind is still patched in
sys.modules because run_batch() calls periodfind.set_device() at the top.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml4em.features.pipeline import FeaturePipeline
from ml4em.types import FeatureVector, LightCurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lc(
    n: int = 60,
    source_id: str = "src_001",
    band: str = "r",
    ra: float = 180.0,
    dec: float = 30.0,
) -> LightCurve:
    return LightCurve(
        source_id=source_id,
        time=np.linspace(2459000.0, 2459300.0, n),
        mag=np.full(n, 18.5),
        mag_err=np.full(n, 0.05),
        band=band,
        survey="ztf",
        ra=ra,
        dec=dec,
    )


def _make_extractor(features: dict | None = None) -> MagicMock:
    """Mock extractor whose extract() mirrors input length.

    Returns one copy of `features` per source so the pipeline can merge them
    into FeatureVectors regardless of batch size.
    """
    if features is None:
        features = {"median": 18.5, "n_obs": 60}
    mock = MagicMock()
    mock.extract.side_effect = lambda sources: [dict(features) for _ in sources]
    return mock


def _make_pipeline(
    min_obs: int = 50,
    features: dict | None = None,
    batch_size: int = 1000,
    compute_dmdt: bool = False,
) -> tuple[FeaturePipeline, MagicMock]:
    extractor = _make_extractor(features)
    pipeline = FeaturePipeline(
        extractors=[extractor],
        min_observations=min_obs,
        compute_dmdt=compute_dmdt,
        device="cpu",
        batch_size=batch_size,
    )
    return pipeline, extractor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_list():
    """run_batch([]) short-circuits before import periodfind."""
    pipeline, _ = _make_pipeline()
    assert pipeline.run_batch([]) == []


def test_output_length_matches_input_length():
    sources = [[_make_lc(source_id=f"s{i}")] for i in range(5)]
    pipeline, _ = _make_pipeline(min_obs=50)
    with patch.dict(sys.modules, {"periodfind": MagicMock()}):
        result = pipeline.run_batch(sources)
    assert len(result) == 5


def test_source_below_min_obs_returns_default_feature_vector():
    """Sources below min_observations skip extraction and get a default FeatureVector."""
    lc_short = _make_lc(n=10, source_id="short")
    pipeline, extractor = _make_pipeline(min_obs=50)
    with patch.dict(sys.modules, {"periodfind": MagicMock()}):
        result = pipeline.run_batch([[lc_short]])

    fv = result[0]
    assert isinstance(fv, FeatureVector)
    assert fv.source_id == "short"
    extractor.extract.assert_not_called()


def test_default_fv_has_nan_scalar_features():
    """Default FeatureVector for skipped sources has NaN for float feature fields."""
    pipeline, _ = _make_pipeline(min_obs=50)
    with patch.dict(sys.modules, {"periodfind": MagicMock()}):
        result = pipeline.run_batch([[_make_lc(n=10, source_id="short")]])

    fv = result[0]
    assert np.isnan(fv.median)
    assert np.isnan(fv.period)
    assert np.isnan(fv.chi2red)


def test_source_above_min_obs_calls_extractor():
    """Valid sources (>= min_observations) trigger extractor.extract()."""
    sources = [[_make_lc(n=60, source_id="ok")]]
    pipeline, extractor = _make_pipeline(min_obs=50)
    with patch.dict(sys.modules, {"periodfind": MagicMock()}):
        pipeline.run_batch(sources)
    extractor.extract.assert_called_once()


def test_extractor_output_appears_in_feature_vector():
    """Fields returned by the extractor are populated in the FeatureVector."""
    features = {"median": 17.3, "n_obs": 60}
    sources = [[_make_lc(n=60, source_id="ok")]]
    pipeline, _ = _make_pipeline(min_obs=50, features=features)
    with patch.dict(sys.modules, {"periodfind": MagicMock()}):
        result = pipeline.run_batch(sources)

    fv = result[0]
    assert abs(fv.median - 17.3) < 1e-6
    assert fv.n_obs == 60


def test_order_preserved_with_mixed_valid_and_skipped_sources():
    """FeatureVectors are returned in the same order as grouped_lcs.

    Layout: [skip, ok, skip, ok] — positions 0 and 2 get default FVs,
    positions 1 and 3 get extracted FVs.
    """
    sources = [
        [_make_lc(n=10,  source_id="skip1")],
        [_make_lc(n=60,  source_id="ok1")],
        [_make_lc(n=5,   source_id="skip2")],
        [_make_lc(n=80,  source_id="ok2")],
    ]
    pipeline, _ = _make_pipeline(min_obs=50)
    with patch.dict(sys.modules, {"periodfind": MagicMock()}):
        result = pipeline.run_batch(sources)

    assert len(result) == 4
    assert result[0].source_id == "skip1"
    assert result[1].source_id == "ok1"
    assert result[2].source_id == "skip2"
    assert result[3].source_id == "ok2"


def test_chunking_returns_all_results_in_order():
    """batch_size < n_sources forces multiple chunks; all results must be present.

    7 sources with batch_size=3 → chunks of [3, 3, 1].  Results must appear
    in the original order across chunk boundaries.
    """
    n = 7
    sources = [[_make_lc(n=60, source_id=f"s{i}")] for i in range(n)]
    extractor = _make_extractor()
    pipeline = FeaturePipeline(
        extractors=[extractor],
        min_observations=50,
        device="cpu",
        batch_size=3,
    )
    with patch.dict(sys.modules, {"periodfind": MagicMock()}):
        result = pipeline.run_batch(sources)

    assert len(result) == n
    for i, fv in enumerate(result):
        assert fv.source_id == f"s{i}"


def test_multi_band_source_uses_primary_band_for_metadata():
    """When a source has multiple bands, the one with most obs drives the FeatureVector
    source_id / survey / ra / dec (consistent with _build_feature_vector internals)."""
    lc_r = _make_lc(n=60, source_id="src", band="r", ra=180.0, dec=30.0)
    lc_g = LightCurve(
        source_id="src",
        time=np.linspace(2459000.0, 2459100.0, 10),
        mag=np.full(10, 18.8),
        mag_err=np.full(10, 0.05),
        band="g",
        survey="ztf",
        ra=180.0,
        dec=30.0,
    )
    sources = [[lc_r, lc_g]]  # r has 60 obs → primary band
    pipeline, _ = _make_pipeline(min_obs=50)
    with patch.dict(sys.modules, {"periodfind": MagicMock()}):
        result = pipeline.run_batch(sources)

    fv = result[0]
    assert fv.source_id == "src"
    assert fv.survey == "ztf"


def test_extractor_not_called_when_all_sources_below_threshold():
    """If every source is below min_observations, extract() is never called."""
    sources = [
        [_make_lc(n=5, source_id="a")],
        [_make_lc(n=3, source_id="b")],
    ]
    pipeline, extractor = _make_pipeline(min_obs=50)
    with patch.dict(sys.modules, {"periodfind": MagicMock()}):
        pipeline.run_batch(sources)
    extractor.extract.assert_not_called()
