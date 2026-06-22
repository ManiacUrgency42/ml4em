"""Integration test: ZTF data layer → feature layer end-to-end.

Requires:
- ML4EM_ZTF_TOKEN environment variable (Kowalski API token)
- Network access to melman.caltech.edu

Run with:
    pytest tests/features/test_integration.py -m integration -v

What this tests
---------------
1. ZTFSource.fetch_batch() returns valid LightCurves (data layer contract).
2. Grouping LightCurves by source_id produces the list[list[LightCurve]] shape
   that FeaturePipeline.run_batch() expects.
3. FeaturePipeline.run_batch() processes the real multi-band batch without error.
4. The output FeatureVectors carry correct source metadata and finite statistics.
5. The dmdt histogram (if computed) has the expected (26, 26) shape.

Source IDs used
---------------
ZTF_sources_84525009 collection on melman.  IDs are confirmed present.
"""

import os
from collections import defaultdict

import numpy as np
import pytest

from ml4em.config.schema import DmdtConfig, FeatureConfig, PeriodConfig, ZTFConfig
from ml4em.constants import N_DM_BINS, N_DT_BINS
from ml4em.data.ztf import ZTFSource
from ml4em.features.pipeline import FeaturePipeline
from ml4em.types import FeatureVector, LightCurve


# Confirmed sources in ZTF_sources_84525009.  Add more here as they are validated.
_ZTF_SOURCE_IDS = [
    "10269362000000",
]

# Fast period config for integration tests — keeps wall-clock time short.
# Note: only pass fields that do NOT involve the algorithms validator, which
# has a Pydantic v2 compatibility issue when algorithms is set explicitly
# (cls._KNOWN is treated as a ModelPrivateAttr descriptor rather than the
# frozenset value).  n_freq_grid alone is enough to speed up the test.
_FAST_PERIOD_CFG = PeriodConfig(n_freq_grid=500)


def _group_by_source(lcs: list[LightCurve]) -> list[list[LightCurve]]:
    """Group a flat list of LightCurves into per-source sublists.

    FeaturePipeline.run_batch() expects list[list[LightCurve]] — one inner list
    per source, each containing all available bands.
    """
    groups: dict[str, list[LightCurve]] = defaultdict(list)
    for lc in lcs:
        groups[lc.source_id].append(lc)
    # Preserve insertion order (Python 3.7+).
    return [groups[sid] for sid in dict.fromkeys(groups)]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ztf_lightcurves_flow_through_feature_pipeline():
    """Fetch a real ZTF source, run it through FeaturePipeline, check output quality.

    This test exercises the full data→feature path:
      ZTFSource.fetch_batch()  →  grouping  →  FeaturePipeline.run_batch()

    The source is a confirmed periodic variable with multi-band ZTF coverage,
    so its FeatureVector should have finite statistics after extraction.
    """
    token = os.environ.get("ML4EM_ZTF_TOKEN", "")
    if not token:
        pytest.skip("ML4EM_ZTF_TOKEN not set")

    # ── 1. Fetch a batch of light curves from the data layer ──────────────
    ztf = ZTFSource(ZTFConfig(), token=token)
    lcs = ztf.fetch_batch(_ZTF_SOURCE_IDS)

    assert len(lcs) > 0, "ZTFSource returned no LightCurves"
    for lc in lcs:
        assert isinstance(lc, LightCurve)
        assert lc.source_id in _ZTF_SOURCE_IDS
        assert lc.band in {"g", "r", "i"}
        assert lc.n_obs > 0

    # ── 2. Group bands per source (data layer → feature layer contract) ───
    grouped = _group_by_source(lcs)

    # fetch_batch was given one unique source_id, so there should be one group.
    assert len(grouped) == len(_ZTF_SOURCE_IDS)

    # Each group must contain at least one LightCurve.
    for group in grouped:
        assert len(group) >= 1

    # ── 3. Run the feature pipeline ───────────────────────────────────────
    cfg = FeatureConfig(
        period=_FAST_PERIOD_CFG,
        dmdt=DmdtConfig(),
        min_observations=10,   # low threshold so real ZTF sources aren't skipped
        compute_dmdt=True,
        device="cpu",
    )
    pipeline = FeaturePipeline.default(cfg)
    fvs = pipeline.run_batch(grouped)

    # ── 4. Structural checks ──────────────────────────────────────────────
    assert len(fvs) == len(grouped), "One FeatureVector expected per source group"

    for src_id, group, fv in zip(_ZTF_SOURCE_IDS, grouped, fvs):
        assert isinstance(fv, FeatureVector)

        # Source metadata must be copied from the primary LightCurve.
        assert fv.source_id == src_id
        assert fv.survey == "ztf"
        assert np.isfinite(fv.ra) and np.isfinite(fv.dec)

        # The source should have enough observations to pass extraction.
        primary_n_obs = max(lc.n_obs for lc in group)
        if primary_n_obs >= cfg.min_observations:
            # Basic statistics must be finite for a real source.
            assert np.isfinite(fv.median), \
                f"median is NaN for source {src_id} with {primary_n_obs} obs"
            assert np.isfinite(fv.chi2red), \
                f"chi2red is NaN for source {src_id}"
            assert fv.n_obs == primary_n_obs

            # dmdt histogram shape.
            if fv.dmdt is not None:
                assert fv.dmdt.shape == (N_DM_BINS, N_DT_BINS), \
                    f"Unexpected dmdt shape: {fv.dmdt.shape}"
