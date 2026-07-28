"""Tests for ml4em.models.base — the FeatureVector → array contract.

The scalar field list and features_to_array() sit between feature extraction
and every model, so a mistake here is silent: it shows up as a dead input
column or as a crash on the first source that lacks a catalogue match.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ml4em.models.base import (
    N_SCALAR_FEATURES,
    SCALAR_FIELDS,
    features_to_array,
)
from ml4em.types import FeatureVector


def _fv(**kwargs) -> FeatureVector:
    base = dict(source_id="s1", survey="ztf", band="g", ra=10.0, dec=20.0)
    base.update(kwargs)
    return FeatureVector(**base)


def test_scalar_fields_all_exist_on_feature_vector():
    """A name not on FeatureVector would silently become an all-NaN column."""
    declared = {f.name for f in dataclasses.fields(FeatureVector)}
    missing = [f for f in SCALAR_FIELDS if f not in declared]
    assert missing == [], f"SCALAR_FIELDS not on FeatureVector: {missing}"


def test_scalar_fields_have_no_duplicates():
    assert len(SCALAR_FIELDS) == len(set(SCALAR_FIELDS))
    assert N_SCALAR_FEATURES == len(SCALAR_FIELDS)


def test_unmatched_source_yields_nan_not_crash():
    """Gaia fields default to None, not absent — float(None) would raise.

    Every source without a Gaia counterpart hits this path, as does every
    source when catalog.include_gaia is False.
    """
    arr = features_to_array([_fv()])

    assert arr.shape == (1, N_SCALAR_FEATURES)
    assert arr.dtype == np.float32

    idx = SCALAR_FIELDS.index("gaia_parallax")
    assert np.isnan(arr[0, idx])


def test_gaia_values_survive_when_present():
    arr = features_to_array([
        _fv(gaia_parallax=5.0, gaia_g_mean_mag=17.25, gaia_bp_rp=-0.5),
    ])
    for name, expected in [
        ("gaia_parallax", 5.0),
        ("gaia_g_mean_mag", 17.25),
        ("gaia_bp_rp", -0.5),
    ]:
        assert arr[0, SCALAR_FIELDS.index(name)] == pytest.approx(expected)


def test_row_order_matches_input_order():
    arr = features_to_array([_fv(median=1.0), _fv(median=2.0), _fv(median=3.0)])
    col = SCALAR_FIELDS.index("median")
    assert arr[:, col].tolist() == [1.0, 2.0, 3.0]


def test_empty_input_gives_empty_array():
    assert features_to_array([]).size == 0


def test_raw_gaia_magnitudes_are_exposed():
    """The colour is derived and lossy; the raw mags must reach the model.

    G with parallax is what gives absolute magnitude, so dropping it would
    remove the HR-diagram position from every downstream architecture.
    """
    for name in ("gaia_g_mean_mag", "gaia_bp_mean_mag", "gaia_rp_mean_mag"):
        assert name in SCALAR_FIELDS
