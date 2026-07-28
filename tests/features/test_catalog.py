"""Tests for ml4em.features.catalog — Gaia cross-match parsing.

No Kowalski client is used; the two dict-producing helpers are exercised
directly, since the failure modes worth catching are in key naming and in
the match / no-match paths disagreeing about the feature surface.
"""

from __future__ import annotations

import dataclasses

import pytest

from ml4em.features.catalog import CatalogExtractor
from ml4em.types import FeatureVector

_GAIA_DOC = {
    "parallax": 5.0,
    "parallax_error": 0.1,
    "phot_g_mean_mag": 17.2,
    "phot_bp_mean_mag": 17.0,
    "phot_rp_mean_mag": 17.5,
    "astrometric_excess_noise": 0.3,
}


def _parse(doc: dict) -> dict:
    # __new__ avoids needing a config or a Kowalski client for a pure parse.
    return CatalogExtractor._parse_gaia_doc(
        CatalogExtractor.__new__(CatalogExtractor), doc
    )


def test_match_and_no_match_produce_identical_keys():
    """Otherwise a matched and an unmatched source disagree on feature shape."""
    assert set(_parse(_GAIA_DOC)) == set(CatalogExtractor._no_match())


def test_emitted_keys_are_real_feature_vector_fields():
    """A typo here is invisible: FeatureVector(**features) would raise, but a
    silently-renamed field would leave the real one at its default forever."""
    declared = {f.name for f in dataclasses.fields(FeatureVector)}
    assert set(_parse(_GAIA_DOC)) <= declared


def test_colour_is_derived_from_the_raw_magnitudes():
    out = _parse(_GAIA_DOC)
    assert out["gaia_bp_rp"] == pytest.approx(17.0 - 17.5)


def test_raw_magnitudes_are_retained_alongside_the_colour():
    """The colour alone cannot be inverted back into the two magnitudes."""
    out = _parse(_GAIA_DOC)
    assert out["gaia_g_mean_mag"] == pytest.approx(17.2)
    assert out["gaia_bp_mean_mag"] == pytest.approx(17.0)
    assert out["gaia_rp_mean_mag"] == pytest.approx(17.5)


def test_missing_photometry_gives_none_colour_not_a_crash():
    out = _parse({"parallax": 5.0})
    assert out["gaia_bp_rp"] is None
    assert out["gaia_parallax"] == pytest.approx(5.0)


def test_parsed_features_construct_a_feature_vector():
    fv = FeatureVector(
        source_id="s1", survey="ztf", band="g", ra=10.0, dec=20.0,
        **_parse(_GAIA_DOC),
    )
    assert fv.gaia_g_mean_mag == pytest.approx(17.2)
