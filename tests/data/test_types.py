"""Tests for ml4em.types — core data contracts."""

import numpy as np
import pytest

from ml4em.types import LightCurve


def test_lightcurve_valid(sample_lightcurve):
    """A properly constructed LightCurve should not raise."""
    assert sample_lightcurve.n_obs == 10
    assert sample_lightcurve.band == "r"
    assert sample_lightcurve.survey == "ztf"


def test_lightcurve_shape_mismatch_raises():
    """Mismatched array lengths must raise ValueError."""
    with pytest.raises(ValueError, match="identical shapes"):
        LightCurve(
            source_id="bad",
            time=np.array([1.0, 2.0, 3.0]),
            mag=np.array([18.0, 18.1]),     # wrong length
            mag_err=np.array([0.05, 0.05, 0.05]),
            band="r",
            survey="ztf",
            ra=0.0,
            dec=0.0,
        )


def test_lightcurve_2d_array_raises():
    """2-D arrays must raise ValueError."""
    arr = np.ones((3, 2))
    with pytest.raises(ValueError, match="1-D"):
        LightCurve(
            source_id="bad",
            time=arr,
            mag=arr,
            mag_err=arr,
            band="g",
            survey="ztf",
            ra=0.0,
            dec=0.0,
        )
