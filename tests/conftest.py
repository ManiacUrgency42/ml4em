"""Shared pytest fixtures for ml4em tests."""

from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv

from ml4em.types import LightCurve

# Load .env from the project root so ML4EM_ZTF_TOKEN etc. are available
load_dotenv(Path(__file__).parent.parent / ".env")


@pytest.fixture
def sample_lightcurve() -> LightCurve:
    """A minimal valid LightCurve for use across test modules."""
    n = 10
    return LightCurve(
        source_id="test_001",
        time=np.linspace(2459000.0, 2459100.0, n),
        mag=np.full(n, 18.5),
        mag_err=np.full(n, 0.05),
        band="r",
        survey="ztf",
        ra=180.0,
        dec=30.0,
    )
