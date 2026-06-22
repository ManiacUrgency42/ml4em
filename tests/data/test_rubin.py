"""Tests for ml4em.data.rubin.

RubinSource is currently a stub pending DP1 schema confirmation.
These tests document the stub status and will be replaced with real
parsing tests once fetch_batch is implemented.
"""

import pytest

from ml4em.config.schema import RubinConfig
from ml4em.data.rubin import RubinSource


@pytest.fixture
def rubin_source() -> RubinSource:
    return RubinSource(RubinConfig(), token="fake_token")


def test_fetch_batch_not_implemented(rubin_source):
    """fetch_batch raises NotImplementedError until DP1 integration is complete."""
    with pytest.raises(NotImplementedError):
        rubin_source.fetch_batch(["123456789"])
