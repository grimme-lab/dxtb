"""
General tests for the Overlap class.
"""

from __future__ import annotations

import pytest

from dxtb.integral.driver.pytorch import OverlapPytorch as Overlap


def test_fail_uplo() -> None:
    with pytest.raises(ValueError):
        Overlap(uplo=None)  # type: ignore
