"""
General tests for the Overlap class.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.integral import Overlap


def test_fail_uplo() -> None:
    dummy = torch.tensor([])
    with pytest.raises(ValueError):
        Overlap(dummy, dummy, dummy, uplo=None)  # type: ignore
