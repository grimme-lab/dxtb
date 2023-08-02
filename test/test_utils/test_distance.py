"""
Test distance calculations.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.utils import geometry


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all(dtype: torch.dtype) -> None:
    x = torch.randn(2, 3, 4, dtype=dtype)

    d1 = geometry.cdist(x)
    d2 = geometry.cdist_direct_expansion(x, x, p=2)
    d3 = geometry.euclidean_dist_quadratic_expansion(x, x)

    assert pytest.approx(d1) == d2
    assert pytest.approx(d2) == d3
    assert pytest.approx(d3) == d1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p", [2, 3, 4, 5])
def test_ps(dtype: torch.dtype, p: int) -> None:
    x = torch.randn(2, 4, 5, dtype=dtype)
    y = torch.randn(2, 4, 5, dtype=dtype)

    d1 = geometry.cdist(x, y, p=p)
    d2 = torch.cdist(x, y, p=p)

    assert pytest.approx(d1) == d2
