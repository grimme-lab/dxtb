"""
Testing overlap gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from dxtb._types import Callable, Tensor
from dxtb.integral import mmd

tol = 1e-7


def gradchecker(
    dtype: torch.dtype, mmd_func, angular: tuple[Tensor, Tensor]
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    alpha = (
        torch.tensor(
            [10.256286, 0.622797, 0.239101, 7.611997, 1.392902, 0.386963, 0.128430],
            dtype=dtype,
        ),
        torch.tensor(
            [1.723363, 0.449418, 0.160806, 0.067220, 0.030738, 0.014532],
            dtype=dtype,
        ),
    )
    coeff = (
        torch.tensor(
            [-1.318654, 1.603878, 0.601323, -0.980904, -1.257964, -0.985990, -0.235962],
            dtype=dtype,
        ),
        torch.tensor(
            [0.022303, 0.026981, 0.027555, 0.019758, 0.007361, 0.000756],
            dtype=dtype,
        ),
    )

    # variables to be differentiated
    vec = torch.tensor(
        [[-0.000000, -0.000000, -3.015935]],
        dtype=dtype,
        requires_grad=True,
    )

    def func(v: Tensor) -> Tensor:
        return mmd_func(angular, alpha, coeff, v)

    return func, vec


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize(
    "mmd_func", [mmd.explicit.mmd_explicit, mmd.recursion.mmd_recursion]
)
@pytest.mark.parametrize("li", [0, 1, 2, 3])
@pytest.mark.parametrize("lj", [0, 1, 2, 3])
def test_grad(dtype: torch.dtype, mmd_func, li: int, lj: int) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    angular = (torch.tensor(li), torch.tensor(lj))
    func, diffvars = gradchecker(dtype, mmd_func, angular)
    assert gradcheck(func, diffvars, atol=tol)


# FIXME: Recursive version fails
@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("mmd_func", [mmd.explicit.mmd_explicit])
@pytest.mark.parametrize("li", [0, 1, 2, 3])
@pytest.mark.parametrize("lj", [0, 1, 2, 3])
def test_gradgrad(dtype: torch.dtype, mmd_func, li: int, lj: int) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    angular = (torch.tensor(li), torch.tensor(lj))
    func, diffvars = gradchecker(dtype, mmd_func, angular)
    assert gradgradcheck(func, diffvars, atol=tol)
