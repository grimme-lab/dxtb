"""
Testing autodiff for analytical overlap gradient.
"""
from __future__ import annotations

import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from dxtb._types import Callable, Literal, Tensor
from dxtb.basis import Basis, IndexHelper
from dxtb.integral.overlap import overlap_gradient
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular

from .samples import samples

sample_list = ["H2", "HHe", "LiH", "SiH4"]

tol = 1e-7


def gradchecker(
    dtype: torch.dtype, name: str, uplo: Literal["l", "n"]
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(torch.unique(numbers), par, ihelp.unique_angular, **dd)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return overlap_gradient(pos, bas, ihelp, uplo=uplo)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("uplo", ["l", "n"])
def test_grad(dtype: torch.dtype, name: str, uplo: Literal["l", "n"]) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, uplo)
    assert gradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("uplo", ["l", "n"])
def test_gradgrad(dtype: torch.dtype, name: str, uplo: Literal["l", "n"]) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, uplo)
    assert gradgradcheck(func, diffvars, atol=tol)

    diffvars.detach_()
