"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float`.)
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import Callable, Tensor
from dxtb.basis import IndexHelper
from dxtb.classical import Repulsion
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular, get_elem_param

from .samples import samples

sample_list = ["H2O", "SiH4", "MB16_43_01", "MB16_43_02", "LYS_xao"]


def gradcheck_param(
    dtype: torch.dtype, name: str
) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.repulsion is not None

    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    # variables to be differentiated
    _arep = get_elem_param(
        torch.unique(numbers),
        par.element,
        "arep",
        pad_val=0,
        **dd,
        requires_grad=True,
    )
    _zeff = get_elem_param(
        torch.unique(numbers),
        par.element,
        "zeff",
        pad_val=0,
        **dd,
        requires_grad=True,
    )
    _kexp = torch.tensor(par.repulsion.effective.kexp, **dd, requires_grad=True)

    def func(arep: Tensor, zeff: Tensor, kexp: Tensor) -> Tensor:
        rep = Repulsion(arep, zeff, kexp, **dd)
        cache = rep.get_cache(numbers, ihelp)
        return rep.get_energy(positions, cache)

    return func, (_arep, _zeff, _kexp)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_param(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradcheck_param(dtype, name)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_gradgrad_param(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    func, diffvars = gradcheck_param(dtype, name)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradgradcheck

    assert gradgradcheck(func, diffvars, atol=tol)
