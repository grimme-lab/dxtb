"""
Testing halogen bond correction gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from dxtb._types import Callable, Tensor
from dxtb.basis import IndexHelper
from dxtb.classical import Halogen
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular, get_elem_param
from dxtb.utils import batch

from .samples import samples

sample_list = ["br2nh3", "br2och2", "tmpda"]

tol = 1e-8


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.halogen is not None

    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    # variables to be differentiated
    _damp = torch.tensor(
        par.halogen.classical.damping,
        **dd,
        requires_grad=True,
    )
    _rscale = torch.tensor(
        par.halogen.classical.rscale,
        **dd,
        requires_grad=True,
    )
    _xbond = get_elem_param(
        torch.unique(numbers),
        par.element,
        "xbond",
        pad_val=0,
        **dd,
        requires_grad=True,
    )

    def func(damp: Tensor, rscale: Tensor, xbond: Tensor) -> Tensor:
        xb = Halogen(damp, rscale, xbond, **dd)
        cache = xb.get_cache(numbers, ihelp)
        return xb.get_energy(positions, cache)

    return func, (_damp, _rscale, _xbond)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert gradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert gradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.halogen is not None

    dd = {"dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        [
            sample1["numbers"],
            sample2["numbers"],
        ]
    )
    positions = batch.pack(
        [
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        ]
    )
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    # variables to be differentiated
    _damp = torch.tensor(
        par.halogen.classical.damping,
        **dd,
        requires_grad=True,
    )
    _rscale = torch.tensor(
        par.halogen.classical.rscale,
        **dd,
        requires_grad=True,
    )
    _xbond = get_elem_param(
        torch.unique(numbers),
        par.element,
        "xbond",
        pad_val=0,
        **dd,
        requires_grad=True,
    )

    def func(damp: Tensor, rscale: Tensor, xbond: Tensor) -> Tensor:
        xb = Halogen(damp, rscale, xbond, **dd)
        cache = xb.get_cache(numbers, ihelp)
        return xb.get_energy(positions, cache)

    return func, (_damp, _rscale, _xbond)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert gradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert gradgradcheck(func, diffvars, atol=tol)
