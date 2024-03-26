"""
Testing halogen bond correction gradient (autodiff).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.typing import DD, Callable, Tensor

from dxtb.basis import IndexHelper
from dxtb.components.classicals import Halogen
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_param
from dxtb.utils import batch

from ..utils import dgradcheck, dgradgradcheck
from .samples import samples

sample_list = ["br2nh3", "br2och2", "tmpda"]

tol = 1e-8

device = None


def gradchecker(dtype: torch.dtype, name: str) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.halogen is not None

    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

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
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(dtype: torch.dtype, name1: str, name2: str) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.halogen is not None

    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = batch.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    ihelp = IndexHelper.from_numbers(numbers, par)

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
    assert dgradcheck(func, diffvars, atol=tol)


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
    assert dgradgradcheck(func, diffvars, atol=tol)
