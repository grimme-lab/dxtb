"""
Testing halogen bond correction gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from dxtb._types import Callable, Tensor
from dxtb.basis import IndexHelper
from dxtb.classical import new_halogen
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch

from .samples import samples

# "LYS_xao" must be the last one as we have to manually exclude it for the
# `backward` and `gradgradcheck` check because the gradient is zero
sample_list = ["br2nh3", "br2och2", "LYS_xao"]

tol = 1e-8


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)

    def func(pos: Tensor) -> Tensor:
        return xb.get_energy(pos, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert gradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list[:-1])
def test_gradgradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert gradgradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
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

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)

    def func(pos: Tensor) -> Tensor:
        return xb.get_energy(pos, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert gradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert gradgradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["br2nh3", "finch", "LYS_xao"])
def test_autograd(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["gradient"].type(dtype)

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)

    energy = xb.get_energy(positions, cache)
    grad_autograd = xb.get_gradient(energy, positions)

    assert pytest.approx(ref, abs=tol * 10) == grad_autograd.detach()

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_autograd_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
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
    ref = batch.pack(
        [
            sample1["gradient"].type(dtype),
            sample2["gradient"].type(dtype),
        ]
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)

    energy = xb.get_energy(positions, cache)
    grad_autograd = xb.get_gradient(energy, positions)

    assert pytest.approx(ref, abs=tol * 10) == grad_autograd.detach()

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list[:-1])
def test_backward(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["gradient"].type(dtype)

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)

    energy = xb.get_energy(positions, cache)
    energy.sum().backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    assert pytest.approx(ref, abs=tol * 10) == grad_backward

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_backward_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Compare with reference values from tblite."""
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
    ref = batch.pack(
        [
            sample1["gradient"].type(dtype),
            sample2["gradient"].type(dtype),
        ]
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)

    energy = xb.get_energy(positions, cache)
    energy.sum().backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    assert pytest.approx(ref, abs=tol * 10) == grad_backward

    positions.detach_()
