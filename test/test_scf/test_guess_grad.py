"""
Testing SCF gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from dxtb._types import Callable, Tensor
from dxtb.basis import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.scf import guess
from dxtb.utils import batch

from .samples import samples

sample_list = ["H2", "HHe", "LiH", "H2O", "SiH4"]

tol = 1e-7


def gradchecker(
    dtype: torch.dtype, name: str, guess_name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charge = torch.tensor(0.0, **dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return guess.get_guess(numbers, pos, charge, ihelp, name=guess_name)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("guess_name", ["eeq", "sad"])
def test_grad(dtype: torch.dtype, name: str, guess_name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, guess_name)
    assert gradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("guess_name", ["eeq", "sad"])
def test_gradgrad(dtype: torch.dtype, name: str, guess_name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, guess_name)
    assert gradgradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str, guess_name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
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
    charge = torch.tensor([0.0, 0.0], **dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return guess.get_guess(numbers, pos, charge, ihelp, name=guess_name)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("guess_name", ["eeq", "sad"])
def test_grad_batch(
    dtype: torch.dtype, name1: str, name2: str, guess_name: str
) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, guess_name)
    assert gradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("guess_name", ["eeq", "sad"])
def test_gradgrad_batch(
    dtype: torch.dtype, name1: str, name2: str, guess_name: str
) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, guess_name)
    assert gradgradcheck(func, diffvars, atol=tol)

    diffvars.detach_()
