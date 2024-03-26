"""
Testing `InteractionList` gradient (autodiff).
"""

from __future__ import annotations

import pytest
import torch

from dxtb._types import DD, Callable, Tensor
from dxtb.basis import IndexHelper
from dxtb.components.interactions import InteractionList
from dxtb.components.interactions.coulomb import new_es2, new_es3
from dxtb.param import GFN1_XTB as par
from dxtb.scf import get_guess
from dxtb.utils import batch

from ..utils import dgradcheck, dgradgradcheck
from .samples import samples

sample_list = ["H2", "HHe", "LiH", "H2O", "SiH4"]

tol = 1e-7

device = None


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    chrg = torch.tensor(0.0, **dd)

    # setup
    ihelp = IndexHelper.from_numbers(numbers, par)
    ilist = InteractionList(new_es2(numbers, par, **dd), new_es3(numbers, par, **dd))

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        icaches = ilist.get_cache(numbers=numbers, positions=pos, ihelp=ihelp)
        charges = get_guess(numbers, positions, chrg, ihelp)
        return ilist.get_energy(charges, icaches, ihelp)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
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
    chrg = torch.tensor([0.0, 0.0], **dd)

    # setup
    ihelp = IndexHelper.from_numbers(numbers, par)
    ilist = InteractionList(new_es2(numbers, par, **dd), new_es3(numbers, par, **dd))

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        icaches = ilist.get_cache(numbers=numbers, positions=pos, ihelp=ihelp)
        charges = get_guess(numbers, positions, chrg, ihelp)
        return ilist.get_energy(charges, icaches, ihelp)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
def test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol)
