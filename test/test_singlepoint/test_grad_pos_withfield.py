"""
Run tests for IR spectra.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD, Callable, Tensor
from dxtb.constants import units
from dxtb.interaction import new_efield
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from ..utils import dgradcheck, dgradgradcheck
from .samples import samples

opts = {
    "maxiter": 100,
    "xitorch_fatol": 1.0e-8,
    "xitorch_xatol": 1.0e-8,
    "verbosity": 0,
    "scf_mode": "full",
    "mixer": "anderson",
}

tol = 1e-4

sample_list = ["H2", "H2O", "SiH4"]

device = None


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": device, "dtype": dtype}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * units.VAA2AU

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        result = calc.singlepoint(numbers, pos, charge)
        energy = result.total.sum(-1)
        return energy

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
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-9)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
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
    charge = torch.tensor([0.0, 0.0], **dd)

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * units.VAA2AU

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        result = calc.singlepoint(numbers, pos, charge)
        energy = result.total.sum(-1)
        return energy

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-8)
