"""
Testing automatic energy gradient w.r.t. electric field vector.
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
    "f_atol": 1.0e-8,
    "x_atol": 1.0e-8,
    "verbosity": 0,
    "scf_mode": "full",
    "mixer": "anderson",
}

tol = 1e-4

# FIXME: There seem to be nultiple issues with this gradient here.
# - SiH4 fails for 0.0 (0.01 check depends on eps)
# - "ValueError: grad requires non-empty inputs." for xitorch
# - non-negligible differences between --fast and --slow
sample_list = ["H2", "H2O"]
xfields = [0.0, 1.0, -2.0]

device = None


def gradchecker(
    dtype: torch.dtype, name: str, xfield: float
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": device, "dtype": dtype}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    field_vector = torch.tensor([xfield, 0.0, 0.0], **dd) * units.VAA2AU

    # variables to be differentiated
    field_vector.requires_grad_(True)

    def func(field_vector: Tensor) -> Tensor:
        efield = new_efield(field_vector)
        calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)
        result = calc.singlepoint(numbers, positions, charge)
        energy = result.total.sum(-1)
        return energy

    return func, field_vector


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("xfield", xfields)
def test_gradcheck(dtype: torch.dtype, name: str, xfield: float) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, xfield)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("xfield", xfields)
def test_gradgradcheck(dtype: torch.dtype, name: str, xfield: float) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, xfield)
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-9)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str, xfield: float
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

    # variables to be differentiated
    field_vector = torch.tensor([xfield, 0.0, 0.0], **dd) * units.VAA2AU
    field_vector.requires_grad_(True)

    def func(field_vector: Tensor) -> Tensor:
        efield = new_efield(field_vector)
        calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)
        result = calc.singlepoint(numbers, positions, charge)
        energy = result.total.sum(-1)
        return energy

    return func, field_vector


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("xfield", xfields)
def test_gradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, xfield: float
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, xfield)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("xfield", xfields)
def test_gradgradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, xfield: float
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, xfield)
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-8)
