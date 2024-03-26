"""
Run tests for geometric polarizability derivative.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.typing import DD, Tensor

from dxtb.components.interactions import new_efield
from dxtb.constants import units
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

slist = ["H", "LiH", "HHe", "H2O", "CH4", "PbH4-BiH3"]
slist_large = ["MB16_43_01"]

opts = {
    "int_level": 3,
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "f_atol": 1e-10,
    "x_atol": 1e-10,
}

device = None


def single(
    name: str,
    field_vector: Tensor,
    dd: DD,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    execute(numbers, positions, charge, field_vector, dd, atol, rtol)


def batched(
    name1: str,
    name2: str,
    field_vector: Tensor,
    dd: DD,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ],
    )
    positions = batch.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ],
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    execute(numbers, positions, charge, field_vector, dd, atol, rtol)


def execute(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    field_vector: Tensor,
    dd: DD,
    atol: float,
    rtol: float,
) -> None:
    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # field is cloned and detached and updated inside
    num = calc.dipole_deriv_numerical(numbers, positions, charge)

    # required for autodiff of energy w.r.t. efield
    calc.interactions.update_efield(field=field_vector.requires_grad_(True))

    # manual jacobian with analytical dipole derivative
    dipder1 = tensor_to_numpy(
        calc.dipole_deriv(
            numbers,
            positions.detach().clone().requires_grad_(True),
            charge,
            use_analytical=True,
            use_functorch=False,
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == dipder1

    # applying AD twice requires detaching
    calc.reset()

    # manual jacobian with AD dipole moment
    dipder2 = tensor_to_numpy(
        calc.dipole_deriv(
            numbers,
            positions.detach().clone().requires_grad_(True),
            charge,
            use_analytical=False,
            use_functorch=False,
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == dipder2

    # applying AD twice requires detaching
    calc.reset()

    # jacrev of analytical dipole moment
    dipder3 = tensor_to_numpy(
        calc.dipole_deriv(
            numbers,
            positions.detach().clone().requires_grad_(True),
            charge,
            use_analytical=True,
            use_functorch=True,
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == dipder3

    # applying AD twice requires detaching
    calc.reset()

    # jacrev of AD dipole moment
    dipder4 = tensor_to_numpy(
        calc.dipole_deriv(
            numbers,
            positions.detach().clone().requires_grad_(True),
            charge,
            use_analytical=False,
            use_functorch=True,
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == dipder4

    assert pytest.approx(dipder1, abs=3e-7) == dipder2
    assert pytest.approx(dipder1, abs=3e-7) == dipder3
    assert pytest.approx(dipder1, abs=3e-7) == dipder4


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)
    single(name, field_vector, dd=dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)
    single(name, field_vector, dd=dd)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single_field(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * units.VAA2AU
    single(name, field_vector, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU
    batched(name1, name2, field_vector, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist_large)
def skip_test_batch_large(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU
    batched(name1, name2, field_vector, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch_field(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * units.VAA2AU
    batched(name1, name2, field_vector, dd=dd)
