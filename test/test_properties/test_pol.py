"""
Run tests for polarizability.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.typing import DD, Tensor

from dxtb.constants import units
from dxtb.interaction import new_efield
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

slist = ["H", "LiH", "HHe", "H2O", "CH4", "SiH4", "PbH4-BiH3"]
slist_large = ["LYS_xao", "MB16_43_01"]

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
    atol: float = 1e-5,
    rtol: float = 1e-5,
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
    atol: float = 1e-5,
    rtol: float = 1e-5,
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
    # required for autodiff of energy w.r.t. efield
    field_vector.requires_grad_(True)

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # field is cloned and detached and updated inside
    num = calc.polarizability_numerical(numbers, positions, charge)

    # manual jacobian
    pol = tensor_to_numpy(
        calc.polarizability(
            numbers,
            positions,
            charge,
            use_functorch=False,
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == pol

    # 2x jacrev of energy
    pol2 = tensor_to_numpy(
        calc.polarizability(
            numbers,
            positions,
            charge,
            use_functorch=True,
            derived_quantity="energy",
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == pol2

    # applying jacrev twice requires detaching
    calc.interactions.reset_efield()

    # jacrev of dipole
    pol3 = tensor_to_numpy(
        calc.polarizability(
            numbers,
            positions,
            charge,
            use_functorch=True,
            derived_quantity="dipole",
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == pol3

    assert pytest.approx(pol, abs=1e-8) == pol2
    assert pytest.approx(pol, abs=1e-8) == pol3


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


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU
    batched(name1, name2, field_vector, dd=dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist_large)
def test_batch_large(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU
    batched(name1, name2, field_vector, dd=dd)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def test_batch_field(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * units.VAA2AU
    batched(name1, name2, field_vector, dd=dd)


# FIXME: charges are unstable (maybe fixed by implicit diff?)
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "LiH", "CH4"])
@pytest.mark.parametrize("scp_mode", ["potential", "fock"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_batch_settings(
    dtype: torch.dtype, name1: str, name2: str, scp_mode: str, mixer: str
) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    atol, rtol = 1e-5, 1e-5

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

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU
    field_vector.requires_grad_(True)

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)

    options = dict(opts, **{"scp_mode": scp_mode, "mixer": mixer})
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    # field is cloned and detached and updated inside
    num = calc.polarizability_numerical(numbers, positions, charge)

    # manual jacobian
    pol = tensor_to_numpy(
        calc.polarizability(
            numbers,
            positions,
            charge,
            use_functorch=False,
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == pol

    # 2x jacrev of energy
    pol2 = tensor_to_numpy(
        calc.polarizability(
            numbers,
            positions,
            charge,
            use_functorch=True,
            derived_quantity="energy",
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == pol2

    # applying jacrev twice requires detaching
    calc.interactions.reset_efield()

    # jacrev of dipole
    pol3 = tensor_to_numpy(
        calc.polarizability(
            numbers,
            positions,
            charge,
            use_functorch=True,
            derived_quantity="dipole",
        )
    )
    assert pytest.approx(num, abs=atol, rel=rtol) == pol3

    assert pytest.approx(pol, abs=1e-8) == pol2
    assert pytest.approx(pol, abs=1e-8) == pol3