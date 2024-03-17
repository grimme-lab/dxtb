"""
Run tests for IR spectra.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck
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


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)

    # required for autodiff of energy w.r.t. efield and dipole
    # field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    def f(pos: Tensor) -> Tensor:
        return calc.dipole_analytical(numbers, pos, charge)

    assert dgradcheck(f, positions)


def single(
    name: str,
    field_vector: Tensor,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    atol2: float = 20,
    rtol2: float = 1e-5,
) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    execute(numbers, positions, charge, field_vector, dd, atol, rtol, atol2, rtol2)


def batched(
    name1: str,
    name2: str,
    field_vector: Tensor,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    atol2: float = 20,
    rtol2: float = 1e-5,
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

    execute(numbers, positions, charge, field_vector, dd, atol, rtol, atol2, rtol2)


def execute(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    field_vector: Tensor,
    dd: DD,
    atol: float,
    rtol: float,
    atol2: float,
    rtol2: float,
) -> None:
    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # field is cloned and detached and updated inside
    numfreqs, numints = calc.ir_numerical(numbers, positions, charge)
    assert numfreqs.grad_fn is None
    assert numints.grad_fn is None

    # only add gradient to field_vector after numerical calculation
    field_vector.requires_grad_(True)
    calc.interactions.update_efield(field=field_vector)

    # required for autodiff of energy w.r.t. positions (Hessian)
    pos = positions.clone().detach().requires_grad_(True)

    # manual jacobian
    freqs1, ints1 = calc.ir(numbers, pos, charge, use_functorch=False)
    freqs1, ints1 = tensor_to_numpy(freqs1), tensor_to_numpy(ints1)

    assert pytest.approx(numfreqs, abs=atol, rel=rtol) == freqs1
    assert pytest.approx(numints, abs=atol2, rel=rtol2) == ints1

    # reset (for vibration) before another AD run
    calc.reset()
    pos = positions.clone().detach().requires_grad_(True)

    # jacrev of energy
    freqs2, ints2 = calc.ir(numbers, pos, charge, use_functorch=True)
    freqs2, ints2 = tensor_to_numpy(freqs2), tensor_to_numpy(ints2)

    assert pytest.approx(numfreqs, abs=atol, rel=rtol) == freqs2
    assert pytest.approx(freqs1, abs=atol, rel=rtol) == freqs2
    assert pytest.approx(numints, abs=atol2, rel=rtol2) == ints2
    assert pytest.approx(ints1, abs=atol2, rel=rtol2) == ints2


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
