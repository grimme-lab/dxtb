"""
Run tests for IR spectra.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.constants import units
from dxtb.interaction import new_efield
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

opts = {
    "maxiter": 100,
    "f_atol": 1.0e-8,
    "x_atol": 1.0e-8,
    "verbosity": 0,
    "scf_mode": "full",
    "mixer": "anderson",
}

device = None


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", samples.keys())
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    ref = samples[name]["dipole"].to(**dd)

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)  # * units.VAA2AU

    # required for autodiff of energy w.r.t. efield and dipole
    field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    dipole = calc.dipole(numbers, positions, charge, False)
    dipole.detach_()

    assert pytest.approx(ref, abs=1e-3) == dipole


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", samples.keys())
def test_single_field(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    ref = samples[name]["dipole2"].to(**dd)

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * units.VAA2AU

    # required for autodiff of energy w.r.t. efield and dipole
    field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    dipole = calc.dipole(numbers, positions, charge)
    dipole.detach_()

    assert pytest.approx(ref, abs=1e-3, rel=1e-3) == dipole


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", samples.keys())
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

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

    ref = batch.pack(
        [
            sample1["dipole"].to(**dd),
            sample2["dipole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU

    # required for autodiff of energy w.r.t. efield and dipole
    field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    dipole = calc.dipole(numbers, positions, charge)
    dipole.detach_()

    assert pytest.approx(ref, abs=1e-3) == dipole


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "LiH", "H2O"])
@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_batch_settings(
    dtype: torch.dtype, name1: str, name2: str, scp_mode: str, mixer: str
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

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

    ref = batch.pack(
        [
            sample1["dipole"].to(**dd),
            sample2["dipole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU

    # required for autodiff of energy w.r.t. efield and dipole
    field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    efield = new_efield(field_vector)
    options = dict(opts, **{"scp_mode": scp_mode, "mixer": mixer})
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    dipole = calc.dipole(numbers, positions, charge)
    dipole.detach_()

    assert pytest.approx(ref, abs=1e-4) == dipole


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "LiH", "H2O"])
def test_batch_unconverged(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

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

    ref = batch.pack(
        [
            sample1["dipole"].to(**dd),
            sample2["dipole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU

    # required for autodiff of energy w.r.t. efield and dipole
    field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    # with 5 iterations, both do not converge, but pass the test
    options = dict(opts, **{"maxiter": 5, "mixer": "simple"})

    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    dipole = calc.dipole(numbers, positions, charge)
    dipole.detach_()

    assert pytest.approx(ref, abs=1e-2, rel=1e-3) == dipole
