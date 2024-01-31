"""
Run tests for IR spectra.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.constants import units
from dxtb.interaction import new_efield
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

sample_list = ["H", "H2", "LiH", "HHe", "H2O", "CH4", "SiH4", "PbH4-BiH3", "MB16_43_01"]

opts = {
    "int_level": 3,  #
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "f_atol": 1.0e-9,
    "x_atol": 1.0e-9,
}

device = None


def single(
    name: str,
    ref: Tensor,
    field_vector: Tensor,
    use_autograd: bool,
    dd: DD,
    atol: float,
    rtol: float,
) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    # required for autodiff of energy w.r.t. efield and quadrupole
    if use_autograd is True:
        field_vector.requires_grad_(True)
        positions.requires_grad_(True)
        field_grad = torch.zeros((3, 3), **dd, requires_grad=True)
    else:
        field_grad = None

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector, field_grad)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    quadrupole = calc.quadrupole(numbers, positions, charge, use_autograd=use_autograd)
    quadrupole = quadrupole.detach()
    print(ref)
    print(quadrupole)

    assert pytest.approx(ref, abs=atol, rel=rtol) == quadrupole


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("use_autograd", [False, True])
def test_single(dtype: torch.dtype, name: str, use_autograd: bool) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    ref = samples[name]["quadrupole"].to(**dd)

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)  # * units.VAA2AU
    atol, rtol = 1e-3, 1e-4
    single(name, ref, field_vector, use_autograd, dd=dd, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["LYS_xao", "C60"])
@pytest.mark.parametrize("use_autograd", [False])
def test_single_medium(dtype: torch.dtype, name: str, use_autograd: bool) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    ref = samples[name]["quadrupole"].to(**dd)

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU
    atol, rtol = 1e-2, 1e-2
    single(name, ref, field_vector, use_autograd, dd=dd, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("use_autograd", [False])
def test_single_field(dtype: torch.dtype, name: str, use_autograd: bool) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    ref = samples[name]["quadrupole2"].to(**dd)

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * units.VAA2AU
    atol, rtol = 1e-3, 1e-3
    single(name, ref, field_vector, use_autograd, dd=dd, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["LYS_xao", "C60"])
@pytest.mark.parametrize("use_autograd", [False])
def test_single_field_medium(dtype: torch.dtype, name: str, use_autograd: bool) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    ref = samples[name]["quadrupole2"].to(**dd)

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * units.VAA2AU
    atol, rtol = 1e-2, 1e-2
    single(name, ref, field_vector, use_autograd, dd=dd, atol=atol, rtol=rtol)


def batched(
    name1: str,
    name2: str,
    refname: str,
    field_vector: Tensor,
    use_autograd: bool,
    dd: DD,
    atol: float,
    rtol: float,
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

    ref = batch.pack(
        [
            sample1[refname].to(**dd),
            sample2[refname].to(**dd),
        ]
    )

    # required for autodiff of energy w.r.t. efield and quadrupole
    if use_autograd is True:
        field_vector.requires_grad_(True)
        positions.requires_grad_(True)
        field_grad = torch.zeros((3, 3), **dd, requires_grad=True)
    else:
        field_grad = None

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector, field_grad)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    quadrupole = calc.quadrupole(numbers, positions, charge, use_autograd=use_autograd)
    quadrupole.detach_()

    assert pytest.approx(ref, abs=atol, rel=rtol) == quadrupole


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("use_autograd", [False])
def test_batch(dtype: torch.dtype, name1: str, name2, use_autograd: bool) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU
    atol, rtol = 1e-3, 1e-3
    batched(
        name1,
        name2,
        "quadrupole",
        field_vector,
        use_autograd,
        dd=dd,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["C60"])
@pytest.mark.parametrize("use_autograd", [False])
def test_batch_medium(
    dtype: torch.dtype, name1: str, name2, use_autograd: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU
    atol, rtol = 1e-2, 1e-2
    batched(
        name1,
        name2,
        "quadrupole",
        field_vector,
        use_autograd,
        dd=dd,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("use_autograd", [False])
def test_batch_field(dtype: torch.dtype, name1: str, name2, use_autograd: bool) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * units.VAA2AU
    atol, rtol = 1e-3, 1e-4
    batched(
        name1,
        name2,
        "quadrupole2",
        field_vector,
        use_autograd,
        dd=dd,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["C60"])
@pytest.mark.parametrize("use_autograd", [False])
def test_batch_field_medium(
    dtype: torch.dtype, name1: str, name2, use_autograd: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * units.VAA2AU
    atol, rtol = 1e-2, 1e-2
    batched(
        name1,
        name2,
        "quadrupole2",
        field_vector,
        use_autograd,
        dd=dd,
        atol=atol,
        rtol=rtol,
    )


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
            sample1["quadrupole"].to(**dd),
            sample2["quadrupole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU

    # required for autodiff of energy w.r.t. efield and quadrupole
    field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    efield = new_efield(field_vector)
    options = dict(opts, **{"scp_mode": scp_mode, "mixer": mixer})
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    quadrupole = calc.quadrupole(numbers, positions, charge)
    quadrupole.detach_()

    assert pytest.approx(ref, abs=1e-4) == quadrupole


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
            sample1["quadrupole"].to(**dd),
            sample2["quadrupole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * units.VAA2AU

    # required for autodiff of energy w.r.t. efield and quadrupole
    field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    # with 5 iterations, both do not converge, but pass the test
    options = dict(opts, **{"maxiter": 5, "mixer": "simple"})

    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    quadrupole = calc.quadrupole(numbers, positions, charge)
    quadrupole.detach_()

    assert pytest.approx(ref, abs=1e-2, rel=1e-3) == quadrupole
