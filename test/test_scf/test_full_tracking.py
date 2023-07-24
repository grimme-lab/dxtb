"""
Test for SCF.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

opts = {"verbosity": 0, "maxiter": 300, "scf_mode": "full_tracking"}

device = None


def single(
    dtype: torch.dtype,
    name: str,
    mixer: str,
    tol: float,
    scp_mode: str = "charge",
    intdriver: str = "libcint",
) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = sample["escf"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    options = dict(
        opts,
        **{
            "damp": 0.05 if mixer == "simple" else 0.4,
            "int_driver": intdriver,
            "mixer": mixer,
            "scp_mode": scp_mode,
            "xitorch_fatol": tol,
            "xitorch_xatol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "CH4", "SiH4"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("intdriver", ["dxtb", "libcint"])
def test_single(dtype: torch.dtype, name: str, mixer: str, intdriver: str):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    single(dtype, name, mixer, tol, intdriver=intdriver)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01", "LYS_xao"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_single_medium(dtype: torch.dtype, name: str, mixer: str):
    """Test a few larger system."""
    tol = sqrt(torch.finfo(dtype).eps) * 10
    single(dtype, name, mixer, tol)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["S2", "LYS_xao_dist"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_single_difficult(dtype: torch.dtype, name: str, mixer: str):
    """These systems do not reproduce tblite energies to high accuracy."""
    tol = 5e-3
    single(dtype, name, mixer, tol, scp_mode="potential")


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["C60", "vancoh2"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_single_large(dtype: torch.dtype, name: str, mixer: str):
    """Test a large systems (only float32 as they take some time)."""
    tol = sqrt(torch.finfo(dtype).eps) * 10
    single(dtype, name, mixer, tol)


def batched(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    mixer: str,
    tol: float,
    intdriver: str = "libcint",
) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample[0]["numbers"],
            sample[1]["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample[0]["positions"],
            sample[1]["positions"],
        )
    ).type(dtype)
    ref = batch.pack(
        (
            sample[0]["escf"],
            sample[1]["escf"],
        )
    ).type(dtype)
    charges = torch.tensor([0.0, 0.0], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.05 if mixer == "simple" else 0.4,
            "mixer": mixer,
            "scp_mode": "charge",
            "int_driver": intdriver,
            "xitorch_fatol": tol,
            "xitorch_xatol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("intdriver", ["dxtb", "libcint"])
def test_batch(
    dtype: torch.dtype, name1: str, name2: str, mixer: str, intdriver: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    batched(dtype, name1, name2, mixer, tol, intdriver=intdriver)


def batched_unconverged(
    ref,
    dtype: torch.dtype,
    name1: str,
    name2: str,
    name3: str,
    mixer: str,
    maxiter: int,
) -> None:
    """
    Regression test for unconverged case. For double precision, the reference
    values are different. Hence, the test only includes single precision.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name1], samples[name2], samples[name3]
    numbers = batch.pack(
        (
            sample[0]["numbers"],
            sample[1]["numbers"],
            sample[2]["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample[0]["positions"],
            sample[1]["positions"],
            sample[2]["positions"],
        )
    ).type(dtype)

    charges = torch.tensor([0.0, 0.0, 0.0], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.3,
            "maxiter": maxiter,
            "mixer": mixer,
            "scf_mode": "full",
            "scp_mode": "potential",
            "xitorch_fatol": tol,
            "xitorch_xatol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_unconverged_partly_anderson(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    # only for regression testing (copied unconverged energies)
    ref = torch.tensor(
        [-1.058598403609326, -0.881055307373386, -4.024551310332887], **dd
    )

    batched_unconverged(ref, dtype, "H2", "LiH", "SiH4", "anderson", 1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_unconverged_partly_simple(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    # only for regression testing (copied unconverged energies)
    ref = torch.tensor(
        [-1.058598403609325, -0.882636830465807, -4.036953625762340], **dd
    )

    batched_unconverged(ref, dtype, "H2", "LiH", "SiH4", "simple", 1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_unconverged_fully_anderson(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    # only for regression testing (copied unconverged energies)
    ref = torch.tensor(
        [-0.881055307373386, -0.881055307373386, -4.024551310332887], **dd
    )

    batched_unconverged(ref, dtype, "LiH", "LiH", "SiH4", "anderson", 1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_unconverged_fully_simple(
    dtype: torch.dtype,
) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    # only for regression testing (copied unconverged energies)
    ref = torch.tensor(
        [-0.882636830465807, -0.882636830465807, -4.036953625762340], **dd
    )

    batched_unconverged(ref, dtype, "LiH", "LiH", "SiH4", "simple", 1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("name3", ["SiH4"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_batch_three(
    dtype: torch.dtype, name1: str, name2: str, name3: str, mixer: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name1], samples[name2], samples[name3]
    numbers = batch.pack(
        (
            sample[0]["numbers"],
            sample[1]["numbers"],
            sample[2]["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample[0]["positions"],
            sample[1]["positions"],
            sample[2]["positions"],
        )
    ).type(dtype)
    ref = batch.pack(
        (
            sample[0]["escf"],
            sample[1]["escf"],
            sample[2]["escf"],
        )
    ).type(dtype)
    charges = torch.tensor([0.0, 0.0, 0.0], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.1 if mixer == "simple" else 0.4,
            "mixer": mixer,
            "scf_mode": "full",
            "scp_mode": "charge",
            "xitorch_fatol": tol,
            "xitorch_xatol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_batch_special(dtype: torch.dtype, mixer: str) -> None:
    """
    Test case for https://github.com/grimme-lab/xtbML/issues/67.

    Note that the tolerance for the energy is quite high because atoms always
    show larger deviations w.r.t. the tblite reference. Secondly, this test
    should check if the overcounting in the IndexHelper and the corresponing
    additional padding upon spreading is prevented.
    """
    tol = 1e-2  # atoms show larger deviations
    dd: DD = {"device": device, "dtype": dtype}

    numbers = torch.tensor([[2, 2], [17, 0]])
    positions = batch.pack(
        [
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], **dd),
            torch.tensor([[0.0, 0.0, 0.0]], **dd),
        ]
    )
    chrg = torch.tensor([0.0, 0.0], **dd)
    ref = torch.tensor([-2.8629311088577, -4.1663539440167], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.05 if mixer == "simple" else 0.4,
            "mixer": mixer,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, chrg)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1)
