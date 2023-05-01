"""
Test for SCF.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

opts = {"verbosity": 0, "maxiter": 50}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "CH4", "SiH4"])
def test_single(dtype: torch.dtype, name: str):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"].type(dtype)
    charges = torch.tensor(0.0, **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "name", ["PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01", "LYS_xao", "C60"]
)
def test_single2(dtype: torch.dtype, name: str):
    """Test a few larger system (only float32 as they take some time)."""
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"].type(dtype)
    charges = torch.tensor(0.0, **dd)

    options = dict(
        opts,
        **{
            "fermi_fenergy_partition": "atomic",
            "use_potential": False,
            "xitorch_fatol": tol**2,
            "xitorch_xatol": tol**2,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", ["S2", "LYS_xao_dist"])
def test_single3(dtype: torch.dtype, name: str):
    """Test a few larger system (only float32 within tolerance)."""
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"].type(dtype)
    charges = torch.tensor(0.0, **dd)

    options = dict(opts, **{"xitorch_fatol": 1e-6, "xitorch_xatol": 1e-6})
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1)


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["vancoh2"])
def test_single_large(dtype: torch.dtype, name: str):
    """Test a large systems (only float32 as they take some time)."""
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"]
    charges = torch.tensor(0.0, **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4"])
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

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

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_special(dtype: torch.dtype):
    """
    Test case for https://github.com/grimme-lab/xtbML/issues/67.

    Note that the tolerance for the energy is quite high because atoms always
    show larger deviations w.r.t. the tblite reference. Secondly, this test
    should check if the overcounting in the IndexHelper and the corresponing
    additional padding upon spreading is prevented.
    """
    tol = 1e-2  # atoms show larger deviations
    dd = {"dtype": dtype}

    numbers = torch.tensor([[2, 2], [17, 0]])
    positions = batch.pack(
        [
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], **dd),
            torch.tensor([[0.0, 0.0, 0.0]], **dd),
        ]
    )
    chrg = torch.tensor([0.0, 0.0], **dd)
    ref = torch.tensor([-2.8629311088577, -4.1663539440167], **dd)

    calc = Calculator(numbers, par, **dd, opts=opts)
    result = calc.singlepoint(numbers, positions, chrg)

    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1)
