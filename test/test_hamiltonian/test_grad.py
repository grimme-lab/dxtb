"""
Run tests for the gradient of the Hamiltonian matrix.
References calculated with tblite 0.3.0.
"""

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb.ncoord import (
    dexp_count,
    exp_count,
    get_coordination_number,
    get_coordination_number_gradient,
    get_dcn,
)
from dxtb.param import GFN1_XTB as par
from dxtb.scf import get_density
from dxtb.utils import batch
from dxtb.xtb import Calculator

from ..utils import load_from_npz
from .samples import samples

ref_grad = np.load("test/test_hamiltonian/grad.npz")

small = ["H2", "LiH", "S2", "H2O", "CH4", "SiH4"]
large = ["PbH4-BiH3", "MB16_43_01", "LYS_xao"]

opts = {"verbosity": 0}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", small)
def test_no_overlap_single(dtype: torch.dtype, name: str) -> None:
    no_overlap_single(dtype, name)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", large)
def test_no_overlap_single_large(dtype: torch.dtype, name: str) -> None:
    no_overlap_single(dtype, name)


def no_overlap_single(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    chrg = torch.tensor(0.0, **dd)

    ref_dedr = load_from_npz(ref_grad, name, dtype)
    ref_dedcn = load_from_npz(ref_grad, f"{name}_dedcn", dtype)

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, chrg)

    # set derivative of overlap to zero
    doverlap = torch.tensor(0.0, **dd)

    cn = get_coordination_number(numbers, positions, exp_count)
    wmat = get_density(
        result.coefficients,
        result.occupation.sum(-2),
        emo=result.emo,
    )

    dedcn, dedr = calc.hamiltonian.get_gradient(
        positions,
        result.overlap,
        doverlap,
        result.density,
        wmat,
        result.potential,
        cn,
    )

    assert pytest.approx(dedcn, abs=tol) == ref_dedcn
    assert pytest.approx(dedr, abs=tol) == ref_dedr

    # full CN gradient
    dcndr = get_coordination_number_gradient(numbers, positions, dexp_count)
    dcn = get_dcn(dcndr, dedcn)

    # ref_dcn = load_from_npz(ref_grad, f"{name}_dcn", dtype)
    # assert pytest.approx(dcn, abs=tol) == ref_dcn


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", small)
def test_no_overlap_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    no_overlap_batch(dtype, name1, name2)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", large)
def test_no_overlap_batch_large(dtype: torch.dtype, name1: str, name2: str) -> None:
    no_overlap_batch(dtype, name1, name2)


def no_overlap_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        )
    )
    chrg = batch.pack(
        (
            torch.tensor(0.0, **dd),
            torch.tensor(0.0, **dd),
        )
    )

    ref_dedr = batch.pack(
        (
            load_from_npz(ref_grad, name1, dtype),
            load_from_npz(ref_grad, name2, dtype),
        )
    )
    ref_dedcn = batch.pack(
        (
            load_from_npz(ref_grad, f"{name1}_dedcn", dtype),
            load_from_npz(ref_grad, f"{name2}_dedcn", dtype),
        )
    )

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, chrg)

    # set derivative of overlap to zero
    doverlap = torch.tensor(0.0, **dd)

    cn = get_coordination_number(numbers, positions, exp_count)
    wmat = get_density(
        result.coefficients,
        result.occupation.sum(-2),
        emo=result.emo,
    )

    dedcn, dedr = calc.hamiltonian.get_gradient(
        positions,
        result.overlap,
        doverlap,
        result.density,
        wmat,
        result.potential,
        cn,
    )

    assert pytest.approx(dedcn, abs=tol) == ref_dedcn
    assert pytest.approx(dedr, abs=tol) == ref_dedr

    # full CN
    dcndr = get_coordination_number_gradient(numbers, positions, dexp_count)
    dcn = get_dcn(dcndr, dedcn)

    # ref_dcn = batch.pack(
    #     (
    #         load_from_npz(ref_grad, f"{name1}_dcn", dtype),
    #         load_from_npz(ref_grad, f"{name2}_dcn", dtype),
    #     )
    # )

    # assert pytest.approx(dcn, abs=tol) == ref_dcn
