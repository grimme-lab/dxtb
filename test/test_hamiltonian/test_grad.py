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
from dxtb._types import Tensor

from ..utils import load_from_npz
from .samples import samples
from .gradients import gradients

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

    ref_dcn = load_from_npz(ref_grad, f"{name}_dcn", dtype)
    assert pytest.approx(dcn, abs=tol) == ref_dcn


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

    ref_dcn = batch.pack(
        (
            load_from_npz(ref_grad, f"{name1}_dcn", dtype),
            load_from_npz(ref_grad, f"{name2}_dcn", dtype),
        )
    )
    assert pytest.approx(dcn, abs=tol) == ref_dcn


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", small + large)
def test_hamiltonian_grad_single(dtype: torch.dtype, name: str) -> None:
    """ Test implementation of analytical gradient de/dr against tblite reference gradient.
        Optionally, autograd and numerical gradient can also be calculated, although they may be numerically unstable.
    """

    dd = {"dtype": dtype}
    atol = 1e-4

    # tblite references
    ref = gradients[name]
    ref_dedcn = load_from_npz(ref_grad, f"{name}_dedcn", dtype)

    # setup
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    positions.requires_grad_(True)
    chrg = torch.tensor(0.0, **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, chrg)

    # analytical overlap gradient
    atomwise = False # [natm, 3] vs [natm, natm, 3]
    doverlap = calc.overlap.get_gradient(positions, atomwise)

    cn = get_coordination_number(numbers, positions, exp_count)
    wmat = get_density(
        result.coefficients,
        result.occupation.sum(-2),
        emo=result.emo,
    )

    # analytical gradient
    dedcn, dedr = calc.hamiltonian.get_gradient(
        positions,
        result.overlap,
        doverlap,
        result.density,
        wmat,
        result.potential,
        cn,
        calc.ihelp,
    )

    # NOTE: Autograd and especially numerical gradient are not robust.
    #       Therefore only mentioned for completeness here.
    '''# autograd gradient
    energy = result.scf.sum(-1)
    autograd = torch.autograd.grad(
        energy,
        positions,
    )[0]

    # numerical gradient
    positions.requires_grad_(False)
    numerical = calc_numerical_gradient(calc, positions, numbers, chrg)
    '''

    dedr = dedr.detach().numpy()
    assert pytest.approx(dedr, abs=atol) == ref

    # NOTE: dedcn is already tested in test_hamiltonian
    dedcn = dedcn.detach().numpy()
    assert pytest.approx(dedcn, abs=atol) == ref_dedcn

def calc_numerical_gradient(
    calc: Calculator, positions: Tensor, numbers: Tensor, chrg: Tensor
) -> Tensor:
    """Calculate numerical gradient of Energies to positions (de/dr)
        by shifting atom positions.
    """

    # setup numerical gradient
    step = 1.0e-6
    natm = positions.shape[0]
    gradient = torch.zeros((natm, 3), dtype=positions.dtype)

    for i in range(natm):
        for j in range(3):
            positions[i, j] += step
            sR = calc.singlepoint(numbers, positions, chrg)
            eR = sR.scf.sum(-1)

            positions[i, j] -= 2 * step
            sL = calc.singlepoint(numbers, positions, chrg)
            eL = sL.scf.sum(-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (eR - eL) / step

    return gradient