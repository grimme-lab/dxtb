"""
Run tests for the gradient of the Hamiltonian matrix.
References calculated with tblite 0.3.0.
"""

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.constants import labels
from dxtb.integral.driver.pytorch import IntDriverPytorch
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

# references
ref_grad_no_overlap = np.load("test/test_hamiltonian/grad_no_overlap.npz")
ref_grad = np.load("test/test_hamiltonian/grad.npz")

# lists of test molecules
small = ["H2", "LiH", "S2", "H2O", "CH4", "SiH4"]
large = ["PbH4-BiH3", "MB16_43_01", "LYS_xao"]

# SCF options
opts = {"verbosity": 0, "int_driver": labels.INTDRIVER_ANALYTICAL}

device = None


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", small)
def test_no_overlap_single(dtype: torch.dtype, name: str) -> None:
    no_overlap_single(dtype, name)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", large)
def test_no_overlap_single_large(dtype: torch.dtype, name: str) -> None:
    no_overlap_single(dtype, name)


def no_overlap_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    chrg = torch.tensor(0.0, **dd)

    ref_dedr = load_from_npz(ref_grad_no_overlap, name, dtype)
    ref_dedcn = load_from_npz(ref_grad_no_overlap, f"{name}_dedcn", dtype)

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, chrg)

    # check setup
    o = result.integrals.overlap
    assert o is not None
    assert o.matrix is not None

    h = result.integrals.hcore
    assert h is not None
    assert h.matrix is not None

    # set derivative of overlap to zero
    doverlap = torch.zeros((*o.matrix.shape, 3), **dd)

    cn = get_coordination_number(numbers, positions, exp_count)
    wmat = get_density(
        result.coefficients,
        result.occupation.sum(-2),
        emo=result.emo,
    )

    dedcn, dedr = h.integral.get_gradient(
        positions,
        o.matrix,
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

    ref_dcn = load_from_npz(ref_grad_no_overlap, f"{name}_dcn", dtype)
    assert pytest.approx(dcn, abs=tol) == ref_dcn


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", small)
def test_no_overlap_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    no_overlap_batch(dtype, name1, name2)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", large)
def test_no_overlap_batch_large(dtype: torch.dtype, name1: str, name2: str) -> None:
    no_overlap_batch(dtype, name1, name2)


def no_overlap_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
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
            load_from_npz(ref_grad_no_overlap, name1, dtype),
            load_from_npz(ref_grad_no_overlap, name2, dtype),
        )
    )
    ref_dedcn = batch.pack(
        (
            load_from_npz(ref_grad_no_overlap, f"{name1}_dedcn", dtype),
            load_from_npz(ref_grad_no_overlap, f"{name2}_dedcn", dtype),
        )
    )

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, chrg)

    # check setup
    o = result.integrals.overlap
    assert o is not None
    assert o.matrix is not None

    h = result.integrals.hcore
    assert h is not None
    assert h.matrix is not None

    # set derivative of overlap to zero
    doverlap = torch.zeros((*o.matrix.shape, 3), **dd)

    cn = get_coordination_number(numbers, positions, exp_count)
    wmat = get_density(
        result.coefficients,
        result.occupation.sum(-2),
        emo=result.emo,
    )

    dedcn, dedr = h.integral.get_gradient(
        positions,
        o.matrix,
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
            load_from_npz(ref_grad_no_overlap, f"{name1}_dcn", dtype),
            load_from_npz(ref_grad_no_overlap, f"{name2}_dcn", dtype),
        )
    )
    assert pytest.approx(dcn, abs=tol) == ref_dcn


################
# With Overlap #
################


def hamiltonian_grad_single(dtype: torch.dtype, name: str) -> None:
    """
    Test implementation of analytical gradient de/dr against tblite reference
    gradient.
    Optionally, autograd and numerical gradient can also be calculated,
    although they may be numerically unstable.
    """
    dd: DD = {"device": device, "dtype": dtype}
    atol = 1e-4

    # tblite references
    ref = load_from_npz(ref_grad, f"{name}", dtype)
    ref_dedcn = load_from_npz(ref_grad_no_overlap, f"{name}_dedcn", dtype)

    # setup
    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    positions.requires_grad_(True)
    chrg = torch.tensor(0.0, **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, chrg)

    # check setup
    o = result.integrals.overlap
    assert o is not None
    assert o.matrix is not None

    s = calc.integrals.overlap
    assert s is not None

    h = result.integrals.hcore
    assert h is not None
    assert h.matrix is not None

    # compare different overlap calculations
    driver = IntDriverPytorch(numbers, par, calc.ihelp, **dd)
    driver.setup(positions)
    overlap2 = s.build(driver).detach()
    overlap = o.matrix.detach()

    tol = sqrt(torch.finfo(dtype).eps) * 5
    assert pytest.approx(overlap2, abs=tol, rel=tol) == overlap

    cn = get_coordination_number(numbers, positions, exp_count)
    wmat = get_density(
        result.coefficients,
        result.occupation.sum(-2),
        emo=result.emo,
    )

    # analytical overlap gradient
    doverlap = s.integral.get_gradient(driver)  # type: ignore

    # analytical gradient
    dedcn, dedr = h.integral.get_gradient(
        positions,
        o.matrix,
        doverlap,
        result.density,
        wmat,
        result.potential,
        cn,
    )

    # NOTE: Autograd and especially numerical gradient are not robust.
    #       Therefore only mentioned for completeness here.
    #
    # # autograd gradient
    # energy = result.scf.sum(-1)
    # autograd = torch.autograd.grad(
    #     energy,
    #     positions,
    # )[0]
    #
    # # numerical gradient
    # positions.requires_grad_(False)
    # numerical = calc_numerical_gradient(calc, positions, numbers, chrg)

    assert pytest.approx(ref, abs=atol) == dedr.detach()

    # NOTE: dedcn is already tested in test_hamiltonian
    assert pytest.approx(ref_dedcn, abs=atol) == dedcn.detach()

    positions.detach_()


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", small)
def test_single(dtype: torch.dtype, name: str) -> None:
    hamiltonian_grad_single(dtype, name)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", large)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    hamiltonian_grad_single(dtype, name)


def hamiltonian_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    # tol = sqrt(torch.finfo(dtype).eps) * 10
    atol = 1e-4

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
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
            load_from_npz(ref_grad, f"{name1}", dtype),
            load_from_npz(ref_grad, f"{name2}", dtype),
        )
    )
    ref_dedcn = batch.pack(
        (
            load_from_npz(ref_grad_no_overlap, f"{name1}_dedcn", dtype),
            load_from_npz(ref_grad_no_overlap, f"{name2}_dedcn", dtype),
        )
    )

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, chrg)

    # check setup
    o = result.integrals.overlap
    assert o is not None
    assert o.matrix is not None

    s = calc.integrals.overlap
    assert s is not None

    h = result.integrals.hcore
    assert h is not None
    assert h.matrix is not None

    # analytical overlap gradient
    driver = IntDriverPytorch(numbers, par, calc.ihelp, **dd)
    driver.setup(positions)
    doverlap = s.integral.get_gradient(driver)  # type: ignore

    cn = get_coordination_number(numbers, positions, exp_count)
    wmat = get_density(
        result.coefficients,
        result.occupation.sum(-2),
        emo=result.emo,
    )

    dedcn, dedr = h.integral.get_gradient(
        positions,
        o.matrix,
        doverlap,
        result.density,
        wmat,
        result.potential,
        cn,
    )

    assert pytest.approx(ref_dedcn, abs=atol) == dedcn.detach()
    assert pytest.approx(ref_dedr, abs=atol) == dedr.detach()

    # full CN
    dcndr = get_coordination_number_gradient(numbers, positions, dexp_count)
    dcn = get_dcn(dcndr, dedcn)

    ref_dcn = batch.pack(
        (
            load_from_npz(ref_grad_no_overlap, f"{name1}_dcn", dtype),
            load_from_npz(ref_grad_no_overlap, f"{name2}_dcn", dtype),
        )
    )
    assert pytest.approx(ref_dcn, abs=atol) == dcn.detach()


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", small)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    hamiltonian_grad_batch(dtype, name1, name2)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH", "PbH4-BiH3"])
@pytest.mark.parametrize("name2", large)
def test_batch_large(dtype: torch.dtype, name1: str, name2: str) -> None:
    hamiltonian_grad_batch(dtype, name1, name2)


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
            sr = calc.singlepoint(numbers, positions, chrg)
            er = sr.scf.sum(-1)

            positions[i, j] -= 2 * step
            sl = calc.singlepoint(numbers, positions, chrg)
            el = sl.scf.sum(-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
