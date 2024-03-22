"""
Test Hessian.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.typing import DD, Tensor

from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator

from .samples import samples

slist = ["H", "LiH", "HHe", "H2O", "CH4", "SiH4", "PbH4-BiH3"]
slist_large = ["MB16_43_01"]  # "LYS_xao"

opts = {
    "int_level": 1,
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

    # required for autodiff of energy w.r.t. efield and dipole
    positions.requires_grad_(True)

    calc = Calculator(numbers, par, opts=opts, **dd)

    def f(pos: Tensor) -> Tensor:
        return calc.hessian(numbers, pos, charge)

    assert dgradcheck(f, positions)


def single(
    name: str,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    execute(numbers, positions, charge, dd, atol, rtol)


def batched(
    name1: str,
    name2: str,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ],
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ],
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    execute(numbers, positions, charge, dd, atol, rtol)


def execute(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    dd: DD,
    atol: float,
    rtol: float,
) -> None:
    calc = Calculator(numbers, par, opts=opts, **dd)

    # field is cloned and detached and updated inside
    numhess = calc.hessian_numerical(numbers, positions, charge)
    assert numhess.grad_fn is None

    # required for autodiff of energy w.r.t. positions (Hessian)
    pos = positions.clone().detach().requires_grad_(True)

    # manual jacobian
    hess1 = tensor_to_numpy(calc.hessian(numbers, pos, charge, use_functorch=False))

    assert pytest.approx(numhess, abs=atol, rel=rtol) == hess1

    # reset before another AD run
    calc.reset()
    pos = positions.clone().detach().requires_grad_(True)

    # jacrev of energy
    hess2 = tensor_to_numpy(calc.hessian(numbers, pos, charge, use_functorch=True))

    assert pytest.approx(numhess, abs=atol, rel=rtol) == hess2
    assert pytest.approx(hess1, abs=atol, rel=rtol) == hess2


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    single(name, dd=dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    single(name, dd=dd)


# TODO: Batched Hessians are not supported yet
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    batched(name1, name2, dd=dd)


# TODO: Batched Hessians are not supported yet
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist_large)
def skip_test_batch_large(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    batched(name1, name2, dd=dd)
