"""
Test vibrational frequencies.
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

slist = ["H", "LiH", "HHe", "H2O"]
# FIXME: Larger systems fail for modes
# slist = ["H", "LiH", "HHe", "H2O", "CH4", "SiH4", "PbH4-BiH3"]

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


# FIXME: Autograd should also work on those
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def skip_test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    # required for autodiff of energy w.r.t. efield and dipole
    positions.requires_grad_(True)

    calc = Calculator(numbers, par, opts=opts, **dd)

    def f(pos: Tensor) -> tuple[Tensor, Tensor]:
        return calc.vibration(numbers, pos, charge)

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
    numfreqs, nummodes = calc.vibration_numerical(numbers, positions, charge)
    assert numfreqs.grad_fn is None
    assert nummodes.grad_fn is None

    # required for autodiff of energy w.r.t. positions (Hessian)
    pos = positions.clone().detach().requires_grad_(True)

    # manual jacobian
    freqs1, modes1 = calc.vibration(numbers, pos, charge, use_functorch=False)
    freqs1, modes1 = tensor_to_numpy(freqs1), tensor_to_numpy(modes1)

    assert pytest.approx(numfreqs, abs=atol, rel=rtol) == freqs1
    assert pytest.approx(nummodes, abs=atol, rel=rtol) == modes1

    # reset before another AD run
    calc.reset()
    pos = positions.clone().detach().requires_grad_(True)

    # jacrev of energy
    freqs2, modes2 = calc.vibration(numbers, pos, charge, use_functorch=True)
    freqs2, modes2 = tensor_to_numpy(freqs2), tensor_to_numpy(modes2)

    assert pytest.approx(numfreqs, abs=atol, rel=rtol) == freqs2
    assert pytest.approx(freqs1, abs=atol, rel=rtol) == freqs2
    assert pytest.approx(nummodes, abs=atol, rel=rtol) == modes2
    assert pytest.approx(modes1, abs=atol, rel=rtol) == modes2


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    single(name, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    batched(name1, name2, dd=dd)
