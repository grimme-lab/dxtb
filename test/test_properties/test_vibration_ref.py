"""
Test vibrational frequencies.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.typing import DD
from tad_mctc.units.spectroscopy import AU2RCM

from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator

from .samples import samples

# FIXME: "HHe" is completely off
slist = ["H", "H2", "LiH", "H2O", "CH4", "SiH4"]
slist_large = ["LYS_xao"]

opts = {
    "int_level": 1,
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "f_atol": 1.0e-9,
    "x_atol": 1.0e-9,
}

device = None


def single(name: str, dd: DD, atol: float, rtol: float) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)
    ref = samples[name]["freqs"].to(**dd)

    # required for autodiff of energy w.r.t. positions (Hessian)
    pos = positions.clone().detach().requires_grad_(True)

    calc = Calculator(numbers, par, opts=opts, **dd)
    freqs, _ = calc.vibration(numbers, pos, charge)
    freqs = tensor_to_numpy(freqs * AU2RCM)

    # low frequencies mismatch
    if name == "PbH4-BiH3":
        freqs, ref = freqs[6:], ref[6:]

    # cut off the negative frequencies
    if name == "MB16_43_01":
        freqs, ref = freqs[1:], ref[1:]
    if name == "LYS_xao":
        freqs, ref = freqs[4:], ref[4:]

    assert pytest.approx(ref, abs=atol, rel=rtol) == freqs


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    atol, rtol = 10, 1e-3
    single(name, dd=dd, atol=atol, rtol=rtol)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["PbH4-BiH3", "MB16_43_01", "LYS_xao"])
def test_single_large(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    atol, rtol = 10, 1e-3
    single(name, dd=dd, atol=atol, rtol=rtol)


def batched(name1: str, name2: str, dd: DD, atol: float, rtol: float) -> None:
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

    ref = pack(
        [
            sample1["freqs"].to(**dd),
            sample2["freqs"].to(**dd),
        ]
    )

    # required for autodiff of energy w.r.t. positions (Hessian)
    pos = positions.clone().detach().requires_grad_(True)

    calc = Calculator(numbers, par, opts=opts, **dd)
    freqs, _ = calc.vibration(numbers, pos, charge)
    freqs = freqs * AU2RCM

    if name1 == "LYS_xao" or name2 == "LYS_xao":
        freqs, ref = freqs[..., 4:], ref[..., 4:]

    assert pytest.approx(ref, abs=atol, rel=rtol) == freqs


# TODO: Batched Hessians are not supported yet
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    batched(name1, name2, dd=dd, atol=10, rtol=1e-3)


# TODO: Batched Hessians are not supported yet
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist_large)
def skip_test_batch_large(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    batched(name1, name2, dd=dd, atol=10, rtol=1e-3)
