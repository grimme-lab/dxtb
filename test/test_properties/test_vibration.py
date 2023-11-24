"""
Test vibrational frequencies.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.constants import units
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

sample_list = ["H", "H2", "LiH", "HHe", "H2O", "CH4", "SiH4"]

opts = {
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "xitorch_fatol": 1.0e-9,
    "xitorch_xatol": 1.0e-9,
}

device = None


def single(name: str, dd: DD, atol: float, rtol: float) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    ref = samples[name]["freqs"].to(**dd)

    calc = Calculator(numbers, par, opts=opts, **dd)
    freqs, _ = calc.vibration(numbers, positions, charge)
    freqs = freqs * units.AU2RCM

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
@pytest.mark.parametrize("name", sample_list)
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
            sample1["freqs"].to(**dd),
            sample2["freqs"].to(**dd),
        ]
    )

    calc = Calculator(numbers, par, opts=opts, **dd)
    freqs, _ = calc.vibration(numbers, positions, charge)
    freqs = freqs * units.AU2RCM

    if name1 == "LYS_xao" or name2 == "LYS_xao":
        freqs, ref = freqs[..., 4:], ref[..., 4:]

    assert pytest.approx(ref, abs=atol, rel=rtol) == freqs


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    atol, rtol = 10, 1e-3

    batched(name1, name2, dd=dd, atol=atol, rtol=rtol)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["LYS_xao"])
def test_batch_large(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    atol, rtol = 10, 1e-3

    batched(name1, name2, dd=dd, atol=atol, rtol=rtol)
