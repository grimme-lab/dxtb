"""
Run tests for energy contribution from instantaneous electric field.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD
from dxtb.components.interactions import new_efield
from dxtb.constants import units
from dxtb.param import GFN1_XTB
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

sample_list = ["MB16_43_01"]
sample_list = ["LiH", "SiH4"]

opts = {"verbosity": 0, "maxiter": 50}

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    ref = sample["energy"].to(**dd)
    # ref1 = sample["energy_monopole"].to(**dd)

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * units.VAA2AU
    efield = new_efield(field_vector)
    calc = Calculator(numbers, GFN1_XTB, interaction=[efield], opts=opts, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.total.sum(-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("scf_mode", ["implicit", "full"])
def test_batch(dtype: torch.dtype, name1: str, name2: str, scf_mode: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"device": device, "dtype": dtype}

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
    charges = torch.tensor([0.0, 0.0], **dd)

    ref1 = torch.stack(
        [
            sample1["energy"].to(**dd),
            sample2["energy"].to(**dd),
        ],
    )

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * units.VAA2AU
    efield = new_efield(field_vector)
    options = dict(opts, **{"scf_mode": scf_mode, "mixer": "anderson"})
    calc = Calculator(numbers, GFN1_XTB, interaction=[efield], opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref1, abs=tol) == result.total.sum(-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("name3", sample_list)
@pytest.mark.parametrize("scf_mode", ["implicit", "full"])
def test_batch_three(
    dtype: torch.dtype, name1: str, name2: str, name3: str, scf_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2, sample3 = samples[name1], samples[name2], samples[name3]
    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
            sample3["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
            sample3["positions"].to(**dd),
        )
    )
    charges = torch.tensor([0.0, 0.0, 0.0], **dd)

    ref = torch.stack(
        [
            sample1["energy"].to(**dd),
            sample2["energy"].to(**dd),
            sample3["energy"].to(**dd),
        ],
    )

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * units.VAA2AU
    efield = new_efield(field_vector)
    options = dict(opts, **{"scf_mode": scf_mode, "mixer": "anderson"})
    calc = Calculator(numbers, GFN1_XTB, interaction=[efield], opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.total.sum(-1)
