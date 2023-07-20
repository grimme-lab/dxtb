"""
Test bond order functionality.
"""
from __future__ import annotations

import pytest
import torch

from dxtb import bond
from dxtb._types import DD
from dxtb.utils import batch

from .samples import samples

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["PbH4-BiH3", "C6H5I-CH3SH"])
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    cn = sample["cn"].to(**dd)
    ref = sample["bo"].to(**dd)

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype
    assert pytest.approx(ref, abs=1.0e3) == bond_order[bond_order > 0.3]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ghost(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"].to(device).detach().clone()
    numbers[[0, 1, 2, 3, 4]] = 0
    positions = sample["positions"].to(**dd)
    cn = sample["cn"].to(**dd)
    ref = torch.tensor([0.5760, 0.5760, 0.5760, 0.5760, 0.5760, 0.5760], dtype=dtype)

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype
    assert pytest.approx(ref, abs=1.0e3) == bond_order[bond_order > 0.3]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
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
    cn = batch.pack(
        (
            sample1["cn"].to(**dd),
            sample2["cn"].to(**dd),
        )
    )
    ref = torch.cat(
        (
            sample1["bo"].to(**dd),
            sample2["bo"].to(**dd),
        )
    )

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype
    assert pytest.approx(ref, abs=1.0e3) == bond_order[bond_order > 0.3]
