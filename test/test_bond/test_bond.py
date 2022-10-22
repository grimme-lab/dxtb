"""
Test bond order functionality.
"""

import torch
import pytest

from xtbml import bond
from xtbml.utils import batch

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["PbH4-BiH3", "C6H5I-CH3SH"])
def test_single(dtype: torch.dtype, name: str):
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    cn = sample["cn"].type(dtype)
    ref = sample["bo"].type(dtype)

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype
    assert torch.allclose(bond_order[bond_order > 0.3], ref, atol=1.0e-3)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ghost(dtype: torch.dtype):
    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"].detach().clone()
    numbers[[0, 1, 2, 3, 4]] = 0
    positions = sample["positions"].type(dtype)
    cn = sample["cn"].type(dtype)
    ref = torch.tensor([0.5760, 0.5760, 0.5760, 0.5760, 0.5760, 0.5760], dtype=dtype)

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype
    assert torch.allclose(bond_order[bond_order > 0.3], ref, atol=1.0e-3)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype):
    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
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
    cn = batch.pack(
        (
            sample1["cn"].type(dtype),
            sample2["cn"].type(dtype),
        )
    )
    ref = torch.cat(
        (
            sample1["bo"].type(dtype),
            sample2["bo"].type(dtype),
        )
    )

    bond_order = bond.guess_bond_order(numbers, positions, cn)
    assert bond_order.dtype == dtype
    assert torch.allclose(bond_order[bond_order > 0.3], ref, atol=1.0e-3)
