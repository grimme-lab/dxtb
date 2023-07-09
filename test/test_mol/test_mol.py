"""
Test the molecule representation.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.mol import Mol

from .samples import samples

sample_list = ["H2", "LiH", "H2O", "SiH4", "MB16_43_01", "vancoh2"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_dist(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    mol = Mol(numbers, positions)
    dist = mol.distances()
    mol.clear_cache()

    assert dist.shape[-1] == numbers.shape[-1]
    assert dist.shape[-2] == numbers.shape[-1]

    assert pytest.approx(numbers) == mol.numbers
    assert pytest.approx(positions) == mol.positions
