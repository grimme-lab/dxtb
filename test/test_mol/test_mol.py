"""
Test the molecule representation.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.exceptions import DeviceError
from dxtb.mol import Mol

from ..utils import get_device_from_str
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


@pytest.mark.parametrize("name", sample_list)
def test_name(name: str) -> None:
    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(device)

    mol = Mol(numbers, positions, name=name)
    assert mol.name == name

    mol.name = "wrong"
    assert mol.name != name


def test_cache() -> None:
    sample = samples["SiH4"]
    numbers = sample["numbers"]
    positions = sample["positions"]

    mol = Mol(numbers, positions)
    assert hasattr(mol.distances, "clear")

    del mol.distances.__dict__["clear"]
    assert not hasattr(mol.distances, "clear")

    # clear cache should still execute without error
    mol.clear_cache()


@pytest.mark.cuda
def test_wrong_device() -> None:
    sample = samples["SiH4"]
    numbers = sample["numbers"].to(get_device_from_str("cpu"))
    positions = sample["positions"].to(get_device_from_str("cuda"))

    with pytest.raises(DeviceError):
        Mol(numbers, positions)
