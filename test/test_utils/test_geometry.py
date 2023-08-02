"""
Test the utility functions.
"""
import pytest
import torch

from dxtb.utils import geometry

from .samples import samples

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH", "CO2"])
def test_linear(dtype: torch.dtype, name: str) -> None:
    positions = samples[name]["positions"].type(dtype).to(device)
    assert geometry.is_linear_molecule(positions)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_nonlinear(dtype: torch.dtype, name: str) -> None:
    positions = samples[name]["positions"].type(dtype).to(device)
    assert not geometry.is_linear_molecule(positions)
