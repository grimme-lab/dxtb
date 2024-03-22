"""
Test the utility functions.
"""

import pytest
import torch

from dxtb.utils import batch, geometry

from .samples import samples

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH", "CO2"])
def test_linear(dtype: torch.dtype, name: str) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].type(dtype).to(device)
    mask = geometry.is_linear_molecule(numbers, positions)
    assert (mask == torch.tensor([True])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH", "CO2"])
@pytest.mark.parametrize("name2", ["LiH", "CO2"])
def test_linear_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    numbers = batch.pack(
        [
            samples[name1]["numbers"].to(device),
            samples[name2]["numbers"].to(device),
        ]
    )
    positions = batch.pack(
        [
            samples[name1]["positions"].type(dtype).to(device),
            samples[name2]["positions"].type(dtype).to(device),
        ]
    )

    mask = geometry.is_linear_molecule(numbers, positions)
    assert (mask == torch.tensor([True, True])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_nonlinear(dtype: torch.dtype, name: str) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].type(dtype).to(device)

    mask = geometry.is_linear_molecule(numbers, positions)
    assert (mask == torch.tensor([False])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2O", "SiH4"])
@pytest.mark.parametrize("name2", ["H2O", "SiH4"])
def test_nonlinear_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    numbers = batch.pack(
        [
            samples[name1]["numbers"].to(device),
            samples[name2]["numbers"].to(device),
        ]
    )
    positions = batch.pack(
        [
            samples[name1]["positions"].type(dtype).to(device),
            samples[name2]["positions"].type(dtype).to(device),
        ]
    )

    mask = geometry.is_linear_molecule(numbers, positions)
    assert (mask == torch.tensor([False, False])).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2O", "SiH4"])
@pytest.mark.parametrize("name2", ["LiH", "CO2"])
def test_mixed_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    numbers = batch.pack(
        [
            samples[name1]["numbers"].to(device),
            samples[name2]["numbers"].to(device),
        ]
    )
    positions = batch.pack(
        [
            samples[name1]["positions"].type(dtype).to(device),
            samples[name2]["positions"].type(dtype).to(device),
        ]
    )

    mask = geometry.is_linear_molecule(numbers, positions)
    assert (mask == torch.tensor([False, True])).all()
