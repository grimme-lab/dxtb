"""
Run tests for calculation of Born radii according to the Onufriev-Bashford-Case
model. Reference values are obtained from the tblite version.
"""

import pytest
import torch

from dxtb.solvation import born, vdw_rad_d3
from dxtb._types import Tensor
from dxtb.utils import batch

from .samples import samples


@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_psi(name: str, dtype: torch.dtype):
    """Test psi for mb16_43_01 and mb16_43_02."""
    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    rvdw = vdw_rad_d3[numbers].type(dtype)

    psi = born.compute_psi(numbers, positions, rvdw)
    assert torch.allclose(psi, sample["psi"].type(dtype))


@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_radii(name: str, dtype: torch.dtype):
    """Test Born radii for mb16_43_01 and mb16_43_02."""
    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    rads = born.get_born_radii(numbers, positions)
    assert torch.allclose(rads, sample["born"].type(dtype))


@pytest.mark.parametrize("name1", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("name2", ["MB16_43_01", "SiH4"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_psi_batch(name1: str, name2: str, dtype: torch.dtype):
    """Test psi for batch."""
    sample1, sample2 = samples[name1], samples[name2]

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
    rvdw = vdw_rad_d3[numbers].type(dtype)

    ref = batch.pack(
        (
            sample1["psi"].type(dtype),
            sample2["psi"].type(dtype),
        ),
    )

    psi = born.compute_psi(numbers, positions, rvdw)
    assert torch.allclose(psi, ref)


@pytest.mark.parametrize("name1", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("name2", ["MB16_43_01", "SiH4"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_radii_batch(name1: str, name2: str, dtype: torch.dtype):
    """Test Born radii for batch."""
    sample1, sample2 = samples[name1], samples[name2]

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
    descreening = batch.pack(
        (
            torch.full(sample1["numbers"].shape, 0.8, dtype=dtype),
            torch.full(sample2["numbers"].shape, 0.8, dtype=dtype),
        )
    )

    ref = batch.pack(
        (
            sample1["born"].type(dtype),
            sample2["born"].type(dtype),
        ),
    )

    rads = born.get_born_radii(numbers, positions, descreening=descreening)
    assert torch.allclose(rads, ref)


@pytest.mark.grad
@pytest.mark.parametrize("name", ["MB16_43_01", "SiH4"])
def test_psi_grad(name: str):
    """Test autograd of psi w.r.t to positions."""
    dtype = torch.double
    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)
    rvdw = vdw_rad_d3[numbers].type(dtype)

    def func(positions: Tensor):
        return born.compute_psi(numbers, positions, rvdw)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)


@pytest.mark.grad
@pytest.mark.parametrize("name", ["MB16_43_01", "SiH4"])
def test_radii_grad(name: str):
    """Test autograd of born radii w.r.t to positions."""
    dtype = torch.double
    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)

    def func(positions: Tensor):
        return born.get_born_radii(numbers, positions)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)
