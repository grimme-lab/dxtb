"""
Run tests for calculation of Born radii according to the Onufriev-Bashford-Case
model. Reference values are obtained from the tblite version.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.data import vdw_rad_d3
from dxtb.solvation import born
from dxtb.utils import batch

from ..utils import dgradcheck
from .samples import samples

device = None


@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_psi(name: str, dtype: torch.dtype):
    """Test psi for mb16_43_01 and mb16_43_02."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = sample["psi"].to(**dd)
    rvdw = vdw_rad_d3[numbers].to(**dd)

    psi = born.compute_psi(numbers, positions, rvdw)
    assert pytest.approx(ref, rel=1e-6, abs=tol) == psi


def test_fail_shape():
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # rcov wrong shape
    with pytest.raises(ValueError):
        rvdw = torch.tensor([1.0])
        born.get_born_radii(numbers, positions, rvdw)

    # wrong numbers
    with pytest.raises(ValueError):
        born.get_born_radii(torch.tensor([1]), positions)

    # wrong
    with pytest.raises(ValueError):
        descreening = torch.tensor([1])
        born.get_born_radii(numbers, positions, descreening=descreening)


@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_radii(name: str, dtype: torch.dtype):
    """Test Born radii for mb16_43_01 and mb16_43_02."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = sample["born"].to(**dd)

    rads = born.get_born_radii(numbers, positions)
    assert pytest.approx(ref, rel=1e-6, abs=tol) == rads


@pytest.mark.parametrize("name1", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("name2", ["MB16_43_01", "SiH4"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_psi_batch(name1: str, name2: str, dtype: torch.dtype):
    """Test psi for batch."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

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
    rvdw = vdw_rad_d3[numbers].to(**dd)

    ref = batch.pack(
        (
            sample1["psi"].to(**dd),
            sample2["psi"].to(**dd),
        ),
    )

    psi = born.compute_psi(numbers, positions, rvdw)
    assert pytest.approx(ref, rel=1e-6, abs=tol) == psi


@pytest.mark.parametrize("name1", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("name2", ["MB16_43_01", "SiH4"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_radii_batch(name1: str, name2: str, dtype: torch.dtype):
    """Test Born radii for batch."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

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
    descreening = batch.pack(
        (
            torch.full(sample1["numbers"].to(device).shape, 0.8, dtype=dtype),
            torch.full(sample2["numbers"].shape, 0.8, dtype=dtype),
        )
    )

    ref = batch.pack(
        (
            sample1["born"].to(**dd),
            sample2["born"].to(**dd),
        ),
    )

    rads = born.get_born_radii(numbers, positions, descreening=descreening)
    assert pytest.approx(ref, rel=1e-6, abs=tol) == rads


@pytest.mark.grad
@pytest.mark.parametrize("name", ["MB16_43_01", "SiH4"])
def test_psi_grad(name: str):
    """Test autograd of psi w.r.t to positions."""
    dd: DD = {"device": device, "dtype": torch.double}

    sample = samples[name]

    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    rvdw = vdw_rad_d3[numbers].to(**dd)

    # variable to be differentiated
    positions.requires_grad_(True)

    def func(positions: Tensor):
        return born.compute_psi(numbers, positions, rvdw)

    assert dgradcheck(func, positions)


@pytest.mark.grad
@pytest.mark.parametrize("name", ["MB16_43_01", "SiH4"])
def test_radii_grad(name: str):
    """Test autograd of born radii w.r.t to positions."""
    dd: DD = {"device": device, "dtype": torch.double}

    sample = samples[name]

    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    # variable to be differentiated
    positions.requires_grad_(True)

    def func(positions: Tensor):
        return born.get_born_radii(numbers, positions)

    assert dgradcheck(func, positions)
