"""
Run tests for calculation of Born radii according to the Onufriev-Bashford-Case
model. Reference values are obtained from the tblite version.
"""

import pytest
import torch

from xtbml.exlibs.tbmalt import batch
from xtbml.solvation import born
from xtbml.solvation.data import vdw_rad_d3
from xtbml.typing import Tensor

from .samples import mb16_43


class TestBorn:
    """Test calculation of Born radii."""

    @classmethod
    def setup_class(cls):
        print(cls.__name__)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_psi_mb1643_01(self, dtype: torch.dtype):
        """Test psi for mb16_43_01."""
        sample = mb16_43["01"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        rvdw = vdw_rad_d3[numbers].type(dtype)

        psi = born.compute_psi(numbers, positions, rvdw)
        assert torch.allclose(psi, sample["psi"].type(dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_radii_mb1643_01(self, dtype: torch.dtype):
        """Test Born radii for mb16_43_01."""
        sample = mb16_43["01"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)

        rads = born.get_born_radii(numbers, positions)
        assert torch.allclose(rads, sample["born"].type(dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_psi_mb1643_02(self, dtype: torch.dtype):
        """Test Born psi for mb16_43_02."""
        sample = mb16_43["02"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        rvdw = vdw_rad_d3[numbers].type(dtype)

        psi = born.compute_psi(numbers, positions, rvdw)
        assert torch.allclose(psi, sample["psi"].type(dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_radii_mb1643_02(self, dtype: torch.dtype):
        """Test Born radii for mb16_43_02."""
        sample = mb16_43["02"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        descreening = torch.full(numbers.shape, 0.8, dtype=dtype)

        rads = born.get_born_radii(numbers, positions, descreening=descreening)
        assert torch.allclose(rads, sample["born"].type(dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_psi_batch(self, dtype: torch.dtype):
        """Test psi for batch."""
        sample1, sample2 = mb16_43["01"], mb16_43["SiH4"]

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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_radii_batch(self, dtype: torch.dtype):
        """Test Born radii for batch."""
        sample1, sample2 = mb16_43["01"], mb16_43["SiH4"]

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
    def test_psi_grad(self):
        """Test autograd of psi w.r.t to positions for mb16_43_01."""
        dtype = torch.float64
        sample = mb16_43["01"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        rvdw = vdw_rad_d3[numbers].type(dtype)

        positions.requires_grad_(True)

        def func(positions: Tensor):
            return born.compute_psi(numbers, positions, rvdw)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, positions)

    @pytest.mark.grad
    def test_radii_grad(self):
        """Test autograd of born radii w.r.t to positions for mb16_43_01."""
        dtype = torch.float64
        sample = mb16_43["01"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        positions.requires_grad_(True)

        def func(positions: Tensor):
            return born.get_born_radii(numbers, positions)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, positions)
