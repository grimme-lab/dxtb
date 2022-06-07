import pytest
import torch

from xtbml.solvation import born

from .samples import mb16_43


class TestBorn:
    """Test calculation of Born radii."""

    @classmethod
    def setup_class(cls):
        print(cls.__name__)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_psi_mb1643_01(self, dtype: torch.dtype):
        """Test Born psi for mb16_43_01."""
        sample = mb16_43["01"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)

        psi = born.compute_psi(numbers, positions)
        print("psi", psi)

        assert torch.allclose(psi, sample["psi"].type(dtype), atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_radii_mb1643_01(self, dtype: torch.dtype):
        """Test Born radii for mb16_43_01."""
        sample = mb16_43["01"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)

        rads = born.get_born_radii(numbers, positions)

        print("rads", rads)
        print(sample["born"].type(dtype))

        assert torch.allclose(rads, sample["born"].type(dtype), atol=1e-4)
