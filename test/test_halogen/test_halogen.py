"""Run tests for energy contribution from halogen bond correction."""

import pytest
import torch

from xtbml.classical.halogen import halogen_bond_correction
from xtbml.exlibs.tbmalt import batch
from xtbml.param.gfn1 import GFN1_XTB

from .samples import samples


class TestHalogenBondCorrection:
    """Test the halogen bond correction."""

    @classmethod
    def setup_class(cls):
        print(cls.__name__)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_br2nh3(self, dtype: torch.dtype):
        sample = samples["br2nh3"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["energy"].type(dtype)

        xb = halogen_bond_correction(numbers, positions, GFN1_XTB)
        assert torch.allclose(ref, torch.sum(xb))

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_br2och2(self, dtype: torch.dtype):
        sample = samples["br2och2"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["energy"].type(dtype)

        xb = halogen_bond_correction(numbers, positions, GFN1_XTB)
        assert torch.allclose(ref, torch.sum(xb))

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_finch(self, dtype: torch.dtype):
        sample = samples["finch"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["energy"].type(dtype)

        xb = halogen_bond_correction(numbers, positions, GFN1_XTB)
        assert torch.allclose(ref, torch.sum(xb))

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_gfn1_batch(self, dtype: torch.dtype):
        sample1, sample2 = samples["br2nh3"], samples["finch"]
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

        ref = torch.stack(
            [
                sample1["energy"].type(dtype),
                sample2["energy"].type(dtype),
            ],
        )

        xb = halogen_bond_correction(numbers, positions, GFN1_XTB)
        print(xb)
        print(ref)
        print(torch.sum(xb, dim=-1))
        assert torch.allclose(ref, torch.sum(xb, dim=-1))
