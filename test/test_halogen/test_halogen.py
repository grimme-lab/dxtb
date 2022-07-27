"""Run tests for energy contribution from halogen bond correction."""

from __future__ import annotations
import pytest
import torch

from xtbml.classical.halogen import (
    get_xbond,
    halogen_bond_correction,
)
from xtbml.exlibs.tbmalt import batch
from xtbml.param.gfn1 import GFN1_XTB
from xtbml.typing import Generator, Tensor, Tuple

from .samples import samples

FixtureParams = Tuple[Tensor, Tensor, Tensor]


@pytest.fixture(name="param", scope="class")
def fixture_param() -> Generator[FixtureParams, None, None]:
    if GFN1_XTB.halogen is None:
        raise ValueError("No halogen bond correction parameters provided.")

    damp = torch.tensor(GFN1_XTB.halogen.classical.damping)
    rscale = torch.tensor(GFN1_XTB.halogen.classical.rscale)
    bond_strength = get_xbond(GFN1_XTB.element)

    yield damp, rscale, bond_strength


class TestHalogenBondCorrection:
    """Test the halogen bond correction."""

    @classmethod
    def setup_class(cls):
        print(cls.__name__)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name", ["br2nh3", "br2nh2o", "br2och2", "finch"])
    def test_small(self, param: FixtureParams, dtype: torch.dtype, name: str) -> None:
        """
        Test the halogen bond correction for small molecules taken from
        the tblite test suite.
        """
        damp, rscale, bond_strength = _cast(param, dtype)

        sample = samples[name]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["energy"].type(dtype)

        xb = halogen_bond_correction(numbers, positions, damp, rscale, bond_strength)
        assert torch.allclose(ref, torch.sum(xb))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name", ["tmpda", "tmpda_mod"])
    def test_large(self, param: FixtureParams, dtype: torch.dtype, name: str) -> None:
        """
        TMPDA@XB-donor from S30L (15AB). Contains three iodine donors and two
        nitrogen acceptors. In the modified version, one I is replaced with
        Br and one O is added in order to obtain different donors and acceptors.
        """
        damp, rscale, bond_strength = _cast(param, dtype)

        sample = samples[name]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["energy"].type(dtype)

        xb = halogen_bond_correction(numbers, positions, damp, rscale, bond_strength)
        assert torch.allclose(ref, torch.sum(xb))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_no_xb(self, param: FixtureParams, dtype: torch.dtype) -> None:
        """Test system without halogen bonds."""
        damp, rscale, bond_strength = _cast(param, dtype)

        sample = samples["LYS_xao"]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["energy"].type(dtype)

        xb = halogen_bond_correction(numbers, positions, damp, rscale, bond_strength)
        assert torch.allclose(ref, torch.sum(xb))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name1", ["br2nh3", "br2och2"])
    @pytest.mark.parametrize("name2", ["finch", "tmpda"])
    def test_batch(
        self, param: FixtureParams, dtype: torch.dtype, name1: str, name2: str
    ) -> None:
        damp, rscale, bond_strength = _cast(param, dtype)

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
        ref = torch.stack(
            [
                sample1["energy"].type(dtype),
                sample2["energy"].type(dtype),
            ],
        )

        xb = halogen_bond_correction(numbers, positions, damp, rscale, bond_strength)
        assert torch.allclose(ref, torch.sum(xb, dim=-1))

    @pytest.mark.grad
    @pytest.mark.parametrize("sample_name", ["br2nh3", "br2och2"])
    def stest_param_grad(self, param: FixtureParams, sample_name: str):
        dtype = torch.float64
        damp, rscale, bond_strength = _cast(param, dtype)
        damp.requires_grad_(True)
        rscale.requires_grad_(True)

        sample = samples[sample_name]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)

        def func(damping, rscaling):
            return halogen_bond_correction(
                numbers, positions, damping, rscaling, bond_strength
            )

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        # NOTE: For smaller values of atol, it fails.
        assert gradcheck(func, (damp, rscale), atol=1e-2)


def _cast(param: FixtureParams, dtype: torch.dtype) -> FixtureParams:
    """Cast the parameters to the given dtype.

    Args:
    -----
    param: FixtureParams
        The parameters to cast.
    dtype: torch.dtype
        The dtype to cast the parameters to.

    Returns:
    --------
    FixtureParams
        The parameters with the corresponding dtype.

    """
    damp, rscale, bond_strength = param
    return damp.type(dtype), rscale.type(dtype), bond_strength.type(dtype)
