"""
Run tests for energy contribution from on-site third-order
electrostatic energy (ES3).
"""

from __future__ import annotations
from collections.abc import Generator
import pytest
import torch

import xtbml.coulomb.thirdorder as es3
from xtbml.exlibs.tbmalt import batch
from xtbml.param import GFN1_XTB, get_element_param
from xtbml.typing import Tensor

from .samples import mb16_43

sample_list = ["01", "02", "SiH4"]


@pytest.fixture(name="hubbard_derivs", scope="class")
def fixture_hubbard_derivs() -> Generator[Tensor, None, None]:
    if GFN1_XTB.thirdorder is None:
        raise ValueError("No ES3 parameters provided.")

    if GFN1_XTB.thirdorder.shell is True:
        raise NotImplementedError("Shell-resolved ES3 treatment not implemented.")

    yield get_element_param(GFN1_XTB.element, "gam3")


class TestThirdOrderElectrostatics:
    """Test the ES3 contribution."""

    @classmethod
    def setup_class(cls):
        print(f"\n{cls.__name__}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name", sample_list)
    def test_mb16_43(
        self,
        hubbard_derivs: Tensor,
        dtype: torch.dtype,
        name: str,
    ) -> None:
        """Test ES3 for some samples from mb16_43."""
        hd = hubbard_derivs.type(dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        qat = sample["q"].type(dtype)
        ref = sample["es3"].type(dtype)

        es = es3.ES3(hd)
        e = es.get_energy(numbers, qat)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name1", sample_list)
    @pytest.mark.parametrize("name2", sample_list)
    def test_batch(
        self,
        hubbard_derivs: Tensor,
        dtype: torch.dtype,
        name1: str,
        name2: str,
    ) -> None:
        hd = hubbard_derivs.type(dtype)

        sample1, sample2 = mb16_43[name1], mb16_43[name2]
        numbers = batch.pack(
            (
                sample1["numbers"],
                sample2["numbers"],
            )
        )
        qat = batch.pack(
            (
                sample1["q"].type(dtype),
                sample2["q"].type(dtype),
            )
        )
        ref = torch.stack(
            [
                sample1["es3"].type(dtype),
                sample2["es3"].type(dtype),
            ],
        )

        es = es3.ES3(hd)
        e = es.get_energy(numbers, qat)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.grad
    @pytest.mark.parametrize("name", sample_list)
    def test_grad_param(self, hubbard_derivs: Tensor, name: str) -> None:
        dtype = torch.float64
        hd = hubbard_derivs.type(dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        qat = sample["q"].type(dtype)

        # variable to be differentiated
        hd.requires_grad_(True)

        def func(hubbard_derivs):
            es = es3.ES3(hubbard_derivs)
            return es.get_energy(numbers, qat)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, hd)
