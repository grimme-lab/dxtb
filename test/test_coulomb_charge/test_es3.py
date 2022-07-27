"""
Run tests for energy contribution from on-site third-order
electrostatic energy (ES3).
"""

from __future__ import annotations
from collections.abc import Generator
import pytest
import torch

import xtbml.coulomb.thirdorder as es3
from xtbml.basis import IndexHelper
from xtbml.exlibs.tbmalt import batch
from xtbml.param import GFN1_XTB, get_elem_param, get_element_angular
from xtbml.typing import Tensor, Dict, List

from .samples import mb16_43

sample_list = ["01", "02", "SiH4"]
FixtureParams = Dict[int, List[int]]


@pytest.fixture(name="param", scope="class")
def fixture_param() -> Generator[FixtureParams, None, None]:

    if GFN1_XTB.thirdorder is None:
        raise ValueError("No ES3 parameters provided.")

    if GFN1_XTB.thirdorder.shell is True:
        raise NotImplementedError("Shell-resolved ES3 treatment not implemented.")

    yield get_element_angular(GFN1_XTB.element)


class TestThirdOrderElectrostatics:
    """Test the ES3 contribution."""

    @classmethod
    def setup_class(cls):
        print(f"\n{cls.__name__}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name", sample_list)
    def test_mb16_43(self, param: FixtureParams, dtype: torch.dtype, name: str) -> None:
        """Test ES3 for some samples from 16_43."""
        angular = param

        sample = mb16_43[name]
        numbers = sample["numbers"]
        qat = sample["q"].type(dtype)
        ref = sample["es3"].type(dtype)
        ihelp = IndexHelper.from_numbers(numbers, angular)

        hd = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "gam3",
            device=numbers.device,
            dtype=dtype,
        )

        es = es3.ES3(hd)
        e = es.get_atom_energy(qat, ihelp, None)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name1", sample_list)
    @pytest.mark.parametrize("name2", sample_list)
    def test_batch(
        self, param: FixtureParams, dtype: torch.dtype, name1: str, name2: str
    ) -> None:
        """Test batched calculation of ES3."""
        angular = param

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
        ihelp = IndexHelper.from_numbers(numbers, angular)

        hd = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "gam3",
            device=numbers.device,
            dtype=dtype,
        )

        es = es3.ES3(hd)
        e = es.get_atom_energy(qat, ihelp, None)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.grad
    @pytest.mark.parametrize("name", sample_list)
    def test_grad_param(self, param: FixtureParams, name: str) -> None:
        """Test autograd for ES3 parameters."""
        angular = param
        dtype = torch.float64

        sample = mb16_43[name]
        numbers = sample["numbers"]
        qat = sample["q"].type(dtype)
        ihelp = IndexHelper.from_numbers(numbers, angular)

        hd = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "gam3",
            device=numbers.device,
            dtype=dtype,
        )

        # variable to be differentiated
        hd.requires_grad_(True)

        def func(hubbard_derivs: Tensor):
            es = es3.ES3(hubbard_derivs)
            return es.get_atom_energy(qat, ihelp, None)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, hd)
