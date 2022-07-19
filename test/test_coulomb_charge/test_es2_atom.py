"""
Run tests for energy contribution from isotropic second-order
electrostatic energy (ES2).
"""

from __future__ import annotations
from collections.abc import Generator
import pytest
import torch

import xtbml.coulomb.secondorder as es2
from xtbml.coulomb import AveragingFunction
from xtbml.exlibs.tbmalt import batch
from xtbml.param import GFN1_XTB, get_element_param
from xtbml.typing import Tensor

from .samples import mb16_43

sample_list = ["01", "02", "SiH4"]


@pytest.fixture(name="param", scope="class")
def fixture_param() -> Generator[tuple[Tensor, AveragingFunction, Tensor], None, None]:
    if GFN1_XTB.charge is None:
        raise ValueError("No charge parameters provided.")

    gexp = torch.tensor(GFN1_XTB.charge.effective.gexp)
    hubbard = get_element_param(GFN1_XTB.element, "gam")

    if GFN1_XTB.charge.effective.average == "harmonic":
        # pylint: disable=import-outside-toplevel
        from xtbml.coulomb.average import harmonic_average as average
    elif GFN1_XTB.charge.effective.average == "geometric":
        # pylint: disable=import-outside-toplevel
        from xtbml.coulomb.average import geometric_average as average
    elif GFN1_XTB.charge.effective.average == "arithmetic":
        # pylint: disable=import-outside-toplevel
        from xtbml.coulomb.average import arithmetic_average as average
    else:
        raise ValueError("Unknown average function.")

    yield gexp, average, hubbard


class TestSecondOrderElectrostatics:
    """Test the ES2 contribution."""

    @classmethod
    def setup_class(cls):
        print(f"\n{cls.__name__}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name", sample_list)
    def test_mb16_43(
        self,
        param: tuple[Tensor, AveragingFunction, Tensor],
        dtype: torch.dtype,
        name: str,
    ) -> None:
        """Test ES2 for some samples from mb16_43."""
        gexp, average, hubbard = _cast(param, dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qat = sample["q"].type(dtype)
        ref = sample["es2"].type(dtype)

        es = es2.ES2(hubbard=hubbard, average=average, gexp=gexp)
        cache = es.get_cache(numbers, positions)
        e = es.get_energy(cache, qat)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name1", sample_list)
    @pytest.mark.parametrize("name2", sample_list)
    def test_batch(
        self,
        param: tuple[Tensor, AveragingFunction, Tensor],
        dtype: torch.dtype,
        name1: str,
        name2: str,
    ) -> None:
        gexp, average, hubbard = _cast(param, dtype)

        sample1, sample2 = mb16_43[name1], mb16_43[name2]
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
        qat = batch.pack(
            (
                sample1["q"].type(dtype),
                sample2["q"].type(dtype),
            )
        )
        ref = torch.stack(
            [
                sample1["es2"].type(dtype),
                sample2["es2"].type(dtype),
            ],
        )

        es = es2.ES2(hubbard=hubbard, average=average, gexp=gexp)
        cache = es.get_cache(numbers, positions)
        e = es.get_energy(cache, qat)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.grad
    @pytest.mark.parametrize("name", sample_list)
    def test_grad_positions(
        self, param: tuple[Tensor, AveragingFunction, Tensor], name: str
    ) -> None:
        dtype = torch.float64
        gexp, average, hubbard = _cast(param, dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qat = sample["q"].type(dtype)

        # variable to be differentiated
        positions.requires_grad_(True)

        def func(positions):
            es = es2.ES2(hubbard=hubbard, average=average, gexp=gexp)
            cache = es.get_cache(numbers, positions)
            return es.get_energy(cache, qat)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, positions)

    @pytest.mark.grad
    @pytest.mark.parametrize("name", sample_list)
    def test_grad_param(
        self, param: tuple[Tensor, AveragingFunction, Tensor], name: str
    ) -> None:
        dtype = torch.float64
        gexp, average, hubbard = _cast(param, dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qat = sample["q"].type(dtype)

        # variables to be differentiated
        gexp.requires_grad_(True)
        hubbard.requires_grad_(True)

        def func(gexp, hubbard):
            es = es2.ES2(hubbard=hubbard, average=average, gexp=gexp)
            cache = es.get_cache(numbers, positions)
            return es.get_energy(cache, qat)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, (gexp, hubbard))


def _cast(
    param: tuple[Tensor, AveragingFunction, Tensor], dtype: torch.dtype
) -> tuple[Tensor, AveragingFunction, Tensor]:
    gexp, average, hubbard = param
    return gexp.type(dtype), average, hubbard.type(dtype)
