"""Run tests for energy contribution from isotropic second-order electrostatic energy (ES2)."""

from __future__ import annotations
import pytest
import torch
from typing import Generator, Tuple

import xtbml.coulomb.secondorder as es2
from xtbml.coulomb import AveragingFunction
from xtbml.exlibs.tbmalt import batch
from xtbml.param import GFN1_XTB, get_element_param, get_elem_param_dict
from xtbml.typing import Tensor

from .samples import mb16_43


@pytest.fixture(scope="class")
def param() -> Generator[
    Tuple[Tensor, AveragingFunction, Tensor, dict[int, list]], None, None
]:
    if GFN1_XTB.charge is None:
        raise ValueError("No charge parameters provided.")

    gexp = torch.tensor(GFN1_XTB.charge.effective.gexp)
    hubbard = get_element_param(GFN1_XTB.element, "gam")
    lhubbard = get_elem_param_dict(GFN1_XTB.element, "lgam")

    if GFN1_XTB.charge.effective.average == "harmonic":
        from xtbml.coulomb.average import harmonic_average as average
    elif GFN1_XTB.charge.effective.average == "geometric":
        from xtbml.coulomb.average import geometric_average as average
    elif GFN1_XTB.charge.effective.average == "arithmetic":
        from xtbml.coulomb.average import arithmetic_average as average
    else:
        raise ValueError("Unknown average function.")

    yield gexp, average, hubbard, lhubbard


class TestSecondOrderElectrostatics:
    """Test the ES2 contribution."""

    @classmethod
    def setup_class(cls):
        print(f"\n{cls.__name__}")

    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("name", ["07"])
    def test_mb16_43(
        self,
        param: Tuple[Tensor, AveragingFunction, Tensor, dict[int, list]],
        dtype: torch.dtype,
        name: str,
    ) -> None:
        """Test ES2 for some samples from mb16_43."""
        gexp, average, hubbard, lhubbard = _cast(param, dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qat = sample["qat"].type(dtype)
        ref = sample["es2"].type(dtype)

        e = es2.get_energy(numbers, positions, qat, hubbard, average, gexp, lhubbard)
        print(e)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("name1", ["07"])
    @pytest.mark.parametrize("name2", ["07"])
    def test_batch(
        self,
        param: Tuple[Tensor, AveragingFunction, Tensor, dict[int, list]],
        dtype: torch.dtype,
        name1: str,
        name2: str,
    ) -> None:
        gexp, average, hubbard, lhubbard = _cast(param, dtype)

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
                sample1["qat"].type(dtype),
                sample2["qat"].type(dtype),
            )
        )
        ref = torch.stack(
            [
                sample1["es2"].type(dtype),
                sample2["es2"].type(dtype),
            ],
        )

        e = es2.get_energy(numbers, positions, qat, hubbard, average, gexp, lhubbard)
        assert torch.allclose(torch.sum(e, dim=-1), ref)


def _cast(
    param: Tuple[Tensor, AveragingFunction, Tensor, dict[int, list]], dtype: torch.dtype
) -> Tuple[Tensor, AveragingFunction, Tensor, dict[int, list]]:
    gexp, average, hubbard, lhubbard = param
    return gexp.type(dtype), average, hubbard.type(dtype), lhubbard
