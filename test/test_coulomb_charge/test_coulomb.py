"""Run tests for energy contribution from Coulomb interaction."""

import pytest
import torch
from typing import Generator, Tuple

from xtbml.coulomb.average import AveragingFunction
from xtbml.coulomb.charge import get_second_order, get_hubbard_params
from xtbml.exlibs.tbmalt import batch
from xtbml.param.gfn1 import GFN1_XTB
from xtbml.typing import Tensor

from .samples import mb16_43


@pytest.fixture(scope="class")
def param() -> Generator[Tuple[Tensor, AveragingFunction, Tensor], None, None]:
    if GFN1_XTB.charge is None:
        raise ValueError("No charge parameters provided.")

    gexp = torch.tensor(GFN1_XTB.charge.effective.gexp)
    hubbard = get_hubbard_params(GFN1_XTB.element)

    if GFN1_XTB.charge.effective.average == "harmonic":
        from xtbml.coulomb.average import harmonic_average as average
    elif GFN1_XTB.charge.effective.average == "geometric":
        from xtbml.coulomb.average import geometric_average as average
    elif GFN1_XTB.charge.effective.average == "arithmetic":
        from xtbml.coulomb.average import arithmetic_average as average
    else:
        raise ValueError("Unknown average function.")

    yield gexp, average, hubbard


class TestCoulomb:
    """Test the Coulomb contribution."""

    @classmethod
    def setup_class(cls):
        print(cls.__name__)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name", ["01", "02", "SiH4"])
    def test_mb16_43(
        self,
        param: Tuple[Tensor, AveragingFunction, Tensor],
        dtype: torch.dtype,
        name: str,
    ) -> None:
        """Test the Coulomb contribution for mb16_43."""
        gexp, average, hubbard = _cast(param, dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qat = sample["qat"].type(dtype)
        ref = sample["gfn1"].type(dtype)

        e = get_second_order(numbers, positions, qat, hubbard, average, gexp)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name1", ["01", "02", "SiH4"])
    @pytest.mark.parametrize("name2", ["01", "02", "SiH4"])
    def test_batch(
        self,
        param: Tuple[Tensor, AveragingFunction, Tensor],
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
                sample1["qat"].type(dtype),
                sample2["qat"].type(dtype),
            )
        )
        ref = torch.stack(
            [
                sample1["gfn1"].type(dtype),
                sample2["gfn1"].type(dtype),
            ],
        )

        e = get_second_order(numbers, positions, qat, hubbard, average, gexp)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.grad
    def test_grad_positions(
        self, param: Tuple[Tensor, AveragingFunction, Tensor]
    ) -> None:
        dtype = torch.float64
        gexp, average, hubbard = _cast(param, dtype)

        sample = mb16_43["SiH4"]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qat = sample["qat"].type(dtype)

        # variable to be differentiated
        positions.requires_grad_(True)

        def func(positions):
            return get_second_order(numbers, positions, qat, hubbard, average, gexp)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, positions)

    @pytest.mark.grad
    def test_grad_param(self, param: Tuple[Tensor, AveragingFunction, Tensor]) -> None:
        dtype = torch.float64
        gexp, average, hubbard = _cast(param, dtype)

        sample = mb16_43["SiH4"]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qat = sample["qat"].type(dtype)

        # variable to be differentiated
        gexp.requires_grad_(True)
        hubbard.requires_grad_(True)

        def func(gexp, hubbard):
            return get_second_order(numbers, positions, qat, hubbard, average, gexp)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, (gexp, hubbard))


def _cast(
    param: Tuple[Tensor, AveragingFunction, Tensor], dtype: torch.dtype
) -> Tuple[Tensor, AveragingFunction, Tensor]:
    gexp, average, hubbard = param
    return gexp.type(dtype), average, hubbard.type(dtype)
