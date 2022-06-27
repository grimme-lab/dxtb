"""Run tests for energy contribution from Coulomb interaction."""

import pytest
import torch
from typing import Generator, Tuple

from xtbml.coulomb.charge import get_second_order, get_hubbard_params
from xtbml.exlibs.tbmalt import batch
from xtbml.param.gfn1 import GFN1_XTB
from xtbml.typing import Tensor

from .samples import mb16_43


@pytest.fixture(scope="class")
def param() -> Generator[Tuple[Tensor, str, Tensor], None, None]:
    if GFN1_XTB.charge is None:
        raise ValueError("No charge parameters provided.")

    gexp = torch.tensor(GFN1_XTB.charge.effective.gexp)
    average = GFN1_XTB.charge.effective.average
    hubbard = get_hubbard_params(GFN1_XTB.element)

    yield gexp, average, hubbard

    # print("teardown")


class TestCoulomb:
    """Test the Coulomb contribution."""

    @classmethod
    def setup_class(cls):
        print(cls.__name__)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_mb16_43(
        self, dtype: torch.dtype, param: Tuple[Tensor, str, Tensor]
    ) -> None:
        """Test the Coulomb contribution for mb16_43."""
        gexp, average, hubbard = _cast(param, dtype)

        sample = mb16_43["01"]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qat = sample["qat"].type(dtype)
        ref = sample["gfn1"].type(dtype)

        e = get_second_order(numbers, positions, qat, hubbard, gexp)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_mb16_sih4(
        self, param: Tuple[Tensor, str, Tensor], dtype: torch.dtype
    ) -> None:
        """Test the Coulomb contribution for mb16_43."""
        gexp, average, hubbard = _cast(param, dtype)

        sample = mb16_43["SiH4"]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qat = sample["qat"].type(dtype)
        ref = sample["gfn1"].type(dtype)

        e = get_second_order(numbers, positions, qat, hubbard, gexp)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("name1", ["01", "SiH4"])
    @pytest.mark.parametrize("name2", ["SiH4", "01"])
    def test_batch(
        self,
        param: Tuple[Tensor, str, Tensor],
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

        e = get_second_order(numbers, positions, qat, hubbard, gexp)
        assert torch.allclose(torch.sum(e, dim=-1), ref)


def _cast(
    param: Tuple[Tensor, str, Tensor], dtype: torch.dtype
) -> Tuple[Tensor, str, Tensor]:
    gexp, average, hubbard = param
    return gexp.type(dtype), average, hubbard.type(dtype)
