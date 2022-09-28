"""
Run tests for the shell-resolved energy contribution from the
isotropic second-order electrostatic energy (ES2).
"""

from __future__ import annotations
from collections.abc import Generator
import pytest
import torch

import xtbml.coulomb.secondorder as es2
from xtbml.basis import IndexHelper
from xtbml.coulomb import AveragingFunction, averaging_function
from xtbml.exlibs.tbmalt import batch
from xtbml.param import (
    GFN1_XTB,
    get_elem_param,
    get_elem_angular,
)
from xtbml.typing import Tensor, Tuple

from .samples import mb16_43

sample_list = ["07", "08", "SiH4_shell"]
FixtureParams = Tuple[Tensor, AveragingFunction, dict]


@pytest.fixture(name="param", scope="class")
def fixture_param() -> Generator[FixtureParams, None, None]:
    if GFN1_XTB.charge is None:
        raise ValueError("No charge parameters provided.")

    gexp = torch.tensor(GFN1_XTB.charge.effective.gexp)
    average = averaging_function[GFN1_XTB.charge.effective.average]
    angular = get_elem_angular(GFN1_XTB.element)

    yield gexp, average, angular


class TestSecondOrderElectrostaticsShell:
    """Test the shell-resolved version of the ES2 contribution."""

    @classmethod
    def setup_class(cls):
        print(f"\n{cls.__name__}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name", sample_list)
    def test_mb16_43(self, param: FixtureParams, dtype: torch.dtype, name: str) -> None:
        """Test ES2 for some samples from mb16_43."""
        gexp, average, angular = _cast(param, dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qsh = sample["q"].type(dtype)
        ref = sample["es2"].type(dtype)
        ihelp = IndexHelper.from_numbers(numbers, angular)

        hubbard = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "gam",
            device=positions.device,
            dtype=positions.dtype,
        )
        lhubbard = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "lgam",
            device=positions.device,
            dtype=positions.dtype,
        )

        es = es2.ES2(
            positions=positions,
            hubbard=hubbard,
            lhubbard=lhubbard,
            average=average,
            gexp=gexp,
        )
        cache = es.get_cache(numbers, positions, ihelp)
        e = es.get_shell_energy(qsh, ihelp, cache)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name1", sample_list)
    @pytest.mark.parametrize("name2", sample_list)
    def test_batch(
        self, param: FixtureParams, dtype: torch.dtype, name1: str, name2: str
    ) -> None:
        gexp, average, angular = _cast(param, dtype)

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
        qsh = batch.pack(
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
        ihelp = IndexHelper.from_numbers(numbers, angular)

        hubbard = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "gam",
            device=positions.device,
            dtype=positions.dtype,
        )
        lhubbard = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "lgam",
            device=positions.device,
            dtype=positions.dtype,
        )

        es = es2.ES2(
            positions=positions,
            hubbard=hubbard,
            lhubbard=lhubbard,
            average=average,
            gexp=gexp,
        )
        cache = es.get_cache(numbers, positions, ihelp)
        e = es.get_shell_energy(qsh, ihelp, cache)
        assert torch.allclose(torch.sum(e, dim=-1), ref)

    @pytest.mark.grad
    @pytest.mark.parametrize("name", sample_list)
    def test_grad_positions(self, param: FixtureParams, name: str) -> None:
        dtype = torch.float64
        gexp, average, angular = _cast(param, dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qsh = sample["q"].type(dtype)
        ihelp = IndexHelper.from_numbers(numbers, angular)

        hubbard = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "gam",
            device=positions.device,
            dtype=positions.dtype,
        )
        lhubbard = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "lgam",
            device=positions.device,
            dtype=positions.dtype,
        )

        # variable to be differentiated
        positions.requires_grad_(True)

        def func(positions: Tensor):
            es = es2.ES2(
                positions=positions,
                hubbard=hubbard,
                lhubbard=lhubbard,
                average=average,
                gexp=gexp,
            )
            cache = es.get_cache(numbers, positions, ihelp)
            return es.get_shell_energy(qsh, ihelp, cache)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, positions)

    @pytest.mark.grad
    @pytest.mark.parametrize("name", sample_list)
    def test_grad_param(self, param: FixtureParams, name: str) -> None:
        dtype = torch.float64
        gexp, average, angular = _cast(param, dtype)

        sample = mb16_43[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        qsh = sample["q"].type(dtype)
        ihelp = IndexHelper.from_numbers(numbers, angular)

        hubbard = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "gam",
            device=positions.device,
            dtype=positions.dtype,
        )
        lhubbard = get_elem_param(
            torch.unique(numbers),
            GFN1_XTB.element,
            "lgam",
            device=positions.device,
            dtype=positions.dtype,
        )

        # variable to be differentiated
        gexp.requires_grad_(True)
        hubbard.requires_grad_(True)

        def func(gexp: Tensor, hubbard: Tensor):
            es = es2.ES2(
                positions=positions,
                hubbard=hubbard, lhubbard=lhubbard, average=average, gexp=gexp)
            cache = es.get_cache(numbers, positions, ihelp)
            return es.get_shell_energy(qsh, ihelp, cache)

        # pylint: disable=import-outside-toplevel
        from torch.autograd.gradcheck import gradcheck

        assert gradcheck(func, (gexp, hubbard))


def _cast(param: FixtureParams, dtype: torch.dtype) -> FixtureParams:
    gexp, average, angular = param
    return gexp.type(dtype), average, angular
