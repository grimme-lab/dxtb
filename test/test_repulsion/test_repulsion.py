"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float32`.)
"""

from __future__ import annotations
from math import sqrt
import torch
import pytest

from xtbml.basis.indexhelper import IndexHelper
from xtbml.classical import Repulsion, new_repulsion
from xtbml.exlibs.tbmalt import batch
from xtbml.param import GFN1_XTB, get_elem_angular
from xtbml.typing import Tensor

from .samples import samples


class TestRepulsion:
    """Testing the calculation of repulsion energy and gradients."""

    cutoff: Tensor = torch.tensor(25.0)
    """Cutoff for repulsion calculation."""

    @classmethod
    def setup_class(cls):
        print(cls.__name__)

        if GFN1_XTB.repulsion is None:
            raise ValueError("GFN1-xTB repulsion not available")

    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("name", ["SiH4", "01", "02", "LYS_xao"])
    def test_single(self, dtype: torch.dtype, name: str) -> None:
        tol = sqrt(torch.finfo(dtype).eps) * 10

        sample = samples[name]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        cutoff = self.cutoff.type(dtype)
        ref = sample["gfn1"].type(dtype)

        rep = new_repulsion(numbers, positions, GFN1_XTB, cutoff)
        if rep is None:
            assert False

        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
        cache = rep.get_cache(numbers, ihelp)
        e = rep.get_energy(positions, cache)

        assert pytest.approx(ref, abs=tol) == e.sum(-1).item()

    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("name1", ["SiH4", "01", "02", "LYS_xao"])
    @pytest.mark.parametrize("name2", ["SiH4", "01", "02", "LYS_xao"])
    def test_batch(self, dtype: torch.dtype, name1: str, name2: str) -> None:
        tol = sqrt(torch.finfo(dtype).eps) * 10

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
                sample1["gfn1"].type(dtype),
                sample2["gfn1"].type(dtype),
            ],
        )
        cutoff = self.cutoff.type(dtype)

        rep = new_repulsion(numbers, positions, GFN1_XTB, cutoff)
        if rep is not None:
            ihelp = IndexHelper.from_numbers(
                numbers, get_elem_angular(GFN1_XTB.element)
            )
            cache = rep.get_cache(numbers, ihelp)
            e = rep.get_energy(positions, cache)

            assert pytest.approx(ref, abs=tol) == e.sum(-1)

    @pytest.mark.parametrize("dtype", [torch.float64])
    @pytest.mark.parametrize("name", ["SiH4", "01", "02", "03", "LYS_xao"])
    def test_grad(self, dtype: torch.dtype, name: str) -> None:
        sample = samples[name]

        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        cutoff = self.cutoff.type(dtype)

        rep = new_repulsion(numbers, positions, GFN1_XTB, cutoff)
        if rep is None:
            assert False

        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
        cache = rep.get_cache(numbers, ihelp)

        grad_analytical = rep.get_grad(positions, cache)
        grad_numerical = calc_numerical_gradient(positions, rep, cache)
        assert torch.allclose(grad_analytical, grad_numerical, rtol=1e-4, atol=1e-6)

    # @pytest.mark.grad
    # @pytest.mark.parametrize("dtype", [torch.float64])
    # def test_param_grad(self, dtype: torch.dtype):
    #     """
    #     Check a single analytical gradient of `RepulsionFactory.alpha` against
    #     numerical gradient from `torch.autograd.gradcheck`.

    #     Args:
    #         dtype (torch.dtype): Numerical precision.

    #     Note:
    #         Although `torch.float32` raises a warning that the gradient check
    #         without double precision will fail, it actually works here.
    #     """
    #     sample = samples["01"]

    #     numbers = sample["numbers"]
    #     positions = sample["positions"].type(dtype)

    #     def func(*_):
    #         repulsion = self.repulsion_gfn1(numbers, positions, True)
    #         return repulsion.get_engrad()

    #     repulsion = self.repulsion_gfn1(numbers, positions, True)
    #     param = (repulsion.alpha, repulsion.zeff, repulsion.kexp)

    #     # pylint: disable=import-outside-toplevel
    #     from torch.autograd.gradcheck import gradcheck

    #     assert gradcheck(func, param)

    # @pytest.mark.grad
    # @pytest.mark.parametrize("dtype", [torch.float64])
    # def test_param_grad_batch(self, dtype: torch.dtype):
    #     """
    #     Check batch analytical gradient against numerical gradient from
    #     `torch.autograd.gradcheck`.

    #     Args:
    #         dtype (torch.dtype): Numerical precision.

    #     Note:
    #         Although `torch.float32` raises a warning that the gradient check
    #         without double precision will fail, it actually works here.
    #     """

    #     sample1, sample2 = samples["01"], samples["SiH4"]
    #     numbers: Tensor = batch.pack(
    #         (
    #             sample1["numbers"],
    #             sample2["numbers"],
    #         )
    #     )
    #     positions: Tensor = batch.pack(
    #         (
    #             sample1["positions"].type(dtype),
    #             sample2["positions"].type(dtype),
    #         )
    #     )

    #     def func(*_):
    #         repulsion = self.repulsion_gfn1(numbers, positions, True)
    #         return repulsion.get_engrad()

    #     repulsion = self.repulsion_gfn1(numbers, positions, True)
    #     param = (repulsion.alpha, repulsion.zeff, repulsion.kexp)

    #     # pylint: disable=import-outside-toplevel
    #     from torch.autograd.gradcheck import gradcheck

    #     assert gradcheck(func, param)


def calc_numerical_gradient(
    positions: Tensor, rep: Repulsion, cache: "Repulsion.Cache"
) -> Tensor:
    """Calculate gradient numerically for reference."""

    n_atoms = positions.shape[0]

    # setup numerical gradient
    gradient = torch.zeros((n_atoms, 3), dtype=positions.dtype)
    step = 1.0e-6

    for i in range(n_atoms):
        for j in range(3):
            er, el = 0.0, 0.0

            positions[i, j] += step
            er = rep.get_energy(positions, cache)
            er = torch.sum(er, dim=-1)

            positions[i, j] -= 2 * step
            el = rep.get_energy(positions, cache)
            el = torch.sum(el, dim=-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
