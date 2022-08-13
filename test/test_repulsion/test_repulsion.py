"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float32`.)
"""

from __future__ import annotations
from math import sqrt
import torch
import pytest

from xtbml.basis.indexhelper import IndexHelper
from xtbml.classical.repulsion import new_repulsion
from xtbml.exlibs.tbmalt import batch
from xtbml.param.gfn1 import GFN1_XTB
from xtbml.param import get_elem_angular
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

    def base_test(
        self,
        numbers: Tensor,
        positions: Tensor,
        repulsion_factory,
        reference_energy: Tensor | None,
        reference_gradient: Tensor | None,
        dtype: torch.dtype,
    ) -> None:
        """Wrapper for testing versus reference energy and gradient"""

        atol = 1.0e-6 if dtype == torch.float32 else 1.0e-8
        rtol = 1.0e-4 if dtype == torch.float32 else 1.0e-5

        # factory to produce repulsion objects
        repulsion = repulsion_factory(numbers, positions)
        e, gradient = repulsion.get_engrad(calc_gradient=True)
        energy = torch.sum(e, dim=-1)

        # test against reference values
        if reference_energy is not None:
            assert torch.allclose(
                energy,
                reference_energy,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
        if reference_gradient is not None:
            assert torch.allclose(
                gradient,
                reference_gradient,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )

    def calc_numerical_gradient(
        self,
        repulsion,
        dtype: torch.dtype,
    ) -> Tensor:
        """Calculate gradient numerically for reference."""

        n_atoms = repulsion.positions.shape[0]

        # numerical gradient
        gradient = torch.zeros(n_atoms, 3, dtype=dtype)
        step = 1.0e-6

        for i in range(n_atoms):
            for j in range(3):
                er, el = 0.0, 0.0
                repulsion.positions[i, j] += step
                er = repulsion.get_engrad(calc_gradient=False)
                er = torch.sum(er, dim=-1)
                repulsion.positions[i, j] -= 2 * step
                el = repulsion.get_engrad(calc_gradient=False)
                el = torch.sum(el, dim=-1)
                repulsion.positions[i, j] += step
                gradient[i, j] = 0.5 * (er - el) / step

        return gradient

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
        if rep is not None:
            ihelp = IndexHelper.from_numbers(
                numbers, get_elem_angular(GFN1_XTB.element)
            )
            cache = rep.get_cache(numbers, positions, ihelp)
            e = rep.get_energy(cache)

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
            cache = rep.get_cache(numbers, positions, ihelp)
            e = rep.get_energy(cache)

            assert pytest.approx(ref, abs=tol) == e.sum(-1)

    # @pytest.mark.parametrize("dtype", [torch.float64])
    # def test_gfn1_batch_grad(self, dtype: torch.dtype):
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

    #     # get reference gradients by just looping over geometries
    #     grads = []
    #     for number, position in zip(numbers, positions):
    #         repulsion = self.repulsion_gfn1(number, position)
    #         grad = self.calc_numerical_gradient(repulsion, dtype)
    #         grads.append(grad)

    #     numerical_gradient = batch.pack(grads)

    #     self.base_test(
    #         numbers=numbers,
    #         positions=positions,
    #         repulsion_factory=self.repulsion_gfn1,
    #         reference_energy=None,
    #         reference_gradient=numerical_gradient,
    #         dtype=dtype,
    #     )

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
