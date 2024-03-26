# pylint: disable=protected-access
"""
Run tests for gradient from isotropic second-order electrostatic energy (ES2).
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.basis import IndexHelper
from dxtb.components.interactions.coulomb import secondorder as es2
from dxtb.param import GFN1_XTB, get_elem_angular
from dxtb.utils import batch

from .samples import samples

sample_list = ["MB16_43_07", "MB16_43_08", "SiH4", "LiH"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    charges = sample["q"].to(**dd)
    ref = sample["grad"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, GFN1_XTB, shell_resolved=True, **dd)
    assert es is not None

    cache = es.get_cache(numbers, positions, ihelp)

    # atom gradient should be zero
    grad_atom = es._get_atom_gradient(numbers, positions, charges, cache)
    assert (torch.zeros_like(positions) == grad_atom).all()

    # analytical (old)
    grad = es._get_shell_gradient(numbers, positions, charges, cache, ihelp)
    assert pytest.approx(grad, abs=tol) == ref

    # numerical
    num_grad = calc_numerical_gradient(numbers, positions, ihelp, charges)
    assert pytest.approx(num_grad, abs=tol) == ref
    assert pytest.approx(num_grad, abs=tol) == grad

    # automatic
    positions.requires_grad_(True)
    mat = es.get_shell_coulomb_matrix(numbers, positions, ihelp)
    energy = 0.5 * mat * charges.unsqueeze(-1) * charges.unsqueeze(-2)
    (agrad,) = torch.autograd.grad(energy.sum(), positions)
    assert pytest.approx(ref, abs=tol) == agrad

    # analytical (automatic)
    cache = es.get_cache(numbers, positions, ihelp)  # recalc with gradients
    egrad = es.get_shell_gradient(charges, positions, cache)
    egrad.detach_()
    assert pytest.approx(ref, abs=tol) == egrad
    assert pytest.approx(egrad, abs=tol) == agrad

    positions.detach_()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    charges = batch.pack(
        (
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            sample1["grad"].to(**dd),
            sample2["grad"].to(**dd),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, GFN1_XTB, shell_resolved=True, **dd)
    assert es is not None

    # analytical (old)
    cache = es.get_cache(numbers, positions, ihelp)
    grad = es._get_shell_gradient(numbers, positions, charges, cache, ihelp)
    assert pytest.approx(grad, abs=tol) == ref

    # automatic
    positions.requires_grad_(True)
    mat = es.get_shell_coulomb_matrix(numbers, positions, ihelp)
    energy = 0.5 * mat * charges.unsqueeze(-1) * charges.unsqueeze(-2)
    (agrad,) = torch.autograd.grad(energy.sum(), positions)
    assert pytest.approx(ref, abs=tol) == agrad

    # analytical (automatic)
    cache = es.get_cache(numbers, positions, ihelp)  # recalc with gradients
    egrad = es.get_shell_gradient(charges, positions, cache)
    egrad.detach_()
    assert pytest.approx(ref, abs=tol) == egrad
    assert pytest.approx(egrad, abs=tol) == agrad

    positions.detach_()


def calc_numerical_gradient(
    numbers: Tensor, positions: Tensor, ihelp: IndexHelper, charges: Tensor
) -> Tensor:
    """Calculate gradient numerically for reference."""
    dtype = torch.double
    es = es2.new_es2(
        numbers,
        GFN1_XTB,
        shell_resolved=True,
        dtype=dtype,
        device=positions.device,
    )
    assert es is not None

    positions = positions.type(dtype)
    charges = charges.type(dtype)

    # setup numerical gradient
    gradient = torch.zeros_like(positions)
    step = 1.0e-6

    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            cache = es.get_cache(numbers, positions, ihelp)
            er = es.get_shell_energy(charges, cache)
            er = torch.sum(er, dim=-1)

            positions[i, j] -= 2 * step
            cache = es.get_cache(numbers, positions, ihelp)
            el = es.get_shell_energy(charges, cache)
            el = torch.sum(el, dim=-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
