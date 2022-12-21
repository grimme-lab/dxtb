"""
Run tests for gradient from isotropic second-order electrostatic energy (ES2).
"""

from math import sqrt

import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.coulomb import secondorder as es2
from dxtb.param import GFN1_XTB, get_elem_angular
from dxtb._types import Tensor
from dxtb.utils import batch

from .samples import samples

sample_list = ["MB16_43_01", "MB16_43_02", "SiH4_atom"]


is_shell_resolved = False


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charges = sample["q"].type(dtype)
    ref = sample["grad"].type(dtype)

    if GFN1_XTB.charge is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, GFN1_XTB, shell_resolved=is_shell_resolved, **dd)
    if es is None:
        assert False

    cache = es.get_cache(numbers, positions, ihelp)
    grad = es.get_gradient(numbers, positions, ihelp, charges, cache)
    assert pytest.approx(grad, abs=tol) == ref

    num_grad = calc_numerical_gradient(numbers, positions, ihelp, charges)
    assert pytest.approx(num_grad, abs=tol) == ref
    assert pytest.approx(num_grad, abs=tol) == grad


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["SiH4_atom"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

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
    charges = batch.pack(
        (
            sample1["q"].type(dtype),
            sample2["q"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample1["grad"].type(dtype),
            sample2["grad"].type(dtype),
        ),
    )

    if GFN1_XTB.charge is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, GFN1_XTB, shell_resolved=is_shell_resolved, **dd)
    if es is None:
        assert False

    cache = es.get_cache(numbers, positions, ihelp)
    grad = es.get_gradient(numbers, positions, ihelp, charges, cache)
    assert pytest.approx(grad, abs=tol) == ref


def calc_numerical_gradient(
    numbers: Tensor, positions: Tensor, ihelp: IndexHelper, charges: Tensor
) -> Tensor:
    """Calculate gradient numerically for reference."""
    dtype = torch.double
    es = es2.new_es2(
        numbers,
        GFN1_XTB,
        shell_resolved=is_shell_resolved,
        dtype=dtype,
        device=positions.device,
    )
    if es is None:
        assert False
    positions = positions.type(dtype)
    charges = charges.type(dtype)

    # setup numerical gradient
    gradient = torch.zeros_like(positions)
    step = 1.0e-6

    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            cache = es.get_cache(numbers, positions, ihelp)
            er = es.get_atom_energy(charges, ihelp, cache)
            er = torch.sum(er, dim=-1)

            positions[i, j] -= 2 * step
            cache = es.get_cache(numbers, positions, ihelp)
            el = es.get_atom_energy(charges, ihelp, cache)
            el = torch.sum(el, dim=-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
