"""
Run tests for gradient from isotropic second-order electrostatic energy (ES2).
"""

from math import sqrt

import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.coulomb import secondorder as es2
from dxtb.param import GFN1_XTB, get_elem_angular
from dxtb.typing import Tensor
from dxtb.utils import batch

from .samples import samples

sample_list = ["MB16_43_07", "MB16_43_08", "SiH4", "LiH"]


@pytest.mark.parametrize("name", sample_list)
def test_single(name: str) -> None:
    dtype = torch.double
    tol = sqrt(torch.finfo(dtype).eps)
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charges = sample["q"].type(dtype)
    ref = sample["grad"].type(dtype)

    if GFN1_XTB.charge is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, GFN1_XTB, shell_resolved=True, **dd)
    if es is None:
        assert False

    cache = es.get_cache(numbers, positions, ihelp)
    grad = es.get_gradient(numbers, positions, ihelp, charges, cache)

    num_grad = calc_numerical_gradient(es, numbers, positions, charges)

    assert pytest.approx(grad, abs=tol) == ref
    assert pytest.approx(num_grad, abs=tol) == ref
    assert pytest.approx(num_grad, abs=tol) == grad


def calc_numerical_gradient(
    es: es2.ES2, numbers: Tensor, positions: Tensor, charges: Tensor
) -> Tensor:
    """Calculate gradient numerically for reference."""

    n_atoms = positions.shape[0]
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))

    # setup numerical gradient
    gradient = torch.zeros((n_atoms, 3), dtype=positions.dtype)
    step = 1.0e-6

    for i in range(n_atoms):
        for j in range(3):
            positions[i, j] += step
            cache = es.get_cache(numbers, positions, ihelp)
            er = es.get_shell_energy(charges, ihelp, cache)
            er = torch.sum(er, dim=-1)

            positions[i, j] -= 2 * step
            cache = es.get_cache(numbers, positions, ihelp)
            el = es.get_shell_energy(charges, ihelp, cache)
            el = torch.sum(el, dim=-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
