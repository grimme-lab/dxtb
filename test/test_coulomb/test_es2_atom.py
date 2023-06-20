"""
Run tests for energy contribution from isotropic second-order
electrostatic energy (ES2).
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import Tensor
from dxtb.basis import IndexHelper
from dxtb.coulomb import averaging_function
from dxtb.coulomb import secondorder as es2
from dxtb.param import GFN1_XTB, get_elem_angular, get_elem_param
from dxtb.utils import batch

from .samples import samples

sample_list = ["MB16_43_01", "MB16_43_02", "SiH4_atom"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    """Test ES2 for some samples from MB16_43."""
    tol = sqrt(torch.finfo(dtype).eps)
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    qat = sample["q"].type(dtype)
    ref = sample["es2"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, GFN1_XTB, shell_resolved=False, **dd)
    assert es is not None

    cache = es.get_cache(numbers, positions, ihelp)
    e = es.get_atom_energy(qat, cache)
    assert pytest.approx(ref, abs=tol, rel=tol) == torch.sum(e, dim=-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)
    dd = {"dtype": dtype}

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

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, GFN1_XTB, shell_resolved=False, **dd)
    assert es is not None

    cache = es.get_cache(numbers, positions, ihelp)
    e = es.get_atom_energy(qat, cache)
    assert pytest.approx(ref, abs=tol, rel=tol) == torch.sum(e, dim=-1)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_positions(name: str) -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach()
    qat = sample["q"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))

    es = es2.new_es2(numbers, GFN1_XTB, shell_resolved=False, **dd)
    assert es is not None

    # variable to be differentiated
    positions.requires_grad_(True)

    def func(positions: Tensor):
        cache = es.get_cache(numbers, positions, ihelp)
        return es.get_atom_energy(qat, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_param(name: str) -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    qat = sample["q"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))

    assert GFN1_XTB.charge is not None

    hubbard = get_elem_param(torch.unique(numbers), GFN1_XTB.element, "gam", **dd)
    gexp = torch.tensor(GFN1_XTB.charge.effective.gexp, **dd)
    average = averaging_function[GFN1_XTB.charge.effective.average]

    # variables to be differentiated
    gexp.requires_grad_(True)
    hubbard.requires_grad_(True)

    def func(gexp: Tensor, hubbard: Tensor):
        es = es2.ES2(hubbard, average=average, gexp=gexp, **dd)
        cache = es.get_cache(numbers, positions, ihelp)
        return es.get_atom_energy(qat, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, (gexp, hubbard))
