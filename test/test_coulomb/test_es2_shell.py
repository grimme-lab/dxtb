"""
Run tests for the shell-resolved energy contribution from the
isotropic second-order electrostatic energy (ES2).
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.basis import IndexHelper
from dxtb.coulomb import averaging_function
from dxtb.coulomb import secondorder as es2
from dxtb.param import GFN1_XTB, get_elem_angular, get_elem_param
from dxtb.utils import batch

from ..utils import dgradcheck
from .samples import samples

sample_list = ["MB16_43_07", "MB16_43_08", "SiH4"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    """Test ES2 for some samples from samples."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    qsh = sample["q"].to(**dd)
    ref = sample["es2"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, GFN1_XTB, **dd)
    assert es is not None

    cache = es.get_cache(numbers, positions, ihelp)
    e = es.get_shell_energy(qsh, cache)

    assert pytest.approx(ref, abs=tol, rel=tol) == torch.sum(e, dim=-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
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
    qsh = batch.pack(
        (
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        )
    )
    ref = torch.stack(
        [
            sample1["es2"].to(**dd),
            sample2["es2"].to(**dd),
        ],
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, GFN1_XTB, **dd)
    assert es is not None

    cache = es.get_cache(numbers, positions, ihelp)
    e = es.get_shell_energy(qsh, cache)

    assert pytest.approx(ref, abs=tol, rel=tol) == torch.sum(e, dim=-1)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_positions(name: str) -> None:
    dd: DD = {"device": device, "dtype": torch.double}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd).detach()
    qsh = sample["q"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))

    # variable to be differentiated
    positions.requires_grad_(True)

    def func(positions: Tensor):
        es = es2.new_es2(numbers, GFN1_XTB, shell_resolved=False, **dd)
        if es is None:
            assert False

        cache = es.get_cache(numbers, positions, ihelp)
        return es.get_shell_energy(qsh, cache)

    assert dgradcheck(func, positions)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_param(name: str) -> None:
    dd: DD = {"device": device, "dtype": torch.double}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    qsh = sample["q"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))

    assert GFN1_XTB.charge is not None

    hubbard = get_elem_param(torch.unique(numbers), GFN1_XTB.element, "gam", **dd)
    lhubbard = get_elem_param(torch.unique(numbers), GFN1_XTB.element, "lgam", **dd)
    gexp = torch.tensor(GFN1_XTB.charge.effective.gexp, **dd)
    average = averaging_function[GFN1_XTB.charge.effective.average]

    # variable to be differentiated
    gexp.requires_grad_(True)
    hubbard.requires_grad_(True)

    def func(gexp: Tensor, hubbard: Tensor):
        es = es2.ES2(hubbard, lhubbard, average=average, gexp=gexp, **dd)
        cache = es.get_cache(numbers, positions, ihelp)
        return es.get_shell_energy(qsh, cache)

    assert dgradcheck(func, (gexp, hubbard))
