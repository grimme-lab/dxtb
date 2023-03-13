"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float`.)
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.classical import new_repulsion
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch

from .samples import samples

sample_list = ["H2O", "SiH4", "MB16_43_01", "MB16_43_02", "LYS_xao"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["gfn1"].type(dtype)

    rep = new_repulsion(numbers, par, **dd)
    if rep is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache)

    assert pytest.approx(ref, abs=tol) == e.sum(-1).item()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd = {"dtype": dtype}
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

    rep = new_repulsion(numbers, par, **dd)
    if rep is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache)

    assert pytest.approx(ref, abs=tol) == e.sum(-1)
