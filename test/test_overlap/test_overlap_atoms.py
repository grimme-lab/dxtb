"""
Run tests for overlap of atoms.
References calculated with tblite 0.3.0.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.integral import Overlap
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch

from ..utils import load_from_npz
from .samples import samples

ref_overlap = np.load("test/test_overlap/overlap.npz")


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "C", "Rn"])
def test_overlap_single(dtype: torch.dtype, name: str):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd = {"dtype": dtype}
    tol = 1e-05

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = load_from_npz(ref_overlap, name, dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    overlap = Overlap(numbers, par, ihelp, **dd)
    s = overlap.build(positions)

    assert pytest.approx(s, rel=tol, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", ["C", "Rn"])
@pytest.mark.parametrize("name2", ["C", "Rn"])
def test_overlap_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack((sample1["numbers"], sample2["numbers"]))
    positions = batch.pack(
        (sample1["positions"].type(dtype), sample2["positions"].type(dtype))
    )
    ref = batch.pack(
        (
            load_from_npz(ref_overlap, name1, dtype),
            load_from_npz(ref_overlap, name2, dtype),
        )
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    overlap = Overlap(numbers, par, ihelp, **dd)
    s = overlap.build(positions)

    assert pytest.approx(s, abs=tol) == s.mT
    assert pytest.approx(s, abs=tol) == ref
