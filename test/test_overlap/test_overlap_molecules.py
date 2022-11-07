"""Run tests for overlap."""

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch
from dxtb.xtb import Hamiltonian

from ..utils import combinations as combis
from ..utils import load_from_npz
from .samples import samples

ref_overlap = np.load("test/test_overlap/overlap.npz")

molecules = ["H2O", "CH4", "SiH4", "PbH4-BiH3"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", molecules)
def test_overlap_single(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = load_from_npz(ref_overlap, name, dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    o = h0.overlap(positions)
    assert torch.allclose(o, o.mT, atol=tol)
    assert torch.allclose(combis(o), combis(ref), atol=tol)


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", molecules)
@pytest.mark.parametrize("name2", molecules)
def test_overlap_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
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
    ref = batch.pack(
        (
            load_from_npz(ref_overlap, name1, dtype),
            load_from_npz(ref_overlap, name2, dtype),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    o = h0.overlap(positions)
    assert torch.allclose(o, o.mT, atol=tol)

    for _batch in range(numbers.shape[0]):
        assert torch.allclose(combis(o[_batch]), combis(ref[_batch]), atol=tol)
