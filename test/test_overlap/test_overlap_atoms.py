"""Run tests for overlap."""

import numpy as np
import pytest
import torch

from xtbml.basis import IndexHelper
from xtbml.param import GFN1_XTB as par
from xtbml.param import get_elem_angular
from xtbml.utils import batch, load_from_npz
from xtbml.xtb import Hamiltonian

from .samples import samples

ref_overlap = np.load("test/test_overlap/overlap.npz")


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "C", "Rn"])
def test_overlap_single(dtype: torch.dtype, name: str):
    """Overlap matrix for monoatomic molecule should be unity."""

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = load_from_npz(ref_overlap, name, dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    hamiltonian = Hamiltonian(numbers, positions, par, ihelp)
    overlap = hamiltonian.overlap()

    assert torch.allclose(overlap, ref, rtol=1e-05, atol=1e-05, equal_nan=False)


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", ["C", "Rn"])
@pytest.mark.parametrize("name2", ["C", "Rn"])
def test_overlap_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
    atol = 1e-06

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
    h0 = Hamiltonian(numbers, positions, par, ihelp)

    o = h0.overlap()
    assert torch.allclose(o, o.mT, atol=atol)
    assert torch.allclose(o, ref, atol=atol)
