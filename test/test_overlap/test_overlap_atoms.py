"""Run tests for overlap."""

import pytest
import torch

from xtbml.basis import IndexHelper
from xtbml.exlibs.tbmalt import batch
from xtbml.param.gfn1 import GFN1_XTB as par
from xtbml.param.util import get_elem_angular
from xtbml.xtb.h0 import Hamiltonian

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["C", "Rn"])
def test_overlap_single(dtype: torch.dtype, name: str):
    """Overlap matrix for monoatomic molecule should be unity."""

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["overlap"].type(dtype)

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
            sample1["overlap"].type(dtype),
            sample2["overlap"].type(dtype),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, positions, par, ihelp)

    o = h0.overlap()
    assert torch.allclose(o, o.mT, atol=atol)
    assert torch.allclose(o, ref, atol=atol)
