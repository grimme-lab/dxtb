"""
Test for Wiberg bond orders.
Reference values obtained with xTB 6.5.1.
"""

import pytest
import torch

from xtbml.exlibs.tbmalt import batch
from xtbml.basis.indexhelper import IndexHelper
from xtbml.param import GFN1_XTB as par
from xtbml.param import get_elem_angular
from xtbml.wavefunction import wiberg

from .samples import samples

sample_list = ["H2", "LiH", "SiH4"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str):
    sample = samples[name]
    numbers = sample["numbers"]
    density = sample["density"].type(dtype)
    overlap = sample["overlap"].type(dtype)
    ref = sample["wiberg"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    wbo = wiberg.get_bond_order(overlap, density, ihelp)
    assert torch.allclose(ref, wbo)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    density = batch.pack(
        (
            sample1["density"].type(dtype),
            sample2["density"].type(dtype),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].type(dtype),
            sample2["overlap"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample1["wiberg"].type(dtype),
            sample2["wiberg"].type(dtype),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    wbo = wiberg.get_bond_order(overlap, density, ihelp)
    assert torch.allclose(ref, wbo)
