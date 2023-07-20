"""
Test for Wiberg bond orders.
Reference values obtained with xTB 6.5.1.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.basis.indexhelper import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch
from dxtb.wavefunction import wiberg

from .samples import samples

sample_list = ["H2", "LiH", "SiH4"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    density = sample["density"].to(**dd)
    overlap = sample["overlap"].to(**dd)
    ref = sample["wiberg"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    wbo = wiberg.get_bond_order(overlap, density, ihelp)
    assert pytest.approx(ref, rel=1e-7, abs=tol) == wbo


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    density = batch.pack(
        (
            sample1["density"].to(**dd),
            sample2["density"].to(**dd),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].to(**dd),
            sample2["overlap"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            sample1["wiberg"].to(**dd),
            sample2["wiberg"].to(**dd),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    wbo = wiberg.get_bond_order(overlap, density, ihelp)
    assert pytest.approx(ref, rel=1e-7, abs=tol) == wbo
