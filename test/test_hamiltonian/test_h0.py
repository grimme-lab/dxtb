"""
Run tests for building the Hamiltonian matrix.
References calculated with tblite 0.3.0.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.basis import IndexHelper
from dxtb.integral.driver.pytorch import IntDriverPytorch as IntDriver
from dxtb.integral.driver.pytorch import OverlapPytorch as Overlap
from dxtb.ncoord import exp_count, get_coordination_number
from dxtb.param import GFN1_XTB, Param, get_elem_angular
from dxtb.utils import batch
from dxtb.xtb import GFN1Hamiltonian as Hamiltonian

from ..utils import load_from_npz
from .samples import samples

small = ["C", "Rn", "H2", "LiH", "HLi", "S2", "SiH4"]
large = ["PbH4-BiH3", "LYS_xao"]

ref_h0 = np.load("test/test_hamiltonian/h0.npz")

device = None


def run(numbers: Tensor, positions: Tensor, par: Param, ref: Tensor, dd: DD) -> None:
    tol = sqrt(torch.finfo(dd["dtype"]).eps) * 10

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    driver = IntDriver(numbers, par, ihelp, **dd)
    overlap = Overlap(**dd)
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    driver.setup(positions)
    s = overlap.build(driver)

    cn = get_coordination_number(numbers, positions, exp_count)
    h = h0.build(positions, s, cn=cn)
    assert pytest.approx(h, abs=tol) == h.mT
    assert pytest.approx(h, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", small)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_h0, name, dtype)

    run(numbers, positions, GFN1_XTB, ref, dd)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["C", "Rn", "H2", "LiH", "S2", "SiH4"])
@pytest.mark.parametrize("name2", ["C", "Rn", "H2", "LiH", "S2", "SiH4"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

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
    ref = batch.pack(
        (
            load_from_npz(ref_h0, name1, dtype),
            load_from_npz(ref_h0, name2, dtype),
        ),
    )

    run(numbers, positions, GFN1_XTB, ref, dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", large)
def test_large(dtype: torch.dtype, name: str) -> None:
    """Compare against reference calculated with tblite."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_h0, name, dtype)

    run(numbers, positions, GFN1_XTB, ref, dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", large)
@pytest.mark.parametrize("name2", large)
def test_large_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
    dd: DD = {"device": device, "dtype": dtype}

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
    ref = batch.pack(
        (
            load_from_npz(ref_h0, name1, dtype),
            load_from_npz(ref_h0, name2, dtype),
        )
    )

    run(numbers, positions, GFN1_XTB, ref, dd)
