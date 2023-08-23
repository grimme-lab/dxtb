"""
Run tests for building the Hamiltonian matrix.
References calculated with tblite 0.3.0.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb._types import DD
from dxtb.basis import IndexHelper
from dxtb.integral import Overlap
from dxtb.ncoord import exp_count, get_coordination_number
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch
from dxtb.xtb import Hamiltonian

from ..utils import load_from_npz
from .samples import samples

small = ["C", "Rn", "H2", "LiH", "HLi", "S2", "SiH4"]
large = ["PbH4-BiH3", "LYS_xao"]

ref_h0 = np.load("test/test_hamiltonian/h0.npz")

device = None


@pytest.mark.parametrize("dtype", [torch.float])
# @pytest.mark.parametrize("name", small)
@pytest.mark.parametrize("name", ["CH4", "MB16_43_01", "C60", "vancoh2"])
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    # ref = load_from_npz(ref_h0, name, dtype)
    print(numbers.shape)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    overlap = Overlap(numbers, par, ihelp, **dd)
    s = overlap.build(positions)
    assert pytest.approx(s, abs=tol) == s.mT

    from dxtb.basis import Basis
    from dxtb.integral import libcint as intor
    from dxtb.integral import OverlapLibcint

    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)
    wrapper = intor.LibcintWrapper(atombases, ihelp)
    overlap = OverlapLibcint(numbers, par, ihelp, driver=wrapper, **dd)
    overlap.build()
    # cn = get_coordination_number(numbers, positions, exp_count)
    # if name == "C":
    #     cn = None
    # h = h0.build(positions, s, cn=cn)
    # assert pytest.approx(h, abs=tol) == h.mT
    # assert pytest.approx(h, abs=tol) == ref


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

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    overlap = Overlap(numbers, par, ihelp, **dd)
    s = overlap.build(positions)
    assert pytest.approx(s, abs=tol) == s.mT

    cn = get_coordination_number(numbers, positions, exp_count)
    h = h0.build(positions, s, cn=cn)
    assert pytest.approx(h, abs=tol) == h.mT
    assert pytest.approx(h, abs=tol) == ref


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", large)
def test_large(dtype: torch.dtype, name: str) -> None:
    """Compare against reference calculated with tblite."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_h0, name, dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    overlap = Overlap(numbers, par, ihelp, **dd)
    s = overlap.build(positions)
    assert pytest.approx(s, abs=tol) == s.mT

    cn = get_coordination_number(numbers, positions, exp_count)
    h = h0.build(positions, s, cn=cn)
    assert pytest.approx(h, abs=tol) == h.mT
    assert pytest.approx(h, abs=tol) == ref


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", large)
@pytest.mark.parametrize("name2", large)
def test_large_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
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
        )
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    overlap = Overlap(numbers, par, ihelp, **dd)
    s = overlap.build(positions)
    assert pytest.approx(s, abs=tol) == s.mT

    cn = get_coordination_number(numbers, positions, exp_count)
    h = h0.build(positions, s, cn=cn)
    assert pytest.approx(h, abs=tol) == h.mT
    assert pytest.approx(h, abs=tol) == ref
