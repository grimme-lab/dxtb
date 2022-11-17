"""
Run tests for building the Hamiltonian matrix.
"""

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.ncoord import exp_count, get_coordination_number
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch
from dxtb.xtb import Hamiltonian

from ..utils import combinations as combis
from ..utils import load_from_npz
from .samples import samples

small = ["C", "Rn", "H2", "H2_nocn", "LiH", "HLi", "S2", "SiH4", "SiH4_nocn"]
large = ["PbH4-BiH3"]  # , "MB16_43_01"]  # , "LYS_xao"]

ref_h0 = np.load("test/test_hamiltonian/h0.npz")


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", small)
def test_single(dtype: torch.dtype, name: str) -> None:
    """
    Compare against reference calculated with tblite-int:
    - H2: fpm run -- H H 0,0,1.4050586229538 --bohr --hamiltonian --method gfn1 --cn 0.91396028097949444,0.91396028097949444
    - H2_nocn: fpm run -- H H 0,0,1.4050586229538 --bohr --hamiltonian --method gfn1
    - LiH: fpm run -- Li H 0,0,3.0159348779447 --bohr --hamiltonian --method gfn1 --cn 0.98684772035550494,0.98684772035550494
    - HLi: fpm run -- H Li 0,0,3.0159348779447 --bohr --hamiltonian --method gfn1 --cn 0.98684772035550494,0.98684772035550494
    - S2: fpm run -- S S 0,0,3.60562542949258 --bohr --hamiltonian --method gfn1 --cn 0.99889747382180494,0.99889747382180494
    - SiH4: tblite with "use dftd3_ncoord, only: get_coordination_number"
    - SiH4_nocn: tblite
    """
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = load_from_npz(ref_h0, name, dtype)

    if "nocn" in name:
        cn = None
    else:
        cn = get_coordination_number(numbers, positions, exp_count)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    o = h0.overlap(positions)
    assert torch.allclose(o, o.mT, atol=tol)

    h = h0.build(positions, o, cn=cn)
    assert torch.allclose(h, h.mT, atol=tol)
    assert torch.allclose(h, ref, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["C", "Rn", "H2", "LiH", "S2", "SiH4"])
@pytest.mark.parametrize("name2", ["C", "Rn", "H2", "LiH", "S2", "SiH4"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
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
            load_from_npz(ref_h0, name1, dtype),
            load_from_npz(ref_h0, name2, dtype),
        ),
    )

    cn = get_coordination_number(numbers, positions, exp_count)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    o = h0.overlap(positions)
    assert torch.allclose(o, o.mT, atol=tol)

    h = h0.build(positions, o, cn=cn)
    assert torch.allclose(h, h.mT, atol=tol)
    assert torch.allclose(h, ref, atol=tol, rtol=tol)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", large)
def test_large(dtype: torch.dtype, name: str) -> None:
    """Compare against reference calculated with tblite."""
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = load_from_npz(ref_h0, name, dtype)

    cn = get_coordination_number(numbers, positions, exp_count)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    o = h0.overlap(positions)
    assert torch.allclose(o, o.mT, atol=tol)

    h = h0.build(positions, o, cn=cn)
    assert torch.allclose(h, h.mT, atol=tol)
    assert torch.allclose(combis(h), combis(ref), atol=tol)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", large)
@pytest.mark.parametrize("name2", large)
def test_large_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
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
            load_from_npz(ref_h0, name1, dtype),
            load_from_npz(ref_h0, name2, dtype),
        )
    )

    cn = get_coordination_number(numbers, positions, exp_count)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, par, ihelp, **dd)

    o = h0.overlap(positions)
    assert torch.allclose(o, o.mT, atol=tol)

    h = h0.build(positions, o, cn)
    assert torch.allclose(h, h.mT, atol=tol)

    for _batch in range(numbers.shape[0]):
        assert torch.allclose(combis(h[_batch]), combis(ref[_batch]), atol=tol)
