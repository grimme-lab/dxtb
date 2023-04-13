"""
Run tests for energy contribution from halogen bond correction.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.classical import new_halogen
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["br2nh3", "br2nh2o", "br2och2", "finch"])
def test_small(dtype: torch.dtype, name: str) -> None:
    """
    Test the halogen bond correction for small molecules taken from
    the tblite test suite.
    """
    tol = sqrt(torch.finfo(dtype).eps)
    dd = {"dtype": dtype}

    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["energy"].type(dtype)

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert pytest.approx(ref, rel=tol, abs=tol) == torch.sum(energy)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["tmpda", "tmpda_mod"])
def test_large(dtype: torch.dtype, name: str) -> None:
    """
    TMPDA@XB-donor from S30L (15AB). Contains three iodine donors and two
    nitrogen acceptors. In the modified version, one I is replaced with
    Br and one O is added in order to obtain different donors and acceptors.
    """
    tol = sqrt(torch.finfo(dtype).eps)
    dd = {"dtype": dtype}

    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["energy"].type(dtype)

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert pytest.approx(ref, abs=tol, rel=tol) == torch.sum(energy)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_no_xb(dtype: torch.dtype) -> None:
    """Test system without halogen bonds."""
    tol = sqrt(torch.finfo(dtype).eps)
    dd = {"dtype": dtype}

    sample = samples["LYS_xao"]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["energy"].type(dtype)

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert pytest.approx(ref, abs=tol, rel=tol) == torch.sum(energy)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_beyond_cutoff(dtype: torch.dtype) -> None:
    dd = {"dtype": dtype}

    numbers = torch.tensor([7, 35])
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
        ]
    )

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert pytest.approx(0.0) == torch.sum(energy)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["br2nh3", "br2och2"])
@pytest.mark.parametrize("name2", ["finch", "tmpda"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)
    dd = {"dtype": dtype}

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
            sample1["energy"].type(dtype),
            sample2["energy"].type(dtype),
        ],
    )

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert pytest.approx(ref, abs=tol, rel=tol) == torch.sum(energy, dim=-1)
