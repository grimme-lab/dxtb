"""
Run tests for energy contribution from halogen bond correction.
"""

import pytest
import torch

from xtbml.basis import IndexHelper
from xtbml.classical import Halogen, new_halogen
from xtbml.param import GFN1_XTB as par
from xtbml.param import get_elem_angular, get_elem_param
from xtbml.typing import Tensor
from xtbml.utils import batch

from .samples import samples


def test_none() -> None:
    dummy = torch.tensor(0.0)
    _par = par.copy(deep=True)

    _par.halogen = None
    assert new_halogen(dummy, dummy, _par) is None

    del _par.halogen
    assert new_halogen(dummy, dummy, _par) is None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["br2nh3", "br2nh2o", "br2och2", "finch"])
def test_small(dtype: torch.dtype, name: str) -> None:
    """
    Test the halogen bond correction for small molecules taken from
    the tblite test suite.
    """

    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["energy"].type(dtype)

    xb = new_halogen(numbers, positions, par)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert torch.allclose(ref, torch.sum(energy))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["tmpda", "tmpda_mod"])
def test_large(dtype: torch.dtype, name: str) -> None:
    """
    TMPDA@XB-donor from S30L (15AB). Contains three iodine donors and two
    nitrogen acceptors. In the modified version, one I is replaced with
    Br and one O is added in order to obtain different donors and acceptors.
    """

    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["energy"].type(dtype)

    xb = new_halogen(numbers, positions, par)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert torch.allclose(ref, torch.sum(energy))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_no_xb(dtype: torch.dtype) -> None:
    """Test system without halogen bonds."""

    sample = samples["LYS_xao"]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["energy"].type(dtype)

    xb = new_halogen(numbers, positions, par)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert torch.allclose(ref, torch.sum(energy))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["br2nh3", "br2och2"])
@pytest.mark.parametrize("name2", ["finch", "tmpda"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:

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

    xb = new_halogen(numbers, positions, par)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert torch.allclose(ref, torch.sum(energy, dim=-1))


@pytest.mark.grad
@pytest.mark.parametrize("name", ["br2nh3", "br2och2"])
def test_grad_pos(name: str) -> None:
    dtype = torch.double

    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, positions, par)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)

    def func(positions: Tensor) -> Tensor:
        return xb.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)


@pytest.mark.grad
@pytest.mark.parametrize("sample_name", ["br2nh3", "br2och2", "tmpda"])
def test_grad_param(sample_name: str):
    dtype = torch.double

    sample = samples[sample_name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    if par.halogen is None:
        assert False

    _damp = torch.tensor(par.halogen.classical.damping, dtype=dtype, requires_grad=True)
    _rscale = torch.tensor(
        par.halogen.classical.rscale, dtype=dtype, requires_grad=True
    )
    _xbond = get_elem_param(
        torch.unique(numbers),
        par.element,
        "xbond",
        pad_val=0,
        dtype=dtype,
        requires_grad=True,
    )

    def func(damp: Tensor, rscale: Tensor, xbond: Tensor) -> Tensor:
        xb = Halogen(numbers, positions, damp, rscale, xbond)
        cache = xb.get_cache(numbers, ihelp)
        return xb.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, (_damp, _rscale, _xbond))
