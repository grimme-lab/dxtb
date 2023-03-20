"""
Run tests for energy contribution from halogen bond correction.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import Tensor
from dxtb.basis import IndexHelper
from dxtb.classical import Halogen, new_halogen
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular, get_elem_param

from .samples import samples


@pytest.mark.grad
@pytest.mark.parametrize("name", ["br2nh3", "br2och2", "LYS_xao"])
def test_grad_pos_gradcheck(name: str) -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)

    def func(positions: Tensor) -> Tensor:
        return xb.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("name", ["br2nh3", "finch", "LYS_xao"])
def test_grad_pos_autograd(name: str) -> None:
    dtype = torch.double
    dd = {"dtype": dtype}
    tol = 1e-7

    sample = samples[name]

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["gradient"]

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    gradient = xb.get_gradient(energy, positions)

    assert pytest.approx(ref, abs=tol) == gradient

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("sample_name", ["br2nh3", "br2och2", "tmpda"])
def test_grad_param(sample_name: str):
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples[sample_name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    if par.halogen is None:
        assert False

    _damp = torch.tensor(par.halogen.classical.damping, **dd, requires_grad=True)
    _rscale = torch.tensor(par.halogen.classical.rscale, **dd, requires_grad=True)
    _xbond = get_elem_param(
        torch.unique(numbers),
        par.element,
        "xbond",
        pad_val=0,
        **dd,
        requires_grad=True,
    )

    def func(damp: Tensor, rscale: Tensor, xbond: Tensor) -> Tensor:
        xb = Halogen(damp, rscale, xbond, **dd)
        cache = xb.get_cache(numbers, ihelp)
        return xb.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, (_damp, _rscale, _xbond))
