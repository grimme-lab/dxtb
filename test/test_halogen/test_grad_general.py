"""
Testing halogen bond correction gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.classical import new_halogen
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular

from .samples import samples


@pytest.mark.grad
@pytest.mark.parametrize("name", ["br2nh3"])
def test_grad_fail(name: str) -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)

    with pytest.raises(RuntimeError):
        xb.get_gradient(energy, positions)