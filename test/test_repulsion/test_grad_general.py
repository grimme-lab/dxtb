"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float`.)
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.basis import IndexHelper
from dxtb.classical import new_repulsion
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular

from .samples import samples

sample_list = ["H2O", "SiH4", "MB16_43_01", "MB16_43_02", "LYS_xao"]

device = None


@pytest.mark.grad
@pytest.mark.parametrize("name", ["H2O"])
def test_grad_fail(name: str) -> None:
    dtype = torch.double
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    cache = rep.get_cache(numbers, ihelp)
    energy = rep.get_energy(positions, cache)

    with pytest.raises(RuntimeError):
        rep.get_gradient(energy, positions)
