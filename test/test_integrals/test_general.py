"""
Test overlap build from integral container.
"""
from __future__ import annotations

import pytest
import torch

from dxtb import integral as ints
from dxtb._types import DD
from dxtb.basis import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_empty(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}
    numbers = torch.tensor([1, 3], device=device)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    i = ints.Integrals(numbers, par, ihelp, **dd)

    assert i._hcore is None
    assert i._overlap is None
    assert i._dipole is None
    assert i._quadrupole is None

    assert i.hcore is None
    assert i.overlap is None
    assert i.dipole is None
    assert i.quadrupole is None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_hcore(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}
    numbers = torch.tensor([1, 3], device=device)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    i = ints.Integrals(numbers, par, ihelp, **dd)
    i.hcore = ints.Hamiltonian(numbers, par, ihelp, **dd)

    assert i.hcore is not None
    assert i.hcore.matrix is None
