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

from .utils import LIBCINT_DRIVER, PYTORCH_DRIVER

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
def test_fail_family(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}
    numbers = torch.tensor([1, 3], device=device)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    i = ints.Integrals(numbers, par, ihelp, driver=LIBCINT_DRIVER, **dd)

    # make sure the checks are turned on
    assert i.run_checks is True

    with pytest.raises(RuntimeError):
        i.overlap = ints.Overlap(PYTORCH_DRIVER, **dd)
    with pytest.raises(RuntimeError):
        i.dipole = ints.Dipole(PYTORCH_DRIVER, **dd)
    with pytest.raises(RuntimeError):
        i.quadrupole = ints.Quadrupole(PYTORCH_DRIVER, **dd)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail_pytorch_multipole(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}
    numbers = torch.tensor([1, 3], device=device)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    i = ints.Integrals(numbers, par, ihelp, driver=PYTORCH_DRIVER, **dd)

    # make sure the checks are turned on
    assert i.run_checks is True

    # incompatible driver
    with pytest.raises(RuntimeError):
        i.overlap = ints.Overlap(LIBCINT_DRIVER, **dd)

    # multipole moments not implemented with PyTorch
    with pytest.raises(NotImplementedError):
        i.dipole = ints.Dipole(PYTORCH_DRIVER, **dd)
    with pytest.raises(NotImplementedError):
        i.quadrupole = ints.Quadrupole(PYTORCH_DRIVER, **dd)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_hcore(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}
    numbers = torch.tensor([1, 3], device=device)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    i = ints.Integrals(numbers, par, ihelp, **dd)
    i.hcore = ints.Hamiltonian(numbers, par, ihelp, **dd)

    h = i.hcore
    assert h is not None
    assert h.integral.matrix is None
