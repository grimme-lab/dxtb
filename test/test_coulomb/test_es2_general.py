# pylint: disable=protected-access
"""
Run generic tests for energy contribution from isotropic second-order
electrostatic energy (ES2).
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.basis import IndexHelper
from dxtb.components.interactions import Charges
from dxtb.components.interactions.coulomb import secondorder as es2
from dxtb.param import GFN1_XTB

from ..utils import get_device_from_str


def test_none() -> None:
    dummy = torch.tensor(0.0)
    par = GFN1_XTB.model_copy(deep=True)

    par.charge = None
    assert es2.new_es2(dummy, par) is None

    del par.charge
    assert es2.new_es2(dummy, par) is None


def test_store_fail() -> None:
    numbers = torch.tensor([3, 1])
    positions = torch.randn((2, 3))
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    es = es2.new_es2(numbers, GFN1_XTB)
    assert es is not None

    cache = es.get_cache(numbers, positions, ihelp=ihelp)
    with pytest.raises(RuntimeError):
        cache.restore()


def test_grad_fail() -> None:
    numbers = torch.tensor([3, 1])
    positions = torch.randn((2, 3), requires_grad=True)
    charges = torch.randn(6)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    es = es2.new_es2(numbers, GFN1_XTB, **dd)
    assert es is not None

    cache = es.get_cache(numbers, positions, ihelp=ihelp)
    energy = es.get_energy(Charges(charges), cache, ihelp)

    # zeroenergy
    grad = es._gradient(torch.zeros_like(energy), positions)
    assert (torch.zeros_like(positions) == grad).all()

    with pytest.raises(RuntimeError):
        positions.requires_grad_(False)
        es._gradient(energy, positions)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    cls = cls.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        cls.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = get_device_from_str(device_str)
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    cls = cls.to(device)
    assert cls.device == device


def test_change_device_fail() -> None:
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.device = "cpu"


def test_fail_shell_resolved() -> None:
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB, shell_resolved=False)
    assert cls is not None

    numbers = torch.tensor([6, 1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0, 0], 6: [0, 1]})

    # shell-resolved function fails if initialzed with `shell_resolved=False`
    with pytest.raises(ValueError):
        cls.get_shell_coulomb_matrix(numbers, numbers, ihelp)


def test_zeros_atom_resolved() -> None:
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB, shell_resolved=False)
    assert cls is not None

    n = torch.tensor([6, 1])

    shell_energy = cls.get_shell_energy(n, n)  # type: ignore
    assert (shell_energy == torch.zeros_like(shell_energy)).all()

    shell_gradient = cls.get_shell_gradient(n, n, n)  # type: ignore
    assert (shell_gradient == torch.zeros_like(shell_gradient)).all()


def test_zeros_shell_resolved() -> None:
    cls = es2.new_es2(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    n = torch.tensor([6, 1])

    atom_energy = cls.get_atom_energy(n, n)  # type: ignore
    assert (atom_energy == torch.zeros_like(atom_energy)).all()

    atom_gradient = cls.get_atom_gradient(n, n, n, n)  # type: ignore
    assert (atom_gradient == torch.zeros_like(atom_gradient)).all()
