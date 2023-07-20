"""
Test the molecule representation.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import Tensor
from dxtb.mol import Mol

device = None


def test_fail() -> None:
    dummy = torch.randn(2)

    with pytest.raises(TypeError):
        Mol("wrong", dummy)  # type: ignore

    with pytest.raises(TypeError):
        Mol(dummy, "wrong")  # type: ignore

    with pytest.raises(TypeError):
        Mol(dummy, dummy, "wrong")  # type: ignore


def test_shape() -> None:
    numbers = torch.randn(5)
    positions = torch.randn((5, 3))

    # shape mismatch with positions
    with pytest.raises(RuntimeError):
        Mol(torch.randn(1), positions)

    # shape mismatch with numbers
    with pytest.raises(RuntimeError):
        Mol(numbers, torch.randn((4, 3)))

    # too many dimensions
    with pytest.raises(RuntimeError):
        Mol(torch.randn(1, 2, 3), positions)

    # too many dimensions
    with pytest.raises(RuntimeError):
        Mol(numbers, torch.randn(1, 2, 3, 4))


def test_setter() -> None:
    numbers = torch.randn(5)
    positions = torch.randn((5, 3))
    mol = Mol(numbers, positions)

    with pytest.raises(RuntimeError):
        mol.numbers = torch.randn(1)

    with pytest.raises(RuntimeError):
        mol.positions = torch.randn(1)


def test_charge() -> None:
    numbers = torch.randn(5)
    positions = torch.randn((5, 3))
    mol = Mol(numbers, positions, charge=1)

    assert isinstance(mol.charge, Tensor)
    assert mol.charge.dtype == torch.int64

    mol.charge = 1.0
    assert isinstance(mol.charge, Tensor)
    assert mol.charge.dtype == torch.float32

    with pytest.raises(TypeError):
        mol.charge = "1"  # type: ignore
