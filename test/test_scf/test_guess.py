"""
Test the SCF guess.
"""

from __future__ import annotations

import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.scf import guess

numbers = torch.tensor([6, 1])
positions = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0, 0], 6: [0, 1]})
charge = torch.tensor(0.0)


def test_fail() -> None:
    with pytest.raises(ValueError):
        guess.get_guess(numbers, positions, charge, ihelp, name="eht")

    # charges change because IndexHelper is broken
    with pytest.raises(RuntimeError):
        ih = IndexHelper.from_numbers_angular(numbers, {1: [0, 0], 6: [0, 1]})
        ih.orbitals_to_shell = torch.tensor([1, 2, 3])
        guess.get_guess(numbers, positions, charge, ih)


def test_eeq() -> None:
    c = guess.get_guess(numbers, positions, charge, ihelp)
    ref = torch.tensor(
        [
            -0.11593066900969,
            -0.03864355757833,
            -0.03864355757833,
            -0.03864355757833,
            +0.11593066900969,
            +0.11593066900969,
        ]
    )

    assert pytest.approx(ref, abs=1e-5) == c


def test_sad() -> None:
    c = guess.get_guess(numbers, positions, charge, ihelp, name="sad")
    size = int(ihelp.orbitals_per_shell.sum().item())

    assert pytest.approx(torch.zeros(size)) == c
