"""
Test collection of classical contributions `ClassicalList`.
"""
from __future__ import annotations

import torch

from dxtb.basis import IndexHelper
from dxtb.classical import ClassicalList


def test_empty() -> None:
    clist = ClassicalList()
    assert len(clist.classicals) == 0

    numbers = torch.tensor([6, 1])
    positions = torch.tensor([[0, 0, 0], [0, 0, 1]])
    ihelp = IndexHelper.from_numbers(numbers, {1: [0, 0], 6: [0, 1]})

    c = clist.get_cache(numbers, ihelp)
    assert isinstance(c, ClassicalList.Cache)

    e = clist.get_energy(positions, c)
    assert "none" in e
    assert (e["none"] == torch.zeros(e["none"].shape)).all()

    g = clist.get_gradient(e, positions)
    assert "none" in e
    assert (g["none"] == torch.zeros(g["none"].shape)).all()
