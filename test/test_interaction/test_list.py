"""
Test InteractionList.
"""
from __future__ import annotations

import torch

from dxtb.basis import IndexHelper
from dxtb.interaction import InteractionList


def test_empty() -> None:
    ilist = InteractionList()
    assert len(ilist.interactions) == 0

    numbers = torch.tensor([6, 1])
    ihelp = IndexHelper.from_numbers(numbers, {1: [0, 0], 6: [0, 1]})
    orbital = ihelp.spread_atom_to_orbital(numbers)

    c = ilist.get_cache(numbers, numbers, ihelp)
    assert isinstance(c, InteractionList.Cache)

    ae = ilist.get_atom_energy(numbers, numbers)
    assert (ae == torch.zeros(ae.shape)).all()

    se = ilist.get_shell_energy(numbers, numbers)
    assert (se == torch.zeros(se.shape)).all()

    e = ilist.get_energy(orbital, numbers, ihelp)  # type: ignore
    assert (e == torch.zeros(e.shape)).all()

    ag = ilist.get_atom_gradient(numbers, numbers)
    assert (ag == torch.zeros(ag.shape)).all()

    sg = ilist.get_shell_gradient(numbers, numbers)
    assert (sg == torch.zeros(sg.shape)).all()

    g = ilist.get_gradient(numbers, e, e, e, ihelp)  # type: ignore
    assert (g == torch.zeros(g.shape)).all()
