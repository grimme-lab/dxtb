"""
Test the parametrization utility of the Hamiltonian.
"""

from __future__ import annotations

import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_param, get_elem_pqn, get_elem_valence, get_pair_param


def test_pair_param() -> None:
    numbers = [6, 1, 1, 1, 1]
    symbols = ["C", "H", "H", "H", "H"]

    assert par.hamiltonian is not None

    ref = torch.tensor(
        [
            [1.000000, 1.000000, 1.000000, 1.000000, 1.000000],
            [1.000000, 0.960000, 0.960000, 0.960000, 0.960000],
            [1.000000, 0.960000, 0.960000, 0.960000, 0.960000],
            [1.000000, 0.960000, 0.960000, 0.960000, 0.960000],
            [1.000000, 0.960000, 0.960000, 0.960000, 0.960000],
        ]
    )

    kpair = get_pair_param(numbers, par.hamiltonian.xtb.kpair)
    assert pytest.approx(ref) == kpair

    kpair = get_pair_param(symbols, par.hamiltonian.xtb.kpair)
    assert pytest.approx(ref) == kpair


def test_elem_param() -> None:
    numbers = torch.tensor([6, 1])

    with pytest.raises(KeyError):
        get_elem_param(numbers, par.element, key="wrongkey")

    _par = par.model_copy(deep=True)
    _par.element["H"].shpoly = "something"  # type: ignore
    with pytest.raises(ValueError):
        get_elem_param(numbers, _par.element, key="shpoly")


def test_elem_valence() -> None:
    numbers = torch.tensor([6, 1])

    _par = par.model_copy(deep=True)
    _par.element["H"].shells = ["5h"]
    with pytest.raises(ValueError):
        get_elem_valence(numbers, _par.element)


def test_elem_pqn() -> None:
    numbers = torch.tensor([6, 1, 0])
    pred = get_elem_pqn(numbers, par.element)
    ref = torch.tensor([2, 2, 1, 2, -1])

    assert ref.device == pred.device
    assert (ref == pred).all()
