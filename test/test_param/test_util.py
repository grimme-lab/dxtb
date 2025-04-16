# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test the parametrization utility of the Hamiltonian.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.data import pse

from dxtb import GFN1_XTB as par
from dxtb._src.param.element import Element
from dxtb._src.typing import Tensor, get_default_dtype
from dxtb._src.utils import is_int_list

from ..utils import get_elem_param


def get_pair_param(
    symbols: list[str] | list[int],
    par_pair: dict[str, float],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Obtain tensor of a pair-wise parametrized quantity for all pairs.

    Parameters
    ----------
    symbols : list[str | int]
        List of atomic symbols or atomic numbers.
    par_pair : dict[str, float]
        Parametrization of pairs.
    device : torch.device | None, optional
        Device to store the tensor. If ``None`` (default), the default device is used.
    dtype : torch.dtype | None, optional
        Data type of the tensor. If ``None`` (default), the data type is inferred.

    Returns
    -------
    Tensor
        Parametrization of all pairs of ``symbols``.
    """
    # convert atomic numbers to symbols
    if is_int_list(symbols):
        symbols = [pse.Z2S.get(i, "X") for i in symbols]

    if dtype is None:
        dtype = get_default_dtype()

    ndim = len(symbols)
    pair_mat = torch.ones(*(ndim, ndim), dtype=dtype, device=device)
    for i, isp in enumerate(symbols):
        for j, jsp in enumerate(symbols):
            # Watch format! ("element1-element2")
            pair_mat[i, j] = par_pair.get(
                f"{isp}-{jsp}", par_pair.get(f"{jsp}-{isp}", 1.0)
            )

    return pair_mat


def test_pair_param() -> None:
    """Test retrieving pair parameters."""
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
    assert pytest.approx(ref.cpu()) == kpair.cpu()

    kpair = get_pair_param(symbols, par.hamiltonian.xtb.kpair)
    assert pytest.approx(ref.cpu()) == kpair.cpu()


def test_elem_param() -> None:
    """Test retrieving element parameters."""
    numbers = torch.tensor([6, 1])

    with pytest.raises(KeyError):
        get_elem_param(numbers, par.element, key="wrongkey")

    _par = par.model_copy(deep=True)
    _par.element["H"].shpoly = "something"  # type: ignore
    with pytest.raises(ValueError):
        get_elem_param(numbers, _par.element, key="shpoly")


###############################################################################


def get_elem_valence(
    numbers: Tensor,
    par_element: dict[str, Element],
    pad_val: int = -1,
    device: torch.device | None = None,
) -> Tensor:
    """
    Obtain valence of the shells of all atoms.

    .. warning::

       Only works if shells of same angular momentum are consecutive,
       as in the GFN1-xTB hydrogen basis set.

    Obtain valence of the shells of all atoms.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    par_element : dict[str, Element]
        Parametrization of elements.
    pad_val : int, optional
        Value to pad the tensor with. Default is `-1`.
    device : torch.device | None
        Device to store the tensor. If ``None`` (default), the default device
        is used.

    Returns
    -------
    dict[int, list[int]]
        Valence of all elements.
    """
    l = []
    key = "shells"
    label2angular = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

    for number in numbers:
        el = pse.Z2S.get(int(number.item()), "X")
        shells = []
        if el in par_element:
            for shell in getattr(par_element[el], key):
                shell = shell[-1]
                if shell not in label2angular:
                    raise ValueError(f"Unknown shell type '{shell}'.")

                shells.append(label2angular[shell])

        else:
            shells = [pad_val]

        # https://stackoverflow.com/questions/62300404/how-can-i-zero-out-duplicate-values-in-each-row-of-a-pytorch-tensor
        r = torch.tensor(shells, dtype=torch.long, device=device)
        tmp = torch.ones(r.shape, dtype=torch.bool, device=device)

        if r.size(0) < 2:
            vals = tmp
        else:
            # Sorting the rows so that duplicate values appear together
            # e.g. [1, 2, 3, 3, 3, 4, 4]
            # `stable=False` gives different results on CPU and GPU
            y, idxs = torch.sort(r, stable=True)

            # subtracting, so duplicate values will become 0
            # e.g. [1, 2, 3, 0, 0, 4, 0]
            tmp[1:] = (y[1:] - y[:-1]) != 0

            # retrieving the original indices of elements
            _, idxs = torch.sort(idxs)

            # re-organizing the rows following original order
            # e.g. [1, 2, 3, 4, 0, 0, 0]
            vals = torch.gather(tmp, 0, idxs)

        for val in vals:
            l.append(val)

    return torch.stack(l)


def test_elem_valence() -> None:
    """Test retrieving valence."""
    numbers = torch.tensor([6, 1])

    _par = par.model_copy(deep=True)
    _par.element["H"].shells = ["5h"]
    with pytest.raises(ValueError):
        get_elem_valence(numbers, _par.element)


###############################################################################


def get_elem_angular(par_element: dict[str, Element]) -> dict[int, list[int]]:
    """
    Obtain angular momenta of the shells of all atoms.

    Parameters
    ----------
    par_element : dict[str, Element]
        Parametrization of elements.

    Returns
    -------
    dict[int, list[int]]
        Angular momenta of all elements.
    """
    label2angular = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

    return {
        pse.S2Z[sym]: [label2angular[label[-1]] for label in par.shells]
        for sym, par in par_element.items()
    }


def test_elem_angular() -> None:
    """Test retrieving angular momenta."""
    pred = get_elem_angular(par.element)
    ref = {
        1: [0, 0],
        2: [0],
        3: [0, 1],
        4: [0, 1],
        5: [0, 1],
        6: [0, 1],
        7: [0, 1],
        8: [0, 1],
        9: [0, 1],
        10: [0, 1, 2],
        11: [0, 1],
        12: [0, 1],
        13: [0, 1, 2],
        14: [0, 1, 2],
        15: [0, 1, 2],
        16: [0, 1, 2],
        17: [0, 1, 2],
        18: [0, 1, 2],
        19: [0, 1],
        20: [0, 1, 2],
        21: [2, 0, 1],
        22: [2, 0, 1],
        23: [2, 0, 1],
        24: [2, 0, 1],
        25: [2, 0, 1],
        26: [2, 0, 1],
        27: [2, 0, 1],
        28: [2, 0, 1],
        29: [2, 0, 1],
        30: [0, 1],
        31: [0, 1, 2],
        32: [0, 1, 2],
        33: [0, 1, 2],
        34: [0, 1, 2],
        35: [0, 1, 2],
        36: [0, 1, 2],
        37: [0, 1],
        38: [0, 1, 2],
        39: [2, 0, 1],
        40: [2, 0, 1],
        41: [2, 0, 1],
        42: [2, 0, 1],
        43: [2, 0, 1],
        44: [2, 0, 1],
        45: [2, 0, 1],
        46: [2, 0, 1],
        47: [2, 0, 1],
        48: [0, 1],
        49: [0, 1, 2],
        50: [0, 1, 2],
        51: [0, 1, 2],
        52: [0, 1, 2],
        53: [0, 1, 2],
        54: [0, 1, 2],
        55: [0, 1],
        56: [0, 1, 2],
        57: [2, 0, 1],
        58: [2, 0, 1],
        59: [2, 0, 1],
        60: [2, 0, 1],
        61: [2, 0, 1],
        62: [2, 0, 1],
        63: [2, 0, 1],
        64: [2, 0, 1],
        65: [2, 0, 1],
        66: [2, 0, 1],
        67: [2, 0, 1],
        68: [2, 0, 1],
        69: [2, 0, 1],
        70: [2, 0, 1],
        71: [2, 0, 1],
        72: [2, 0, 1],
        73: [2, 0, 1],
        74: [2, 0, 1],
        75: [2, 0, 1],
        76: [2, 0, 1],
        77: [2, 0, 1],
        78: [2, 0, 1],
        79: [2, 0, 1],
        80: [0, 1],
        81: [0, 1],
        82: [0, 1],
        83: [0, 1],
        84: [0, 1, 2],
        85: [0, 1, 2],
        86: [0, 1, 2],
    }

    assert ref == pred


###############################################################################


def get_elem_pqn(
    numbers: Tensor,
    par_element: dict[str, Element],
    pad_val: int = -1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Obtain principal quantum numbers of the shells of all atoms.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    par_element : dict[str, Element]
        Parametrization of elements.
    pad_val : int, optional
        Value to pad the tensor with. Default is `-1`.
    device : torch.device | None, optional
        Device to store the tensor. If ``None`` (default), the default device
        is used.
    dtype : torch.dtype | None, optional
        Data type of the tensor. If ``None`` (default), the data type is
        inferred.

    Returns
    -------
    Tensor
        Principal quantum numbers of the shells of all atoms.
    """
    key = "shells"

    shells = []
    for number in numbers:
        el = pse.Z2S.get(int(number.item()), "X")
        if el in par_element:
            for shell in getattr(par_element[el], key):
                shells.append(int(shell[0]))
        else:
            shells.append(pad_val)

    return torch.tensor(shells, device=device, dtype=dtype)


def test_elem_pqn() -> None:
    """Test retrieving principal quantum number."""
    numbers = torch.tensor([6, 1, 0])
    pred = get_elem_pqn(numbers, par.element)
    ref = torch.tensor([2, 2, 1, 2, -1])

    assert ref.device == pred.device
    assert (ref == pred).all()
