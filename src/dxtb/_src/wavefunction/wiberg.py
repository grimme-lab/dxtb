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
Wavefunction: Wiberg/Mayer Bond Orders
======================================

Wiberg (or better Mayer) bond orders are calculated from the off-diagonal
elements of the matrix product of the density and the overlap matrix.
"""

from __future__ import annotations

from dxtb import IndexHelper
from dxtb._src.typing import Tensor

__all__ = ["get_bond_order"]


def get_bond_order(overlap: Tensor, density: Tensor, ihelp: IndexHelper) -> Tensor:
    """
    Calculate Wiberg bond orders.

    Parameters
    ----------
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.
    ihelp : IndexHelper
        Helper class for indexing.

    Returns
    -------
    Tensor
        Wiberg bond orders.
    """

    # matrix product PS is not symmetric, since P and S do not commute.
    tmp = density @ overlap

    wbo = ihelp.reduce_orbital_to_atom(tmp * tmp.mT, dim=(-2, -1))
    wbo.diagonal(dim1=-2, dim2=-1).fill_(0.0)

    return wbo
