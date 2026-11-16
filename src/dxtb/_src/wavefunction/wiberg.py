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

import torch

from dxtb import IndexHelper
from dxtb._src.typing import Tensor

from .spin import updown_to_charge_magnetization

__all__ = ["get_bond_order"]


def get_bond_order(
    overlap: Tensor,
    density: Tensor,
    ihelp: IndexHelper,
    *,
    nspin: int = 1,
) -> Tensor:
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
    nspin : int, optional
        Number of spin channels. For two channels, the alpha/beta density
        matrices are given in the third to last dimension. Defaults to ``1``.

    Returns
    -------
    Tensor
        Wiberg bond orders. For two spin channels, the bond orders are
        returned in charge/magnetization representation (shape:
        ``(..., 2, nat, nat)``).
    """

    def one_channel(channel_density: Tensor) -> Tensor:
        # PS is not symmetric because P and S do not commute.
        tmp = channel_density @ overlap
        wbo = ihelp.reduce_orbital_to_atom(tmp * tmp.mT, dim=(-2, -1))
        wbo.diagonal(dim1=-2, dim2=-1).fill_(0.0)
        return wbo

    if nspin == 1:
        return one_channel(density)

    # The unrestricted Mayer convention carries a factor of two per spin
    # block, preserving restricted/UHF parity for a closed shell.
    updown = 2.0 * torch.stack(
        tuple(one_channel(density.select(-3, channel)) for channel in range(2)),
        dim=-3,
    )
    return updown_to_charge_magnetization(updown, dim=-3)
