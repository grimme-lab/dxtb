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
Wavefunction: Spin
==================

Conversion between the alpha/beta (up/down) representation and the
charge/magnetization representation of spin-resolved quantities.

As in tblite, populations and potentials are handled in the
charge/magnetization representation, while orbital energies, occupations
and density matrices are given in the alpha/beta representation.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Tensor

__all__ = ["charge_magnetization_to_updown", "updown_to_charge_magnetization"]


def updown_to_charge_magnetization(x: Tensor, dim: int) -> Tensor:
    """
    Convert from alpha/beta to charge/magnetization representation, i.e.,
    ``(alpha, beta) -> (alpha + beta, alpha - beta)``.

    Parameters
    ----------
    x : Tensor
        Quantity in alpha/beta representation.
    dim : int
        Spin dimension (of size 2).

    Returns
    -------
    Tensor
        Quantity in charge/magnetization representation.
    """
    alpha, beta = torch.unbind(x, dim=dim)
    return torch.stack((alpha + beta, alpha - beta), dim=dim)


def charge_magnetization_to_updown(x: Tensor, dim: int) -> Tensor:
    """
    Convert from charge/magnetization to alpha/beta representation, i.e.,
    ``(charge, magnet) -> ((charge + magnet) / 2, (charge - magnet) / 2)``.

    Parameters
    ----------
    x : Tensor
        Quantity in charge/magnetization representation.
    dim : int
        Spin dimension (of size 2).

    Returns
    -------
    Tensor
        Quantity in alpha/beta representation.
    """
    charge, magnet = torch.unbind(x, dim=dim)
    return torch.stack(
        (0.5 * (charge + magnet), 0.5 * (charge - magnet)), dim=dim
    )
