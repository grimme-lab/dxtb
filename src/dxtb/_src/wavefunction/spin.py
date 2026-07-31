# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Wavefunction: Spin Information Handling
=======================================

Provides conversion routines to change the representation
of spin-polarized densities.

Spin is represented as charge and magnetization density in the population
based properties, e.g. Mulliken partial charges, atomic dipole moments, ...,
and in up/down representation for orbital energies, occupation numbers, ...
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Tensor

__all__ = [
    "magnet_to_updown_1",
    "magnet_to_updown_2",
    "magnet_to_updown_3",
    "magnet_to_updown_4",
    "updown_to_magnet_1",
    "updown_to_magnet_2",
    "updown_to_magnet_3",
    "updown_to_magnet_4",
]

# === magnet_to_updown_* ===


def _magnet_to_updown(x: Tensor, dim: int) -> Tensor:
    charge = x.select(dim, 0)
    magnetization = x.select(dim, 1)
    return torch.stack(
        (
            0.5 * (charge + magnetization),
            0.5 * (charge - magnetization),
        ),
        dim=dim,
    )


def _updown_to_magnet(x: Tensor, dim: int) -> Tensor:
    up = x.select(dim, 0)
    down = x.select(dim, 1)
    return torch.stack((up + down, up - down), dim=dim)


def magnet_to_updown_1(x: Tensor) -> Tensor:
    """Convert charge/magnetization to up/down, 1D."""
    if x.shape[0] != 2:
        raise ValueError("Length must be 2.")
    return _magnet_to_updown(x, 0)


def magnet_to_updown_2(x: Tensor) -> Tensor:
    """Convert charge/magnetization to up/down, 2D (..., 2, n_shells)."""
    if x.shape[-2] != 2:
        raise ValueError("Second-to-last dimension must be 2.")
    return _magnet_to_updown(x, -2)


def magnet_to_updown_3(x: Tensor) -> Tensor:
    """Convert charge/magnetization to up/down, 3D (..., :, :, 2)."""
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2.")
    return _magnet_to_updown(x, -1)


def magnet_to_updown_4(x: Tensor) -> Tensor:
    """Convert charge/magnetization to up/down, 4D (..., :, :, :, 2)."""
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2.")
    return _magnet_to_updown(x, -1)


# === updown_to_magnet_* ===


def updown_to_magnet_1(x: Tensor) -> Tensor:
    """Convert up/down to charge/magnetization, 1D."""
    if x.shape[0] != 2:
        raise ValueError("Length must be 2.")
    return _updown_to_magnet(x, 0)


def updown_to_magnet_2(x: Tensor) -> Tensor:
    """Convert up/down to charge/magnetization, 2D (..., 2, n_shells)."""
    if x.shape[-2] != 2:
        raise ValueError("Second-to-last dimension must be 2.")
    return _updown_to_magnet(x, -2)


def updown_to_magnet_3(x: Tensor) -> Tensor:
    """Convert up/down to charge/magnetization, 3D (..., :, :, 2)."""
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2.")
    return _updown_to_magnet(x, -1)


def updown_to_magnet_4(x: Tensor) -> Tensor:
    """Convert up/down to charge/magnetization, 4D (..., :, :, :, 2)."""
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2.")
    return _updown_to_magnet(x, -1)


# === General Formulas ===


def magnet_to_updown(x: Tensor) -> Tensor:
    """Convert charge/magnetization to up/down along the last length-2 axis."""
    # locate spin dimension
    dims = [i for i, s in enumerate(x.shape) if s == 2]
    if not dims:
        raise ValueError("No dimension of length 2 found.")
    dim = dims[-1]

    return _magnet_to_updown(x, dim)


def updown_to_magnet(x: Tensor) -> Tensor:
    """Convert up/down to charge/magnetization along the last length-2 axis."""
    dims = [i for i, s in enumerate(x.shape) if s == 2]
    if not dims:
        raise ValueError("No dimension of length 2 found.")
    dim = dims[-1]

    return _updown_to_magnet(x, dim)
