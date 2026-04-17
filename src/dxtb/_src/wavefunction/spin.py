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


def magnet_to_updown_1(x: Tensor) -> Tensor:
    """In-place conversion: charge/magnetization → up/down, 1D."""
    if x.shape[0] != 2:
        raise ValueError("Length must be 2.")
    x[0] = 0.5 * (x[0] + x[1])
    x[1] = x[0] - x[1]
    return x


def magnet_to_updown_2(x: Tensor) -> Tensor:
    """In-place conversion: charge/magnetization → up/down, 2D (..., 2, n_shells)."""
    if x.shape[-2] != 2:
        raise ValueError("Second-to-last dimension must be 2.")
    x[..., 0, :] = 0.5 * (x[..., 0, :] + x[..., 1, :])
    x[..., 1, :] = x[..., 0, :] - x[..., 1, :]
    return x


def magnet_to_updown_3(x: Tensor) -> Tensor:
    """In-place conversion: charge/magnetization → up/down, 3D (..., :, :, 2)."""
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2.")
    x[..., 0] = 0.5 * (x[..., 0] + x[..., 1])
    x[..., 1] = x[..., 0] - x[..., 1]
    return x


def magnet_to_updown_4(x: Tensor) -> Tensor:
    """In-place conversion: charge/magnetization → up/down, 4D (..., :, :, :, 2)."""
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2.")
    x[..., 0] = 0.5 * (x[..., 0] + x[..., 1])
    x[..., 1] = x[..., 0] - x[..., 1]
    return x


# === updown_to_magnet_* ===


def updown_to_magnet_1(x: Tensor) -> Tensor:
    """In-place conversion: up/down → charge/magnetization, 1D."""
    if x.shape[0] != 2:
        raise ValueError("Length must be 2.")
    x[0] = x[0] + x[1]
    x[1] = x[0] - 2.0 * x[1]
    return x


def updown_to_magnet_2(x: Tensor) -> Tensor:
    """In-place conversion: up/down → charge/magnetization, 2D (..., 2, n_shells)."""
    if x.shape[-2] != 2:
        raise ValueError("Second-to-last dimension must be 2.")
    x[..., 0, :] = x[..., 0, :] + x[..., 1, :]
    x[..., 1, :] = x[..., 0, :] - 2.0 * x[..., 1, :]
    return x


def updown_to_magnet_3(x: Tensor) -> Tensor:
    """In-place conversion: up/down → charge/magnetization, 3D (..., :, :, 2)."""
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2.")
    x[..., 0] = x[..., 0] + x[..., 1]
    x[..., 1] = x[..., 0] - 2.0 * x[..., 1]
    return x


def updown_to_magnet_4(x: Tensor) -> Tensor:
    """In-place conversion: up/down → charge/magnetization, 4D (..., :, :, :, 2)."""
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2.")
    x[..., 0] = x[..., 0] + x[..., 1]
    x[..., 1] = x[..., 0] - 2.0 * x[..., 1]
    return x


# === General Forumlas ===


def magnet_to_updown(x: Tensor) -> Tensor:
    """In-place conversion: charge/magnetization → up/down.
    Works for any tensor where exactly one dimension has length 2."""
    # locate spin dimension
    dims = [i for i, s in enumerate(x.shape) if s == 2]
    if not dims:
        raise ValueError("No dimension of length 2 found.")
    dim = dims[-1]

    # compute slices and update sequentially
    a = x.select(dim, 0)
    b = x.select(dim, 1)
    a.copy_(0.5 * (a + b))
    b.copy_(a - b)
    return x


def updown_to_magnet(x: Tensor) -> Tensor:
    """In-place conversion: up/down → charge/magnetization.
    Works for any tensor where exactly one dimension has length 2."""
    dims = [i for i, s in enumerate(x.shape) if s == 2]
    if not dims:
        raise ValueError("No dimension of length 2 found.")
    dim = dims[-1]

    a = x.select(dim, 0)
    b = x.select(dim, 1)
    a.add_(b)
    b.copy_(a - 2.0 * b)
    return x
