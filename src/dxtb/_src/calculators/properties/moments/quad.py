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
Properties: Quadrupole Moment
=============================

Analytical calculation of the traceless quadrupole moment.

This module serves more as a short-cut for the calculation in
:class:`~dxtb.calculator.Calculator`, hiding some implementation details.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb._src.typing import Tensor

__all__ = ["quadrupole"]


def quadrupole(qat: Tensor, dpat: Tensor, qpat: Tensor, positions: Tensor) -> Tensor:
    """
    Analytical calculation of traceless electric quadrupole moment.

    Parameters
    ----------
    qat : Tensor
        Atom-resolved monopolar charges.
    dpat : Tensor
        Atom-resolved dipolar charges.
    qpat : Tensor
        Atom-resolved quadrupolar charges.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

    Returns
    -------
    Tensor
        Traceless electric quadrupole moment.
    """
    # TODO: Shape checks

    if qpat.shape[-1] == 9:
        # (..., nat, 9) -> (..., nat, 3, 3)
        qpat = qpat.view(*qpat.shape[:-1], 3, 3)

        # trace: (..., nat, 3, 3) -> (..., nat)
        tr = 0.5 * einsum("...ii->...", qpat)

        qpat = torch.stack(
            [
                1.5 * qpat[..., 0, 0] - tr,  # xx
                3 * qpat[..., 1, 0],  # yx
                1.5 * qpat[..., 1, 1] - tr,  # yy
                3 * qpat[..., 2, 0],  # zx
                3 * qpat[..., 2, 1],  # zy
                1.5 * qpat[..., 2, 2] - tr,  # zz
            ],
            dim=-1,
        )

    # This incorporates the electric quadrupole contribution from the
    # nuclei: Q_ij = ∑_k Z_k r_ki r_kj
    vec = einsum("...ij,...i->...ij", positions, qat)

    # temporary
    pv2d = positions * (vec + 2 * dpat)

    # Compute the atomic contributions to molecular quadrupole moment
    cart = torch.empty(
        (*positions.shape[:-1], 6), device=positions.device, dtype=positions.dtype
    )
    cart[..., 0] = pv2d[..., 0]
    cart[..., 1] = (
        positions[..., 0] * (vec[..., 1] + dpat[..., 1])
        + dpat[..., 0] * positions[..., 1]
    )
    cart[..., 2] = pv2d[..., 1]
    cart[..., 3] = (
        positions[..., 0] * (vec[..., 2] + dpat[..., 2])
        + dpat[..., 0] * positions[..., 2]
    )
    cart[..., 4] = (
        positions[..., 1] * (vec[..., 2] + dpat[..., 2])
        + dpat[..., 1] * positions[..., 2]
    )
    cart[..., 5] = pv2d[..., 2]

    # Compute the trace and make the tensor traceless
    tr = 0.5 * (cart[..., 0] + cart[..., 2] + cart[..., 5])
    cart[..., 0] = 1.5 * cart[..., 0] - tr
    cart[..., 1] *= 3.0
    cart[..., 2] = 1.5 * cart[..., 2] - tr
    cart[..., 3] *= 3.0
    cart[..., 4] *= 3.0
    cart[..., 5] = 1.5 * cart[..., 5] - tr

    # sum up contributions
    return qpat.sum(dim=-2) + cart.sum(dim=-2)
