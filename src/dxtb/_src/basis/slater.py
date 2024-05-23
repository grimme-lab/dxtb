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
Basis: Slater Expansion
=======================

Expansion coefficients for Slater functions into primitive Gaussian functions
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import (
    CGTOAzimuthalQuantumNumberError,
    CGTOPrimitivesError,
    CGTOPrincipalQuantumNumberError,
    CGTOQuantumNumberError,
    CGTOSlaterExponentsError,
)

__all__ = ["slater_to_gauss"]

base = Path(__file__).parent / "sto-ng"
sto_ng = [
    torch.from_numpy(np.load(base / f"sto-{n}g.npy")).type(torch.double)
    for n in range(1, 7)
]


# Two over pi
top = 2.0 / math.pi

dfactorial = torch.tensor([1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0])
"""
Double factorial up to 7!! for normalization of the Gaussian basis functions.

See `OEIS A001147 <https://oeis.org/A001147>`__.
"""

MAX_PRINCIPAL = 6
"""Maximum principal quantum number."""

MAX_AZIMUTHAL = 4
"""Maximum azimuthal quantum number."""


def slater_to_gauss(
    ng: Tensor,
    n: Tensor,
    l: Tensor,
    zeta: Tensor,
    norm: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Expand Slater function in primitive gaussian functions.

    Parameters
    ----------
    ng : int
        Number of Gaussian functions for the expansion.
    n : int
        Principal quantum number of shell.
    l : int
        Azimuthal quantum number of shell.
    zeta : Tensor
        Exponent of Slater function to expand.
    norm : bool, optional
        Include normalization in contraction coefficients.
        Defaults to ``True``.

    Returns
    -------
    (Tensor, Tensor):
        Contraction coefficients of primitive gaussians, can contain
        normalization, and exponents of primitive gaussian functions.
    """
    if not 1 <= ng <= 6:
        raise CGTOPrimitivesError()
    if zeta <= 0:
        raise CGTOSlaterExponentsError()
    if l > MAX_AZIMUTHAL:
        raise CGTOAzimuthalQuantumNumberError(MAX_AZIMUTHAL)
    if n > MAX_PRINCIPAL:
        raise CGTOPrincipalQuantumNumberError(MAX_PRINCIPAL)
    if n <= l:  # l ∊ [n-1, n-2, ..., 1, 0]
        raise CGTOQuantumNumberError()

    # we have to use a little hack here,
    # if you pass n and l correctly, everything is fine
    # ityp: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
    #    n: 1 2 3 4 5 2 3 4 5  3  4  5  4  5  5  6  6
    #    l: 0 0 0 0 0 1 1 1 1  2  2  2  3  3  4  0  1
    itype = n + torch.tensor([0, 4, 7, 9, 10], device=zeta.device)[l] - 1
    if n == 6 and ng == 6:
        itype = 15 + l

    _coeff, _alpha = sto_ng[ng - 1][:, itype, :]

    alpha = _alpha.type(zeta.dtype).to(zeta.device) * zeta.unsqueeze(-1) ** 2
    coeff = _coeff.type(zeta.dtype).to(zeta.device)

    # normalize the gaussian if requested
    # <φ|φ> = (2i-1)!!(2j-1)!!(2k-1)!!/(4α)^(i+j+k) · sqrt(π/2α)³
    # N² = (4α)^(i+j+k)/((2i-1)!!(2j-1)!!(2k-1)!!)  · sqrt(2α/π)³
    # N = (4α)^((i+j+k)/2) / sqrt((2i-1)!!(2j-1)!!(2k-1)!!) · (2α/π)^(3/4)
    if norm:
        coeff = coeff * (
            (top * alpha) ** 0.75
            * torch.sqrt(4 * alpha) ** l
            / torch.sqrt(dfactorial[l])
        )

    return (alpha, coeff)
