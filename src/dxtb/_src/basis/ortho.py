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
Basis: Orthonormalization
=========================

Gram-Schmidt orthonormalization routines for contracted Gaussian basis
functions.
"""

from __future__ import annotations

import math

import torch

from dxtb._src.typing import Tensor

__all__ = ["orthogonalize", "gaussian_integral"]


def gaussian_integral(ai: Tensor, aj: Tensor, ci: Tensor, cj: Tensor) -> Tensor:
    """
    Integral over two Gaussians (overlap).

    Parameters
    ----------
    ai : Tensor
        Exponent of GTO i. Can also be a tensor of exponents corresponding to
        all primitive GTOs of GTO i.
    aj : Tensor
        Exponent of GTO j. Can also be a tensor of exponents corresponding to
        all primitive GTOs of GTO j.
    ci : Tensor
        Contraction coefficients for CGTO i.
    cj : Tensor
        Contraction coefficients for CGTO j.

    Returns
    -------
    Tensor
        Overlap (summed).
    """
    oij = 1.0 / (ai.unsqueeze(-1) + aj.unsqueeze(-2))
    kab = torch.sqrt(math.pi * oij) ** 3
    overlap = kab * ci.unsqueeze(-1) * cj.unsqueeze(-2)

    # OLD: loop-based version
    # overlap = ai.new_tensor(0.0)
    # for _ai, _ci in zip(ai, ci):
    #     for _aj, _cj in zip(aj, cj):
    #         eij = _ai + _aj
    #         oij = 1.0 / eij
    #         kab = torch.sqrt(math.pi * oij) ** 3
    #         overlap += _ci * _cj * kab

    return overlap.sum()


def orthogonalize(
    alpha: tuple[Tensor, Tensor], coeff: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    """
    Orthogonalize a contracted Gaussian basis function to an existing basis
    function using. The second basis function is orthonormalized against the
    first basis function.

    Parameters
    ----------
    alpha : (Tensor, Tensor)
        Primitive Gaussian exponents for the shell pair.
    coeff : (Tensor, Tensor)
        Contraction coefficients for the shell pair.

    Returns
    -------
    (Tensor, Tensor)
        Primitive Gaussian exponents and contraction coefficients for the
        orthonormalized basis function.
    """

    coeff_i, coeff_j = coeff
    alpha_i, alpha_j = alpha

    coeff_new = coeff_i.new_zeros(coeff_i.shape[-1] + coeff_j.shape[-1])
    alpha_new = alpha_i.new_zeros(alpha_i.shape[-1] + alpha_j.shape[-1])

    # Calculate overlap between basis functions
    overlap = gaussian_integral(alpha_i, alpha_j, coeff_i, coeff_j)

    # Create new basis function from the pair which is orthogonal to the first
    # basis function
    alpha_new[: alpha_j.shape[-1]], alpha_new[alpha_j.shape[-1] :] = alpha_j, alpha_i
    coeff_new[: coeff_j.shape[-1]], coeff_new[coeff_j.shape[-1] :] = (
        coeff_j,
        -overlap * coeff_i,
    )

    # Normalization of the new basis function might be off, calculate self overlap
    selfoverlap = gaussian_integral(alpha_new, alpha_new, coeff_new, coeff_new)
    coeff_new /= torch.sqrt(selfoverlap)

    return alpha_new, coeff_new
