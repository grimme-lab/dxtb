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
Shared shell normalization
==========================

Exact renormalization of a contracted shell to unit self-overlap, computed
with the same ``compute_1d``-contract kernel that assembles the integrals.

``assemble_matrix`` applies it by default (``normalize=True``), folded into
the contraction coefficients. ``Basis.create_cgtos`` already normalizes every
contracted shell exactly, so for the built-in bases this changes the result
only at the level of rounding errors. It keeps the integrals exactly
normalized when the exponents or coefficients are modified after
``create_cgtos``, e.g., while fitting basis parameters.
"""

from __future__ import annotations

import torch
from tad_mctc import storch

from dxtb._src.typing import Tensor

from .pipeline import Kernel1D, assemble_overlap_1d

__all__ = ["shell_self_overlap_norm"]


def shell_self_overlap_norm(
    kernel: Kernel1D, la: int, alpha: Tensor, coeff: Tensor
) -> Tensor:
    """
    Per-orbital normalization factor ``1 / sqrt(self-overlap)`` for one
    contracted shell, computed by assembling that shell's overlap with
    itself at zero displacement.

    Parameters
    ----------
    kernel : Kernel1D
        A ``compute_1d``-contract kernel (MD's or OS's).
    la : int
        Angular momentum of the shell.
    alpha : Tensor
        Primitive Gaussian exponents of the shell, shape ``(nprim,)``.
    coeff : Tensor
        Contraction coefficients of the shell, shape ``(nprim,)``.

    Returns
    -------
    Tensor
        Normalization factor, shape ``(2 * la + 1,)`` (one value per
        spherical-harmonic component of the shell; in practice these are
        numerically identical for a given shell, but returned per-component
        rather than assumed uniform).
    """
    vec0 = torch.zeros((1, 3), dtype=alpha.dtype, device=alpha.device)
    self_overlap = assemble_overlap_1d(
        kernel, (la, la), (alpha, alpha), (coeff, coeff), vec0
    )
    diag = torch.diagonal(self_overlap[0])
    return storch.reciprocal(storch.sqrt(diag))
