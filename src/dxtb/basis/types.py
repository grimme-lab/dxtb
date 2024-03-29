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
Data classes for basis construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from tad_mctc.typing import Tensor

__all__ = ["AtomCGTOBasis", "CGTOBasis"]


@dataclass
class CGTOBasis:
    angmom: int
    alphas: Tensor  # (nbasis,)
    coeffs: Tensor  # (nbasis,)
    normalized: bool = True

    def wfnormalize_(self) -> CGTOBasis:
        # will always be normalized already in dxtb because we have to also
        # include the orthonormalization of the H2s against the H1s
        return self


@dataclass
class AtomCGTOBasis:
    atomz: int | float | Tensor
    bases: list[CGTOBasis]
    pos: Tensor  # (ndim,)
