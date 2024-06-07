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
from __future__ import annotations

from tad_mctc.math import einsum

from dxtb._src.typing import Tensor

from ..timing.decorator import timer_decorator

__all__ = ["get_density"]


@timer_decorator("Density", "SCF")
def get_density(coeffs: Tensor, occ: Tensor, emo: Tensor | None = None) -> Tensor:
    """
    Calculate the density matrix from the coefficient vector and the occupation.

    Parameters
    ----------
    evecs : Tensor
        MO coefficients.
    occ : Tensor
        Occupation numbers (diagonal matrix).
    emo : Tensor | None, optional
        Orbital energies for energy weighted density matrix. Defaults to ``None``.

    Returns
    -------
    Tensor
        (Energy-weighted) Density matrix.
    """
    o = occ if emo is None else occ * emo

    # equivalent: coeffs * o.unsqueeze(-2) @ coeffs.mT
    return einsum("...ik,...k,...jk->...ij", coeffs, o, coeffs)
