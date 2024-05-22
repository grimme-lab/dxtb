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
Typing for the overlap functions.
"""

from __future__ import annotations

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.constants import defaults
from dxtb._src.typing import Literal, Protocol, Tensor

__all__ = ["OverlapFunction"]


class OverlapFunction(Protocol):
    """
    Type annotation for overlap and gradient function.
    """

    def __call__(
        self,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
    ) -> Tensor:
        """
        Evaluation of the overlap integral or its gradient.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        bas : Basis
            Basis set information.
        ihelp : IndexHelper
            Helper class for indexing.
        uplo : Literal['n';, 'u', 'l'], optional
            Whether the matrix of unique shell pairs should be create as a
            triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
            Defaults to `l` (lower triangular matrix).
        cutoff : Tensor | float | int | None, optional
            Real-space cutoff for integral calculation in Angstrom. Defaults to
            `constants.defaults.INTCUTOFF` (50.0).

        Returns
        -------
        Tensor
            Overlap matrix or overlap gradient.
        """
        ...  # pylint: disable=unnecessary-ellipsis
