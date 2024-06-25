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
Implementation: Dipole
======================

Dipole integral implementation based on `libcint`.
"""

from __future__ import annotations

from tad_mctc.math import einsum

from dxtb._src.typing import Tensor

from .driver import IntDriverLibcint
from .multipole import MultipoleLibcint

__all__ = ["DipoleLibcint"]


class DipoleLibcint(MultipoleLibcint):
    """
    Dipole integral from atomic orbitals.
    """

    def build(self, driver: IntDriverLibcint) -> Tensor:
        """
        Calculation of dipole integral using libcint.

        Parameters
        ----------
        driver : IntDriverLibcint
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Dipole integral.
        """
        return self.multipole(driver, "r0")

    def shift_r0_rj(self, overlap: Tensor, pos: Tensor) -> Tensor:
        r"""
        Shift the centering of the dipole integral (moment operator) from the
        origin (:math:`r0 = r - (0, 0, 0)`) to atoms (ket index,
        :math:`rj = r - r_j`).

        .. math::

            \begin{align}
            D &= D^{r_j}  \\
            &= \langle i | r_j | j \rangle  \\
            &= \langle i | r | j \rangle - r_j \langle i | j \rangle \\
            &= \langle i | r_0 | j \rangle - r_j S_{ij}  \\
            &= D^{r_0} - r_j S_{ij}
            \end{align}

        Parameters
        ----------
        r0 : Tensor
            Origin centered dipole integral.
        overlap : Tensor
            Overlap integral.
        pos : Tensor
            Orbital-resolved atomic positions.

        Raises
        ------
        RuntimeError
            Shape mismatch between ``positions`` and `overlap`.
            The positions must be orbital-resolved.

        Returns
        -------
        Tensor
            Second-index (ket) atom-centered dipole integral.
        """
        if pos.shape[-2] != overlap.shape[-1]:
            raise RuntimeError(
                "Shape mismatch between positions and overlap integral. "
                "The position tensor must be spread to orbital-resolution."
            )

        shift = einsum("...jx,...ij->...xij", pos, overlap)
        self.matrix = self.matrix - shift
        return self.matrix
