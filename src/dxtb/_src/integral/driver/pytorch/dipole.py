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

PyTorch-based dipole integral implementations.
"""

from __future__ import annotations

from dxtb._src.typing import Tensor

from ...types import DipoleIntegral
from .driver import IntDriverPytorch
from .impls.pipeline import DIPOLE_COMPONENTS
from .multipole import MultipolePytorch

__all__ = ["DipolePytorch"]


class DipolePytorch(DipoleIntegral, MultipolePytorch):
    """
    Dipole integral from atomic orbitals.
    """

    def build(self, driver: IntDriverPytorch) -> Tensor:
        """
        Dipole integral about the Cartesian origin (``r0``), using the
        algorithm selected on the driver (``int_algorithm``).

        Parameters
        ----------
        driver : IntDriverPytorch
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Integral of shape ``(..., 3, norb, norb)``.
        """
        return self.multipole(driver, DIPOLE_COMPONENTS)
