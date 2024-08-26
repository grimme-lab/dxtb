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
Implementation: Quadrupole
==========================

Quadrupole integral implementation based on `libcint`.
"""

from __future__ import annotations

from dxtb._src.integral.base import IntDriver
from dxtb._src.typing import Any, Tensor

from ...types import QuadrupoleIntegral
from .driver import IntDriverLibcint
from .multipole import MultipoleLibcint

__all__ = ["QuadrupoleLibcint"]


class QuadrupoleLibcint(QuadrupoleIntegral, MultipoleLibcint):
    """
    Quadrupole integral from atomic orbitals.
    """

    def build(self, driver: IntDriverLibcint) -> Tensor:
        """
        Calculation of quadrupole integral using libcint.

        Parameters
        ----------
        driver : IntDriverLibcint
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Quadrupole integral.
        """
        return self.multipole(driver, "r0r0")

    def get_gradient(self, driver: IntDriver, **kwargs: Any) -> Tensor:
        raise NotImplementedError("Gradient calculation not implemented.")
