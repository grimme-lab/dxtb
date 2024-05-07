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
Exceptions: Integral
====================

Exceptions for the integral calculation.
"""

from __future__ import annotations

from ..pytorch import Tensor

__all__ = [
    "CGTOAzimuthalQuantumNumberError",
    "CGTOPrimitivesError",
    "CGTOPrincipalQuantumNumberError",
    "CGTOQuantumNumberError",
    "CGTOSlaterExponentsError",
    "IntegralTransformError",
]


class CGTOAzimuthalQuantumNumberError(ValueError):
    def __init__(self, l: int | Tensor) -> None:
        s = ["s", "p", "d", "f", "g", "h"][l]
        self.message = f"Maximum azimuthal QN supported is {l} ({s}-orbitals)."
        super().__init__(self.message)


class CGTOPrimitivesError(ValueError):
    def __init__(self) -> None:
        self.message = "Number of primitives must be between 1 and 6."
        super().__init__(self.message)


class CGTOPrincipalQuantumNumberError(ValueError):
    def __init__(self, n: int) -> None:
        self.message = f"Maximum principal QN supported is {n}."
        super().__init__(self.message)


class CGTOQuantumNumberError(ValueError):
    def __init__(self) -> None:
        self.message = (
            "Azimuthal QN 'l' and principal QN 'n' must adhere to "
            "l ∊ [n-1, n-2, ..., 1, 0]."
        )
        super().__init__(self.message)


class CGTOSlaterExponentsError(ValueError):
    def __init__(self) -> None:
        self.message = "Negative Slater exponents not allowed."
        super().__init__(self.message)


class IntegralTransformError(ValueError):
    def __init__(self) -> None:
        self.message = "[Fatal] Moments higher than f are not supported"
        super().__init__(self.message)
