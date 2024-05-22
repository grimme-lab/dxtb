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
Calculators: Base
=================

Base calculator for the extended tight-binding (xTB) models.
This calculator provides analytical, autograd, and numerical versions of all
properties.
"""
from __future__ import annotations

from dxtb._src.constants import defaults
from dxtb._src.typing import Any, Tensor

from .types.analytical import AnalyticalCalculator
from .types.autograd import AutogradCalculator
from .types.numerical import NumericalCalculator

__all__ = ["Calculator"]


class Calculator(AnalyticalCalculator, AutogradCalculator, NumericalCalculator):
    """
    Calculator for the extended tight-binding models (xTB).
    """

    implemented_properties = list(
        set(
            AnalyticalCalculator.implemented_properties
            + AutogradCalculator.implemented_properties
            + NumericalCalculator.implemented_properties
        )
    )
    """Names of implemented methods of the Calculator."""

    def calculate(
        self,
        properties: list[str],
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ):
        AutogradCalculator.calculate(self, properties, positions, chrg, spin, **kwargs)
