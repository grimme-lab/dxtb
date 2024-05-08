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
Calculators
===========

Entrypoints for ``dxtb`` calculations.
"""

from dxtb._src.calculators.gfn1 import GFN1Calculator as GFN1Calculator
from dxtb._src.calculators.gfn2 import GFN2Calculator as GFN2Calculator
from dxtb._src.calculators.properties.vibration import IRResult as IRResult
from dxtb._src.calculators.properties.vibration import RamanResult as RamanResult
from dxtb._src.calculators.properties.vibration import VibResult as VibResult
from dxtb._src.calculators.types import AnalyticalCalculator as AnalyticalCalculator
from dxtb._src.calculators.types import AutogradCalculator as AutogradCalculator
from dxtb._src.calculators.types import EnergyCalculator as EnergyCalculator
from dxtb._src.calculators.types import NumericalCalculator as NumericalCalculator
