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

The :class:`dxtb.Calculator` object is the center-piece of ``dxtb``, providing a simple interface to compute energies, forces and other properties of molecules.

Examples
--------
The calculator is instantiated with the atomic ``numbers`` and a
parametrization.

.. code-block:: python

    import torch
    from dxtb import Calculator
    from dxtb import GFN1_XTB

    numbers = torch.tensor([1, 1])
    calc = Calculator(numbers, GFN1_XTB)

It is recommended to always pass the :class:`~torch.dtype` and
:class:`~torch.device` to the calculator.

.. code-block:: python

    import torch
    import dxtb
    from dxtb._src.typing import DD

    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
    print(calc.device)  # Expected output: cpu
    print(calc.dtype)  # Expected output: torch.float64

The :class:`~torch.dtype` and :class:`~torch.device` can be conveniently
changed after instantiation in the same way as for any other PyTorch tensor.

.. code-block:: python

    import dxtb

    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
    calc = calc.type(torch.float32)
    print(calc.dtype)  # Expected output: torch.float32

To configure settings, a dictionary of settings can be passed to the calculator.

.. code-block:: python

    import torch
    import dxtb

    settings = {"maxiter": 100}
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=settings)

Additional tight binding components can also be added.

.. code-block:: python

    import torch
    import dxtb

    dd = {"device": torch.device("cpu"), "dtype": torch.double}
    ef = dxtb.field.new_efield(torch.tensor([0.0, 0.0, 0.0], **dd))
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, interactions=[ef], **dd)
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
