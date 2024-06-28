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
Calculators: Overview
=====================

The :class:`dxtb.Calculator` object is the center-piece of ``dxtb``, providing
a simple interface to compute energies, forces and other properties of
molecules.

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
    from dxtb.typing import DD

    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    numbers = torch.tensor([1, 1])

    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
    print(calc.device)  # Expected output: cpu
    print(calc.dtype)  # Expected output: torch.float64

The :class:`~torch.dtype` and :class:`~torch.device` can be conveniently
changed after instantiation in the same way as for any other PyTorch tensor.

.. code-block:: python

    import torch
    import dxtb

    numbers = torch.tensor([1, 1])
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
    calc = calc.type(torch.float32)
    print(calc.dtype)  # Expected output: torch.float32

To configure settings, a dictionary of settings can be passed to the calculator.

.. code-block:: python

    import torch
    import dxtb

    numbers = torch.tensor([1, 1])
    settings = {"maxiter": 100}
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=settings)

Additional tight binding components can also be added.

.. code-block:: python

    import torch
    import dxtb

    dd = {"device": torch.device("cpu"), "dtype": torch.double}
    numbers = torch.tensor([1, 1])

    ef = dxtb.components.field.new_efield(torch.tensor([0.0, 0.0, 0.0], **dd))
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, interaction=[ef], **dd)


Calculators: Get Properties
===========================

The calculator can be used to compute energies, forces, dipole moments and
other properties. The properties are computed by calling the respective
:meth:`get_<property>` method, just as in ASE.

Depending on which calculator you choose, the properties are calculated using
analytical, autograd, or numerical derivatives. The default uses automatic
differentation. For details, see the calculator types.

Calculators: Caching
====================

All properties can be cached. However, caching is not enabled by default. To
enable caching, pass ``{"cache_enabled": True}`` to the calculator options.

.. warning::

    Caching may lead to side effects if automatic differentiation is used
    multiple times. If you encounter any issues, try running
    :meth:`~dxtb.Calculator.reset_all`. If this does not help, disable caching
    or report the issue.
"""

from dxtb._src.calculators.gfn1 import GFN1Calculator as GFN1Calculator
from dxtb._src.calculators.gfn2 import GFN2Calculator as GFN2Calculator
from dxtb._src.calculators.types import AnalyticalCalculator as AnalyticalCalculator
from dxtb._src.calculators.types import AutogradCalculator as AutogradCalculator
from dxtb._src.calculators.types import EnergyCalculator as EnergyCalculator
from dxtb._src.calculators.types import NumericalCalculator as NumericalCalculator
