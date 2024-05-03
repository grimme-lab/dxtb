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

Calculators for the extended tight-binding models (xTB). The `Calculator`
object is the center-piece of `dxtb`, providing a simple interface to compute
energies, forces and other properties of molecules.

Examples
--------
The calculator is instantiated with the atomic numbers and a parametrization.

>>> import torch
>>> from dxtb.xtb import Calculator
>>> from dxtb.param import GFN1_XTB
>>>
>>> numbers = torch.tensor([1, 1])
>>> calc = Calculator(numbers, GFN1_XTB)

It is recommended to always pass the `dtype` and `device` to the calculator.

>>> from dxtb.typing import DD
>>>
>>> dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
>>> calc = Calculator(numbers, GFN1_XTB, **dd)
>>> print(calc.device)
cpu
>>> print(calc.dtype)
torch.float64

The `dtype` and `device` can be conveniently changed after instantiation in the
same way as for any other PyTorch tensor.

>>> calc = Calculator(numbers, GFN1_XTB, **dd)
>>> calc = calc.type(torch.float32)
>>> print(calc.dtype)
torch.float32

To configure settings, a dictionary of settings can be passed to the calculator.

>>> settings = {"maxiter": 100}
>>> calc = Calculator(numbers, GFN1_XTB, opts=settings)

Additional tight binding components can also be added.

>>> dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
>>> ef = dxtb.field.new_efield(torch.tensor([0.0, 0.0, 0.0], **dd))
>>> calc = Calculator(numbers, GFN1_XTB, interactions=[ef], **dd)
"""
from .base import *
from .gfn1 import *
from .gfn2 import *
