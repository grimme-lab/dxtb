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
Integrals: Overview
===================

This module stores the container, drivers and underlying
implementations of the integrals.

For the GFNn-xTB family of methods, only two-center one-electron integrals
are required.

- GFN1-xTB: Overlap Integral
- GFN2-xTB: Overlap, Dipole, and Quadrupole Integral

Fundametally, there are two drivers (backends) for the integral computation:

- *PyTorch*: pure PyTorch implementation, only overlap integral
- *libcint*: Python interface with custom backward functions for derivatives;
  arbitrary integrals and derivatives

We generally recommend the *libcint* driver as it is much faster, especially
for derivatives.
The driver can be selected with the ``int_driver`` keyword in the calculator
options:

.. code-block:: python

    import torch
    from dxtb.calculators import GFN1Calculator as Calculator

    numbers = torch.tensor([3, 1])
    opts = {"int_driver": "libcint"}
    calc = Calculator(numbers, opts=opts)

    print(calc.opts.ints.driver)  # 0
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dxtb.integrals import levels as levels
    from dxtb.integrals import types as types
    from dxtb.integrals import wrappers as wrappers
else:
    import dxtb._src.loader.lazy as _lazy

    __getattr__, __dir__, __all__ = _lazy.attach_module(
        __name__,
        ["levels", "types", "wrappers"],
    )
    del _lazy

del TYPE_CHECKING

from dxtb._src.integral.container import Integrals as Integrals
