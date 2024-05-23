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
Calculators: Types
==================

All calculator types for the extended tight-binding models (xTB).

Besides the basic energy calculator, different calculator types utilize
different methods for differentiation, i.e., analytical, numerical, or
automatic differentiation (autograd). The central :class:`~dxtb.Calculator`
class **inherits from all types**, i.e., it provides the energy and properties
via automatic, analytical, and numerical differentiation.

The available methods of the specific types can be checked with:

.. code:: python

    from dxtb.calculators import AutogradCalculator

    print(AutogradCalculator.implemented_properties)

Since automatic differentiation is the main mode for derivative calculations,
all methods are available. The same is true for numerical differentiation,
which can be used for testing and debugging purposes. Only a few properties are
implemented with analytical differentiation. The syntax for calling the methods
is as follows:

- *Automatic differentiation*: ``calc.forces(positions)``
- *Analytical differentiation*: ``calc.forces_analytical(positions)``
- *Numerical differentiation*: ``calc.forces_numerical(positions)``


Calculators: AD
===============

Using the :meth:`get_<property>` methods will automatically select     automatic differentiation when multiple methods are available.

Note that most properties require Jacobians instead of vector-Jacobian products
(VJP), which can be calculated row-by-row using the standard
:func:`torch.autograd.grad` function with unit vectors in the VJP (see
`here <https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.\
html#computing-the-jacobian>`__)
or using :mod:`torch.func`'s function transforms (e.g.,
:func:`torch.func.jacrev`). The selection can be made with the ``use_functorch``
keyword argument.

While using functorch is faster, it is sometimes less stable and does not work
if we need to differentiate for multiple properties at once (e.g., Hessian and
dipole moment for IR spectra). Hence, the default is ``use_functorch=False``.
"""
from .analytical import *
from .autograd import *
from .energy import *
from .numerical import *
