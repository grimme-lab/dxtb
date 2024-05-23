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
Halogen Bond Correction
=======================

This module implements the halogen bond correction. The Halogen class is
constructed similar to the Repulsion class.

Example
-------
The following example demonstrates how to use the halogen bond correction with
the :class:`.Halogen` class in a simple scenario:

.. code-block:: python

    import torch
    from dxtb import IndexHelper
    from dxtb.classical import new_halogen
    from dxtb import GFN1_XTB

    # Define atomic numbers and positions
    numbers = torch.tensor([35, 35, 7, 1, 1, 1])
    positions = torch.tensor([
        [+0.00000000000000, +0.00000000000000, +3.11495251300000],
        [+0.00000000000000, +0.00000000000000, -1.25671880600000],
        [+0.00000000000000, +0.00000000000000, -6.30201130100000],
        [+0.00000000000000, +1.78712709700000, -6.97470840000000],
        [-1.54769692500000, -0.89356260400000, -6.97470840000000],
        [+1.54769692500000, -0.89356260400000, -6.97470840000000],
    ])

    # Instantiate the halogen bond correction class and the IndexHelper
    xb = new_halogen(numbers, positions, GFN1_XTB)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    # Build cache and use it for energy calculation
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)

    # Output the calculated energy
    print(energy.sum(-1))  # tensor(0.0025)
"""

from .factory import *
from .hal import *
