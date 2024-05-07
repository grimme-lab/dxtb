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
Repulsion
=========

This module implements the classical repulsion energy term.

Note
----
The Repulsion class is constructed for geometry optimization, i.e., the atomic
numbers are set upon instantiation (`numbers` is a property), and the parameters
in the cache are created for only those atomic numbers. The positions, however,
must be supplied to the ``get_energy`` (or ``get_grad``) method.

Example
-------

.. code-block:: python

    import torch
    from dxtb import IndexHelper
    from dxtb.classical import new_repulsion
    from dxtb import GFN1_XTB

    # Define atomic numbers and positions
    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [+0.00000000000000, +0.00000000000000, +0.00000000000000],
        [+1.61768389755830, +1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [+1.61768389755830, -1.61768389755830, +1.61768389755830],
        [-1.61768389755830, +1.61768389755830, +1.61768389755830],
    ])

    # Initialize the repulsion and IndexHelper objects
    rep = new_repulsion(numbers, positions, GFN1_XTB)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    # Get the cache and calculate the energy
    cache = rep.get_cache(numbers, ihelp)
    energy = rep.get_energy(positions, cache)

    # Output the summed energy across all atoms
    print(energy.sum(-1))
"""

from .factory import new_repulsion
from .rep import LABEL_REPULSION, Repulsion, RepulsionAnalytical

__all__ = [
    "LABEL_REPULSION",
    "new_repulsion",
    "Repulsion",
    "RepulsionAnalytical",
]
