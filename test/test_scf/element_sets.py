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
Element sets for the element-wise SCF tests.

The parameter files define one shell layout per element, and only a handful of
layouts exist (`ss`: H, `s`/`sp`: He and main-group elements, `spd`: heavier
main-group elements, and `dsp`: transition metals and lanthanides, in which the
d shell comes *first*). Checking every element in every configuration mostly
repeats identical code paths, so charged states, float32 and the SCF-mode
comparison run on a representative subset that covers every layout and period.
The neutral-atom energy is still checked for *all* elements (in double
precision) to guard the per-element parameter data.

Elements that are known to fail (42 and 75 in GFN2, 25 in the Fermi-energy
test) are included on purpose, so they are not hidden.
"""
from __future__ import annotations

import torch

__all__ = [
    "REPRESENTATIVE_ELEMENTS",
    "all_double_reps_float",
    "reps_both_dtypes",
]

ALL_ELEMENTS = list(range(1, 87))

REPRESENTATIVE_ELEMENTS = [
    # ss / s
    1,
    2,
    # sp: alkali, p-block, 3d/4d/5d closed-shell metals, heavy p-block
    3,
    6,
    9,
    11,
    19,
    30,
    37,
    48,
    55,
    80,
    82,
    # spd: noble gases and heavier main group
    10,
    14,
    17,
    20,
    35,
    53,
    86,
    # dsp (d shell first): 3d, 4d, 5d transition metals and lanthanides
    21,
    25,
    26,
    29,
    39,
    42,
    47,
    57,
    71,
    75,
    79,
]


def all_double_reps_float() -> list[tuple[torch.dtype, int]]:
    """(dtype, element) pairs: all elements in double, the subset in float."""
    return [(torch.double, n) for n in ALL_ELEMENTS] + [
        (torch.float, n) for n in REPRESENTATIVE_ELEMENTS
    ]


def reps_both_dtypes() -> list[tuple[torch.dtype, int]]:
    """(dtype, element) pairs for the representative subset in both dtypes."""
    return [
        (dtype, n)
        for dtype in (torch.float, torch.double)
        for n in REPRESENTATIVE_ELEMENTS
    ]
