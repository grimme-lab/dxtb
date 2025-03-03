#!/usr/bin/env python3
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
Simple integral interface. Can be helpful for testing.
"""
from pathlib import Path

import torch
from tad_mctc import read

from dxtb import GFN1_XTB
from dxtb.integrals.wrappers import overlap

# SiH4
numbers = torch.tensor([14, 1, 1, 1, 1])
positions = torch.tensor(
    [
        [+0.00000000000000, +0.00000000000000, +0.00000000000000],
        [+1.61768389755830, +1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [+1.61768389755830, -1.61768389755830, +1.61768389755830],
        [-1.61768389755830, +1.61768389755830, +1.61768389755830],
    ]
)

s = overlap(numbers, positions, GFN1_XTB)

print(f"Overlap integral of SiH4 has shape: {s.shape}\n\n")

##################################################################


torch.set_printoptions(linewidth=200)

path = Path(__file__).resolve().parent / "molecules" / "lih.xyz"
numbers, positions = read(path)

s = overlap(numbers, positions, GFN1_XTB)
print(f"Overlap integral of LiH has shape: {s.shape}\n")
print("The normalized overlap integral is:\n", s)

s = overlap(numbers, positions, GFN1_XTB, normalize=False)
print("\nThe non-normalized overlap integral is:\n", s)
