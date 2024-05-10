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
Calculating forces for vancomycin via AD.
"""
from pathlib import Path

import torch
from tad_mctc.io import read

import dxtb
from dxtb.typing import DD

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "vancoh2.coord"
numbers, positions = read.read_from_path(f, ftype="tm", **dd)
charge = read.read_chrg_from_path(f, **dd)

opts = {"verbosity": 3}

######################################################################

print("Calculating forces manually with :func:`torch.autograd.grad`.\n")

dxtb.timer.reset()

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
pos = positions.clone().requires_grad_(True)
energy = calc.energy(pos, chrg=charge)

(g,) = torch.autograd.grad(energy, pos, grad_outputs=torch.ones_like(energy))
forces1 = -g

dxtb.timer.print()

######################################################################

print("\n\n\nCalculating forces with Calculator method.\n")

dxtb.timer.reset()

calc.reset()
pos = positions.clone().requires_grad_(True)
forces2 = calc.forces(pos, chrg=charge)

dxtb.timer.print()

equal = torch.allclose(forces1, forces2, atol=1e-6, rtol=1e-6)
print("\n\nForces are equal:", equal)
