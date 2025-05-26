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
https://github.com/grimme-lab/dxtb/issues/223
"""
import torch
from tad_mctc import read
from tad_mctc.batch import pack

import dxtb

torch.set_printoptions(precision=10)

dd = {"dtype": torch.double, "device": torch.device("cpu")}

opts = {
    "verbosity": 0,
    "scf_mode": dxtb.labels.SCF_MODE_FULL,
    "scp_mode": dxtb.labels.SCP_MODE_CHARGE,
    "damp": 0.1,
}


# total charge of both system
charge = torch.tensor([0.0, 0.0], **dd)

# ------------------------------------------------------------
# small systems
# --------------------------------------------------------------

num1, pos1 = read("coord1_small.xyz", **dd)
num2, pos2 = read("coord2_small.xyz", **dd)

# RUN 1

numbers = pack([num1, num1])
positions = pack([pos1, pos1])

# instantiate calculator and calculate GFN1 energy in Hartree
calc = dxtb.calculators.GFN1Calculator(numbers, opts=opts, **dd)
energy = calc.get_energy(positions, charge)

print(f"Calculated 1 small: {energy}")

# RUN 2

numbers = pack([num1, num2])
positions = pack([pos1, pos2])

# instantiate calculator and calculate GFN1 energy in Hartree
calc = dxtb.calculators.GFN1Calculator(numbers, opts=opts, **dd)
energy = calc.get_energy(positions, charge)

print(f"Calculated 2 small: {energy}")

# ------------------------------------------------------------
# large systems
# ------------------------------------------------------------

num3, pos3 = read("coord1.xyz", **dd)
num4, pos4 = read("coord2.xyz", **dd)

# RUN 1

numbers = pack([num3, num3])
positions = pack([pos3, pos3])

# instantiate calculator and calculate GFN1 energy in Hartree
calc = dxtb.calculators.GFN1Calculator(numbers, opts=opts, **dd)
energy = calc.get_energy(positions, charge)

print(f"Calculated 1 big: {energy}")

# RUN 2

numbers = pack([num3, num4])
positions = pack([pos3, pos4])

# instantiate calculator and calculate GFN1 energy in Hartree
calc = dxtb.calculators.GFN1Calculator(numbers, opts=opts, **dd)
energy = calc.get_energy(positions, charge)

print(f"Calculated 2 big: {energy}")
