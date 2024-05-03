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
from pathlib import Path

import torch
from tad_mctc.io import read

import dxtb
from dxtb.typing import DD

print("ACONF20 subset of ACONFL benchmark set")

###############################################################################

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

# subset of ACONFL benchmark set (conformers, same number of atoms)
f = Path(__file__).parent.parent / "molecules" / "aconf20.xyz"
numbers, positions = read.read_from_path(f, **dd)

# manually create the corresponding charge tensor
nbatch = numbers.shape[0]
charge = torch.tensor([0] * nbatch, **dd)

print(f"Shape: {numbers.shape}")

###############################################################################

print("\nStarting batch mode...")

# conformer batched mode
opts = {"verbosity": 0, "batch_mode": 2}

dxtb.timer.reset()
dxtb.timer.start("Batch")

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
result = calc.energy(positions, chrg=charge)

dxtb.timer.stop("Batch")
dxtb.timer.print(v=-999)

###############################################################################

print("\nStarting looped version...")

opts = {"verbosity": 0, "batch_mode": 0}

dxtb.timer.reset()
dxtb.timer.start("Loop")

for i in range(nbatch):
    calc = dxtb.Calculator(numbers[i], dxtb.GFN1_XTB, opts=opts, **dd)
    result = calc.energy(numbers[i], positions[i], chrg=charge[i])

dxtb.timer.stop("Loop")
dxtb.timer.print(v=-999)
