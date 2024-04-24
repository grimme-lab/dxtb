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

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "penicillin.xyz"
f = Path(__file__).parent / "molecules" / "8-hydroxyquinoline.xyz"
f = Path(__file__).parent / "molecules" / "aconf20.xyz"
n, p = read.read_from_path(f, **dd)
c = read.read_chrg_from_path(f, **dd)

nbatch = 1000

# numbers = dxtb.batch.pack([n for _ in range(nbatch)])
# positions = dxtb.batch.pack([p for _ in range(nbatch)])
# charge = dxtb.batch.pack([c for _ in range(nbatch)])
numbers, positions = n, p
charge = dxtb.batch.pack([c for _ in range(n.shape[0])])

print(numbers.shape, positions.shape, charge.shape)

# numbers = torch.tensor(
#     [
#         [3, 1, 0],
#         [8, 1, 1],
#     ]
# )
# positions = torch.tensor(
#     [
#         [
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0],
#             [0.0, 0.0, 0.0],
#         ],
#         [
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0],
#             [0.0, 0.0, 2.0],
#         ],ht
#     ],
#     **dd
# )
# charge = torch.tensor([0, 0], **dd)


# conformer batched mode
opts = {"verbosity": 0, "batch_mode": 2}

dxtb.timer.reset()
dxtb.timer.start("Batch")
ef = dxtb.efield.new_efield(torch.tensor([0.0, 0.0, 0.0], **dd))
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, interaction=[ef], **dd)
result = calc.energy(numbers, positions, chrg=charge)
dxtb.timer.stop("Batch")
dxtb.timer.print(v=-999)


opts = {"verbosity": 0}
dxtb.timer.reset()
dxtb.timer.start("Loop")
for i in range(numbers.size(0)):
    calc = dxtb.Calculator(numbers[i], dxtb.GFN1_XTB, opts=opts, interaction=[ef], **dd)
    result = calc.energy(numbers[i], positions[i], chrg=charge[i])
dxtb.timer.stop("Loop")
dxtb.timer.print(v=-999)
