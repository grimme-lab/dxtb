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

f = Path(__file__).parent / "molecules" / "vancoh2.coord"
n, p = read.read_from_path(f, ftype="tm", **dd)
c = read.read_chrg_from_path(f, **dd)

nbatch = 60

numbers = dxtb.batch.pack([n for _ in range(nbatch)])
positions = dxtb.batch.pack([p for _ in range(nbatch)])
charge = dxtb.batch.pack([c for _ in range(nbatch)])

print(numbers.shape, positions.shape, charge.shape)

# conformer batched mode
opts = {"verbosity": 6, "batch_mode": 2}

dxtb.timer.reset()
dxtb.timer.start("Batch")
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
pos = positions.clone().requires_grad_(True)
result = calc.energy(numbers, pos, chrg=charge)


print(result)

(g,) = torch.autograd.grad(result, pos, grad_outputs=torch.ones_like(result))
print(g.shape)
dxtb.timer.stop("Batch")
dxtb.timer.print()
