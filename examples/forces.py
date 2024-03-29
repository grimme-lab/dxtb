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
import logging
from pathlib import Path

import torch
from tad_mctc.io import read
from tad_mctc.typing import DD

import dxtb

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
# )

device = torch.device("cpu")
dd: DD = {"device": device, "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "ch4.coord"
numbers, positions = read.read_from_path(f, ftype="tm", **dd)
f = Path(__file__).parent / "molecules" / "nh3-planar.xyz"
numbers, positions = read.read_from_path(f, ftype="xyz", **dd)
charge = torch.tensor(0.0, **dd)

# position gradient for intensities
positions.requires_grad_(True)

opts = {
    "scf_mode": "full",
    "mixer": "anderson",
    "verbosity": 6,
}

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)

print(dxtb.io.get_short_version())

dxtb.timer.start("Forces")
force = calc.forces(numbers, positions, charge)
dxtb.timer.stop("Forces")

dxtb.timer.print()
dxtb.timer.reset()

dxtb.timer.start("Num Forces")
numforce = calc.forces_numerical(numbers, positions, charge, step_size=1e-6)
dxtb.timer.stop("Num Forces")

dxtb.timer.print()

print(-force)
print(-numforce)
