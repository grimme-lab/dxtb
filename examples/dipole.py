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
from tad_mctc.typing import DD

import dxtb

# import logging

# logging.basicConfig(
#     level=logging.CRITICAL,
#     format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
# )

device = torch.device("cpu")
dd: DD = {"device": device, "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "lysxao.coord"
f = Path(__file__).parent / "molecules" / "h2o-dimer.coord"
numbers, positions = read.read_from_path(f, ftype="tm", **dd)
charge = torch.tensor(0.0, **dd)

opts = {
    "scf_mode": "full",
    "mixer": "anderson",
    "verbosity": 6,
}


print(dxtb.io.get_short_version())

# dipole moment requires electric field
field_vector = torch.tensor([0.0, 0.0, 0.0], **dd, requires_grad=True)
ef = dxtb.external.new_efield(field_vector)

calc = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    opts=opts,
    interaction=[ef],
    **dd,
)


dxtb.timer.start("Dipole")
agrad = calc.dipole(numbers, positions, charge, use_functorch=False)
dxtb.timer.stop("Dipole")

print(agrad.shape)

dxtb.timer.print()
dxtb.timer.reset()


dxtb.timer.start("Dipole2")
agrad2 = calc.dipole(numbers, positions, charge, use_functorch=True)
dxtb.timer.stop("Dipole2")

print(agrad2)


dxtb.timer.start("Num Dipole")
num = calc.dipole_numerical(numbers, positions, charge)
dxtb.timer.stop("Num Dipole")

dxtb.timer.print()
dxtb.timer.reset()

print(num.shape)
print("agrad\n", agrad)
print("agrad2\n", agrad2)
print("num\n", num)
print(num - agrad)
print(num - agrad2)
# print(numhess - hess2)
# print(hess - hess2)
