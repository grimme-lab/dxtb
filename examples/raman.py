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

device = torch.device("cpu")
dd: DD = {"device": device, "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "h2o.coord"
numbers, positions = read.read_from_path(f, ftype="tm", **dd)
charge = read.read_chrg_from_path(f, **dd)

# position gradient for intensities
pos = positions.clone()
positions.requires_grad_(True)

# dipole moment requires electric field
field_vector = torch.tensor([0.0, 0.0, 0.0], **dd, requires_grad=True)
ef = dxtb.external.new_efield(field_vector)

opts = {
    "f_atol": 1e-6,
    "x_atol": 1e-6,
    "scf_mode": "full",
    "mixer": "anderson",
    "verbosity": 6,
}

print(dxtb.io.get_short_version())

# SETUP
calc = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    opts=opts,
    interaction=[ef],
    **dd,
)


# AUTODIFF
res = calc.raman(numbers, positions, charge)
res.use_common_units()
res.save_prop_to_pt("freqs", Path(__file__).parent / "raman-freqs.pt")
res.save_prop_to_pt("ints", Path(__file__).parent / "raman-ints.pt")

# NUMERICAL
res = calc.raman_numerical(numbers, positions, charge)
res.use_common_units()
res.save_prop_to_pt("freqs", Path(__file__).parent / "raman-num-freqs.pt")
res.save_prop_to_pt("ints", Path(__file__).parent / "raman-num-ints.pt")
