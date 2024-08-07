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
Example for xitorch's inability to be used together with functorch.
"""
from pathlib import Path

import torch
from tad_mctc.io import read

import dxtb
from dxtb.typing import DD

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "lih.xyz"
numbers, positions = read.read_from_path(f, **dd)
charge = read.read_chrg_from_path(f, **dd)

opts = {"verbosity": 3, "scf_mode": "nonpure"}

######################################################################

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
pos = positions.clone().requires_grad_(True)
hess = calc.hessian(pos, chrg=charge, use_functorch=True)
