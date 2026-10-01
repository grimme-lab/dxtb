#!/usr/bin/env python3
# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
"""Compare restricted, independent UHF, and spin-polarized UHF runs."""

import torch

import dxtb
from dxtb.components.spin import new_spin_polarization

dd = {"device": torch.device("cpu"), "dtype": torch.float64}

# Open-shell OH at approximately its equilibrium bond length (atomic units).
numbers = torch.tensor([8, 1], device=dd["device"])
positions = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
    **dd,
)
spin = 1  # N_alpha - N_beta: one unpaired electron

# 1. Restricted: one shared spatial-orbital channel.
restricted = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    opts={"verbosity": 0},
    **dd,
).singlepoint(positions, spin=spin)

# 2. Independent UHF: alpha/beta channels, without spin-polarization energy.
unrestricted = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    opts={"verbosity": 0, "uhf_mode": True},
    **dd,
).singlepoint(positions, spin=spin)

# 3. Spin-polarized UHF: the interaction itself requests two channels.
spinpol = new_spin_polarization(numbers, wscale=1.0, **dd)
spin_polarized = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    interaction=[spinpol],
    opts={"verbosity": 0},
    **dd,
).singlepoint(positions, spin=spin)

assert restricted.nspin == 1
assert unrestricted.nspin == spin_polarized.nspin == 2
assert restricted.density.shape == (6, 6)
assert unrestricted.density.shape == spin_polarized.density.shape == (2, 6, 6)

print("restricted:        ", restricted.total.sum().item())
print("independent UHF:   ", unrestricted.total.sum().item())
print("spin-polarized UHF:", spin_polarized.total.sum().item())
