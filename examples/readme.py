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
Example from README.md.
"""
import torch

import dxtb

dd = {"dtype": torch.double, "device": torch.device("cpu")}

# LiH
numbers = torch.tensor([3, 1], device=dd["device"])
positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], **dd)

# instantiate a calculator
calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

# compute the energy
pos = positions.clone().requires_grad_(True)
energy = calc.get_energy(pos)

# obtain gradient (dE/dR) via autograd
(g,) = torch.autograd.grad(energy, pos)

# Alternatively, forces can directly be requested from the calculator.
# (Don't forget to reset the calculator manually when the inputs are identical.)
calc.reset()
pos = positions.clone().requires_grad_(True)
forces = calc.get_forces(pos)

assert torch.equal(forces, -g)
