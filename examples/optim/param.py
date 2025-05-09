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
"""Parameter gradient."""

import torch

import dxtb

dd = {"dtype": torch.double, "device": torch.device("cpu")}

# LiH
numbers = torch.tensor([3, 1], device=dd["device"])
positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], **dd)

# Create differentiable parameters
par = dxtb.ParamModule(dxtb.GFN1_XTB, **dd)
par.set_differentiable("element", "H", "slater")  # or "element.H.slater"

# Instantiate calculator and calculate energy
calc = dxtb.Calculator(numbers, par, **dd)
energy = calc.get_energy(positions)

# -------------------------------------------------------------------------
# Calculate gradient using autograd
# -------------------------------------------------------------------------

# Inspect slater parameters
slater = par.get("element", "H", "slater")
assert slater.requires_grad is True
print(f"\n{slater}")

# Obtain gradient (dE/dparam) via autograd
(grad_ad,) = torch.autograd.grad(energy, slater)

# -------------------------------------------------------------------------
# Calculate gradient using finite-difference
# -------------------------------------------------------------------------
with torch.no_grad():
    eps = 1.0e-4
    grad_num = torch.zeros_like(slater)
    opts = {"verbosity": 0}

    for i in range(slater.numel()):
        orig = slater[i].item()

        # f(x + ε)
        slater[i] = orig + eps
        calc = dxtb.Calculator(numbers, par, opts=opts, **dd)
        e_plus = calc.get_energy(positions)

        # f(x − ε)
        slater[i] = orig - eps
        calc = dxtb.Calculator(numbers, par, opts=opts, **dd)
        e_minus = calc.get_energy(positions)

        # central finite difference
        grad_num[i] = (e_plus - e_minus) / (2.0 * eps)

        slater[i] = orig

# -------------------------------------------------------------------------
# Compare gradients
# -------------------------------------------------------------------------
print(f"\nAutograd gradient:\n{grad_ad}")
print(f"\nFinite-difference gradient:\n{grad_num}")

max_abs_err = (grad_ad - grad_num).abs().max()
rel_err = max_abs_err / grad_ad.abs().max()
print(f"\nMax |autograd - numgrad|  : {max_abs_err:.3e}")
print(f"Max relative error        : {rel_err:.3e}")
