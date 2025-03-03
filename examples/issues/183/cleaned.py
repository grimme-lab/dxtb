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
https://github.com/grimme-lab/dxtb/issues/183
"""
import torch

import dxtb
from dxtb.typing import DD

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

numbers = torch.tensor([8, 1, 1], device=dd["device"])
positions = torch.tensor(
    [
        [-2.95915993, 1.40005084, 0.24966306],
        [-2.1362031, 1.4795743, -1.38758999],
        [-2.40235213, 2.84218589, 1.24419946],
    ],
    requires_grad=True,
    **dd,
)

opts = {
    "scf_mode": dxtb.labels.SCF_MODE_FULL,
    "cache_enabled": True,
}


def main() -> int:
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
    assert calc.integrals.hcore is not None

    def get_energy_force(calc: dxtb.Calculator):
        # Using get_force() instead messes with the autograd graph
        # forces = calc.get_forces(positions, create_graph=True)
        energy = calc.get_energy(positions)
        forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
        return energy, forces

    es2 = calc.interactions.get_interaction("ES2")
    es2.gexp = es2.gexp.clone().detach().requires_grad_(True)

    hcore = calc.integrals.hcore
    hcore.selfenergy = hcore.selfenergy.clone().detach().requires_grad_(True)

    # AD gradient w.r.t. params
    energy, force = get_energy_force(calc)
    de_dparam = torch.autograd.grad(
        energy, (es2.gexp, hcore.selfenergy), retain_graph=True
    )

    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)

    es2 = calc.interactions.get_interaction("ES2")
    es2.gexp = es2.gexp.clone().detach().requires_grad_(True)
    hcore = calc.integrals.hcore
    assert hcore is not None
    hcore.selfenergy = hcore.selfenergy.clone().detach().requires_grad_(True)

    pos = positions.clone().detach().requires_grad_(True)
    energy = calc.get_energy(pos)
    force = -torch.autograd.grad(energy, pos, create_graph=True)[0]
    dfnorm_dparam = torch.autograd.grad(
        torch.norm(force), (es2.gexp, hcore.selfenergy)
    )

    # Numerical gradient w.r.t. params
    dparam = 2e-6
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
    es2 = calc.interactions.get_interaction("ES2")

    es2.gexp += dparam / 2
    energy1, force1 = get_energy_force(calc)

    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
    es2 = calc.interactions.get_interaction("ES2")

    es2.gexp -= dparam / 2
    energy2, force2 = get_energy_force(calc)

    de_dgexp = (energy1 - energy2) / dparam

    print(f"dE / dgexp (AD)  = {de_dparam[0]: .8f}")
    print(f"dE / dgexp (Num) = {de_dgexp: .8f}")

    dF_dgexp = (torch.norm(force1) - torch.norm(force2)) / dparam
    print(f"d|F| / dgexp (AD)  = {dfnorm_dparam[0]: .8f}")
    print(f"d|F| / dgexp (Num) = {dF_dgexp: .8f}")

    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
    assert calc.integrals.hcore is not None
    calc.integrals.hcore.selfenergy[0] += dparam / 2
    energy1, force1 = get_energy_force(calc)

    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
    assert calc.integrals.hcore is not None
    calc.integrals.hcore.selfenergy[0] -= dparam / 2
    energy2, force2 = get_energy_force(calc)

    de_dp = (energy1 - energy2) / dparam
    print(f"dE / dselfenergy[0] (AD)   = {de_dparam[1][0]: .8f}")
    print(f"dE / dselfenergy[0] (Num)  = {de_dp: .8f}")

    df_dp = (torch.norm(force1) - torch.norm(force2)) / dparam
    print(f"d|F| / dselfenergy[0] (AD)  = {dfnorm_dparam[1][0]: .8f}")
    print(f"d|F| / dselfenergy[0] (Num) = {df_dp: .8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
