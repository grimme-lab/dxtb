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

# FIX
opts = {"scf_mode": dxtb.labels.SCF_MODE_FULL}

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)


def get_energy_force(calc):
    energy = calc.energy(positions)
    force = -torch.autograd.grad(energy, positions, create_graph=True)[0]
    return energy, force


for c in calc.interactions.components:
    if isinstance(c, dxtb._src.components.interactions.coulomb.secondorder.ES2):
        break
c.gexp = c.gexp.clone().detach().requires_grad_(True)

hcore = calc.integrals.hcore
hcore.selfenergy = hcore.selfenergy.clone().detach().requires_grad_(True)

# energy and AD force
energy, force = get_energy_force(calc)

# AD gradient w.r.t. params
de_dparam = torch.autograd.grad(
    energy, (c.gexp, hcore.selfenergy), retain_graph=True
)
dfnorm_dparam = torch.autograd.grad(
    torch.norm(force), (c.gexp, hcore.selfenergy)
)

# Numerical gradient w.r.t. params
dparam = 2e-6
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
for c in calc.interactions.components:
    if isinstance(c, dxtb._src.components.interactions.coulomb.secondorder.ES2):
        break

c.gexp += dparam / 2
energy1, force1 = get_energy_force(calc)
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
for c in calc.interactions.components:
    if isinstance(c, dxtb._src.components.interactions.coulomb.secondorder.ES2):
        break

c.gexp -= dparam / 2
energy2, force2 = get_energy_force(calc)

print(
    f"dE / dgexp = {de_dparam[0].item()} (AD) {(energy1-energy2)/dparam} (Numerical)"
)
print(
    f"d|F| / dgexp = {dfnorm_dparam[0].item()} (AD) {(torch.norm(force1)-torch.norm(force2)).item()/dparam} (Numerical)"
)

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
calc.integrals.hcore.selfenergy[0] += dparam / 2
energy1, force1 = get_energy_force(calc)
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)
calc.integrals.hcore.selfenergy[0] -= dparam / 2
energy2, force2 = get_energy_force(calc)

print(
    f"dE / dselfenergy[0] = {de_dparam[1][0].item()} (AD) {(energy1-energy2)/dparam} (Numerical)"
)
print(
    f"d|F| / dselfenergy[0] = {dfnorm_dparam[1][0].item()} (AD) {(torch.norm(force1)-torch.norm(force2)).item()/dparam} (Numerical)"
)
