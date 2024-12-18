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
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
assert calc.integrals.hcore is not None


def get_energy_force(calc: dxtb.Calculator):
    forces = calc.get_forces(positions, create_graph=True)
    energy = calc.get_energy(positions)
    return energy, forces


es2 = calc.interactions.get_interaction("ES2")
es2.gexp = es2.gexp.clone().detach().requires_grad_(True)

hcore = calc.integrals.hcore
hcore.selfenergy = hcore.selfenergy.clone().detach().requires_grad_(True)

# energy and AD force
# energy, force = get_energy_force(calc)

# AD gradient w.r.t. params
energy, force = get_energy_force(calc)
de_dparam = torch.autograd.grad(
    energy, (es2.gexp, hcore.selfenergy), retain_graph=True
)

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)

es2 = calc.interactions.get_interaction("ES2")
es2.gexp = es2.gexp.clone().detach().requires_grad_(True)
hcore = calc.integrals.hcore
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
calc.integrals.hcore.selfenergy[0] += dparam / 2
energy1, force1 = get_energy_force(calc)
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
calc.integrals.hcore.selfenergy[0] -= dparam / 2
energy2, force2 = get_energy_force(calc)

de_dp = (energy1 - energy2) / dparam
print(f"dE / dselfenergy[0] (AD)   = {de_dparam[1][0]: .8f}")
print(f"dE / dselfenergy[0] (Num)  = {de_dp: .8f}")

df_dp = (torch.norm(force1) - torch.norm(force2)) / dparam
print(f"d|F| / dselfenergy[0] (AD)  = {dfnorm_dparam[1][0]: .8f}")
print(f"d|F| / dselfenergy[0] (Num) = {df_dp: .8f}")
