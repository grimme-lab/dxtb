import torch

import dxtb

dd = {"dtype": torch.double, "device": torch.device("cpu")}

numbers = torch.tensor([3, 1], device=dd["device"])
calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
positions.requires_grad_(True)

energy = calc.energy(positions)
(g,) = torch.autograd.grad(energy, positions)

print(g)
