import torch

import dxtb

dd = {"dtype": torch.double, "device": torch.device("cpu")}

numbers = torch.tensor(["Li", "H"], device=dd["device"])
calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
energy = calc.energy(positions)

print(energy)
