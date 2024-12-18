import torch

import dxtb
from dxtb.typing import DD

############################################
# Setup
############################################

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

opts = {"verbosity": 0}
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
assert calc.integrals.hcore is not None


############################################
# Minimization
############################################

from dxtb._src.exlibs.xitorch.optimize import minimize


def get_energy(positions) -> torch.Tensor:
    return calc.get_energy(positions)


minpos = minimize(
    get_energy,
    positions,
    method="gd",
    maxiter=200,
    step=1e-2,
    verbose=True,
)


print("\nInitial geometry:")
print(positions.detach().numpy())

print("Optimized geometry:")
print(minpos.detach().numpy())
