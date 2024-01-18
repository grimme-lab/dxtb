from pathlib import Path

import torch
from tad_mctc.io import read
from tad_mctc.typing import DD

import dxtb

# import logging

# logging.basicConfig(
#     level=logging.CRITICAL,
#     format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
# )

device = torch.device("cpu")
dd: DD = {"device": device, "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "lysxao.coord"
f = Path(__file__).parent / "molecules" / "ethylacetate.xyz"
numbers, positions = read.read_from_path(f, ftype="xyz", **dd)
charge = torch.tensor(0.0, **dd)

opts = {
    "scf_mode": "full",
    "mixer": "anderson",
    "verbosity": 6,
}


print(dxtb.io.get_short_version())

# dipole moment requires electric field
field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)
ef = dxtb.external.new_efield(field_vector)

calc = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    opts=opts,
    interaction=[ef],
    **dd,
)


dxtb.timer.start("DipDer")
pos = positions.clone().requires_grad_(True)
agrad = calc.dipole_deriv(numbers, pos, charge)
dxtb.timer.stop("DipDer")

print(agrad.shape)

dxtb.timer.print_times()
dxtb.timer.reset()


# calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
# dxtb.timer.start("DipDer2")
# pos = positions.clone().requires_grad_(True)
# hess2 = calc.hessian2(numbers, pos, charge)
# dxtb.timer.stop("DipDer2")

# dxtb.timer.print_times()
# dxtb.timer.reset()


dxtb.timer.start("Num DipDer")
num = calc.dipole_deriv_numerical(numbers, positions, charge)
dxtb.timer.stop("Num DipDer")

print(num.shape)
print(num - agrad)
# print(numhess - hess2)
# print(hess - hess2)
