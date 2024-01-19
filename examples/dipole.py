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
f = Path(__file__).parent / "molecules" / "h2o-dimer.coord"
numbers, positions = read.read_from_path(f, ftype="tm", **dd)
charge = torch.tensor(0.0, **dd)

opts = {
    "scf_mode": "full",
    "mixer": "anderson",
    "verbosity": 6,
}


print(dxtb.io.get_short_version())

# dipole moment requires electric field
field_vector = torch.tensor([0.0, 0.0, 0.0], **dd, requires_grad=True)
ef = dxtb.external.new_efield(field_vector)

calc = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    opts=opts,
    interaction=[ef],
    **dd,
)


dxtb.timer.start("Dipole")
agrad = calc.dipole(numbers, positions, charge, use_autograd=True)
dxtb.timer.stop("Dipole")

print(agrad.shape)

dxtb.timer.print_times()
dxtb.timer.reset()


dxtb.timer.start("Dipole2")
agrad2 = calc.dipole(numbers, positions, charge, use_autograd=True, use_functorch=True)
dxtb.timer.stop("Dipole2")

print(agrad2)


dxtb.timer.start("Num Dipole")
num = calc.dipole_numerical(numbers, positions, charge)
dxtb.timer.stop("Num Dipole")

dxtb.timer.print_times()
dxtb.timer.reset()

print(num.shape)
print("agrad\n", agrad)
print("agrad2\n", agrad2)
print("num\n", num)
print(num - agrad)
print(num - agrad2)
# print(numhess - hess2)
# print(hess - hess2)
