from pathlib import Path

import torch
from tad_mctc.io import read
from tad_mctc.typing import DD
from tad_mctc.units import AU2RCM

import dxtb

torch.set_printoptions(linewidth=400, precision=5)

# import logging

# logging.basicConfig(
#     level=logging.CRITICAL,
#     format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
# )

device = torch.device("cpu")
dd: DD = {"device": device, "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "nh3-planar.xyz"
numbers, positions = read.read_from_path(f, ftype="xyz", **dd)

# f = Path(__file__).parent / "molecules" / "h2o.coord"
# numbers, positions = read.read_from_path(f, ftype="tm", **dd)
charge = torch.tensor(0.0, **dd)

opts = {
    "scf_mode": "full",
    "mixer": "anderson",
    "verbosity": 6,
    "f_atol": 1e-10,
    "x_atol": 1e-10,
}


print(dxtb.io.get_short_version())

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
dxtb.timer.start("Hessian")
pos = positions.clone().requires_grad_(True)
freqs, modes = calc.vibration(numbers, pos, charge)
dxtb.timer.stop("Hessian")

print("Shape of numbers", numbers.shape)
print("Shape of modes", modes.shape)
print("Shape of freqs", freqs.shape)
print("")
print("")
print("")

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
dxtb.timer.start("Hessian")
freqs, modes = calc.vibration_numerical(numbers, positions, charge)
dxtb.timer.stop("Hessian")

dxtb.timer.print_times()
dxtb.timer.reset()


# calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
# dxtb.timer.start("Hessian")
# pos = positions.clone().requires_grad_(True)
# hess = calc.hessian(numbers, pos, charge)
# dxtb.timer.stop("Hessian")

# dxtb.timer.print_times()
# dxtb.timer.reset()

# print(hess)

# dxtb.timer.start("Num Hessian")
# numhess = calc.hessian_numerical(numbers, positions, charge, step_size=0.005)
# dxtb.timer.stop("Num Hessian")

# s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
# numhess = numhess.reshape(*s)
# print(numhess)
# print(numhess - hess)
# print(numhess - hess2)
# print(hess - hess2)
