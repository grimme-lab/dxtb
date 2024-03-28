import logging
from pathlib import Path

import torch
from tad_mctc.io import read
from tad_mctc.typing import DD

import dxtb

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
)

device = torch.device("cpu")
dd: DD = {"device": device, "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "lysxao.coord"
f = Path(__file__).parent / "molecules" / "h2o.coord"
numbers, positions = read.read_from_path(f, ftype="tm", **dd)

# f = Path(__file__).parent / "molecules" / "capsaicin.xyz"
# numbers, positions = read.read_from_path(f, ftype="xyz", **dd)

charge = torch.tensor(0.0, **dd)

opts = {
    "scf_mode": "full",
    "mixer": "anderson",
    "verbosity": 7,
    "f_atol": 1e-3,
    "x_atol": 1e-3,
}


print(dxtb.io.get_short_version())

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
dxtb.timer.start("Hessian")
pos = positions.clone().requires_grad_(True)
hess = calc.hessian_numerical(numbers, pos, charge)
dxtb.timer.stop("Hessian")
dxtb.timer.print()
dxtb.timer.reset()

# torch.save(hess, "hess_num.pt")

# calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
# dxtb.timer.start("Hessian2")
# pos = positions.clone().requires_grad_(True)
# hess2 = calc.hessian2(numbers, pos, charge)
# dxtb.timer.stop("Hessian2")

# dxtb.timer.print()
# dxtb.timer.reset()


# dxtb.timer.start("Num Hessian")
# numhess = calc.hessian_numerical(numbers, positions, charge)
# dxtb.timer.stop("Num Hessian")

# s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
# numhess = numhess.reshape(*s)
# print(numhess - hess)
# print(numhess - hess2)
# print(hess - hess2)
