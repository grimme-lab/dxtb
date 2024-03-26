from pathlib import Path

import torch
from tad_mctc.io import read
from tad_mctc.typing import DD

import dxtb

torch.set_printoptions(linewidth=400, precision=5)


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
    "verbosity": 0,
    "f_atol": 1e-10,
    "x_atol": 1e-10,
}


print(dxtb.io.get_short_version())

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
dxtb.timer.start("Vib")
num_vibres = calc.vibration_numerical(numbers, positions, charge)
num_vibres.use_common_units()
dxtb.timer.stop("Vib")


dxtb.timer.start("Hessian")
vibres = calc.vibration(numbers, positions.clone().requires_grad_(True), charge)
vibres.use_common_units()
dxtb.timer.stop("Hessian")

print("Shape of numbers", numbers.shape)
print("Shape of modes", vibres.modes.shape)
print("Shape of freqs", vibres.freqs.shape)


dxtb.timer.print_times()
dxtb.timer.reset()
