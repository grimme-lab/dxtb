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

f = Path(__file__).parent / "molecules" / "ch4.coord"
numbers, positions = read.read_from_path(f, ftype="tm", **dd)
charge = torch.tensor(0.0, **dd)

# position gradient for intensities
positions.requires_grad_(True)

opts = {
    "scf_mode": "full",
    "mixer": "anderson",
    "verbosity": 6,
}

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)

print(dxtb.io.get_short_version())

dxtb.timer.start("Forces")
hess = calc.forces(numbers, positions, charge)
dxtb.timer.stop("Forces")

dxtb.timer.print_times()
dxtb.timer.reset()

dxtb.timer.start("Num Forces")
numhess = calc.forces_numerical(numbers, positions, charge)
dxtb.timer.stop("Num Forces")

dxtb.timer.print_times()

print(hess)
print(numhess)
