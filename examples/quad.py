import logging
from pathlib import Path

import torch
from tad_mctc.io import read
from tad_mctc.typing import DD

import dxtb

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
# )

device = torch.device("cpu")
dd: DD = {"device": device, "dtype": torch.double}

f = Path(__file__).parent / "molecules" / "h2o.coord"
numbers, positions = read.read_from_path(f, ftype="tm", **dd)
# f = Path(__file__).parent / "molecules" / "nh3-planar.xyz"
# numbers, positions = read.read_from_path(f, ftype="xyz", **dd)
charge = torch.tensor(0.0, **dd)

# position gradient for intensities
efg_mat = torch.zeros((3, 3), **dd, requires_grad=True)
efg = dxtb.external.new_efield_grad(efg_mat)

opts = {
    "scf_mode": "full",
    "mixer": "anderson",
    "verbosity": 6,
}

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, interaction=[efg], **dd)

print(dxtb.io.get_short_version())

dxtb.timer.start("Quad")
quad = calc.quadrupole_analytical(numbers, positions, charge)
dxtb.timer.stop("Quad")

dxtb.timer.print()
dxtb.timer.reset()

####################################################

dxtb.timer.start("Quad")
q2 = calc.quadrupole(numbers, positions, charge, use_functorch=True)
dxtb.timer.stop("Quad")

dxtb.timer.print()
dxtb.timer.reset()


print(quad)
print(q2)
