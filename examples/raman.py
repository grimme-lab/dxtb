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

# raman requires electric field and positions with grad
field_vector = torch.tensor([0.0, 0.0, 0.0], **dd, requires_grad=True)
ef = dxtb.external.new_efield(field_vector)


calc = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    opts=opts,
    interaction=[ef],
    **dd,
)


dxtb.timer.start("Raman")
pos = positions.clone().requires_grad_(True)
freqs, ints = calc.raman(numbers, pos, charge)
dxtb.timer.stop("Raman")

print(ints.shape)

dxtb.timer.print_times()
dxtb.timer.reset()


dxtb.timer.start("Num Raman")
num_freqs, num_ints = calc.raman_numerical(numbers, positions, charge)
dxtb.timer.stop("Num Raman")

dxtb.timer.print_times()
dxtb.timer.reset()

print(num_freqs.shape)
print(freqs)
print(num_freqs)
print(num_freqs - freqs)
# print(numhess - hess2)
# print(hess - hess2)
