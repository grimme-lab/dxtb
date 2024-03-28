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


dxtb.timer.start("HyperPol")
agrad = calc.hyperpol(numbers, positions, charge)
calc.interactions.update_efield(field=field_vector)
dxtb.timer.stop("HyperPol")

print(agrad.shape)

dxtb.timer.print()
dxtb.timer.reset()


dxtb.timer.start("HyperPol2")
agrad2 = calc.hyperpol(numbers, positions, charge, use_functorch=True)
calc.interactions.update_efield(field=field_vector)
dxtb.timer.stop("HyperPol2")

print(agrad2.shape)


dxtb.timer.start("HyperPol3")
agrad3 = calc.hyperpol(
    numbers, positions, charge, use_functorch=True, derived_quantity="dipole"
)
calc.interactions.update_efield(field=field_vector)
dxtb.timer.stop("HyperPol3")

print(agrad3.shape)

dxtb.timer.start("HyperPol4")
agrad4 = calc.hyperpol(
    numbers, positions, charge, use_functorch=True, derived_quantity="energy"
)
dxtb.timer.stop("HyperPol4")

print(agrad4.shape)


dxtb.timer.start("Num HyperPol")
num = calc.hyperpol_numerical(numbers, positions, charge)
dxtb.timer.stop("Num HyperPol")

dxtb.timer.print()
dxtb.timer.reset()

print("agrad\n", agrad)
print("agrad2\n", agrad2)
print("agrad3\n", agrad3)
print("agrad4\n", agrad4)
print("num\n", num)
print()
print(num - agrad)
print(num - agrad2)
print(num - agrad3)
print(num - agrad4)
