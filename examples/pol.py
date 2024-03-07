from pathlib import Path

import torch
from tad_mctc.io import read
from tad_mctc.typing import DD

import dxtb

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

# init a molecule
f = Path(__file__).parent / "molecules" / "h2o-dimer.coord"
numbers, positions = read.read_from_path(f, ftype="tm", **dd)
charge = torch.tensor(0.0, **dd)

# setup electric field interaction for field derivatives
field_vector = torch.tensor([0.0, 0.0, 0.0], **dd, requires_grad=True)
ef = dxtb.external.new_efield(field_vector)

# setup calculator
opts = {"scf_mode": "full", "mixer": "anderson", "verbosity": 6}
calc = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    opts=opts,
    interaction=[ef],
    **dd,
)

pol = calc.polarizability(numbers, positions, charge)
print(pol)
