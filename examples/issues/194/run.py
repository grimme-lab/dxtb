import torch

import dxtb
from dxtb.typing import DD

dd: DD = {"device": torch.device("cuda:0"), "dtype": torch.double}


numbers = torch.tensor(
    [
        [3, 1, 0],
        [8, 1, 1],
    ],
    device=dd["device"],
)
positions = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
    ],
    **dd
)
positions.requires_grad_(True)
charge = torch.tensor([0, 0], **dd)


# no conformers -> batched mode 1
opts = {"verbosity": 6, "batch_mode": 1, "scf_mode": dxtb.labels.SCF_MODE_FULL}

dxtb.timer.reset()

calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
result = calc.energy(positions, chrg=charge)

dxtb.timer.print(v=-999)
