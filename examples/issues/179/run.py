#!/usr/bin/env python3
# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
https://github.com/grimme-lab/dxtb/issues/179
"""
import torch

import dxtb
from dxtb.typing import DD

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

num1 = torch.tensor(
    [8, 1, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1],
    device=dd["device"],
)
num2 = torch.tensor(
    [8, 1, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1],
    device=dd["device"],
)

pos1 = torch.tensor(
    [
        [-7.68281384, 1.3350934, 0.74846383],
        [-5.7428588, 1.31513411, 0.36896714],
        [-8.23756184, -0.19765779, 1.67193897],
        [-8.13313558, 2.93710683, 1.6453921],
        [-2.95915993, 1.40005084, 0.24966306],
        [-2.1362031, 1.4795743, -1.38758999],
        [-2.40235213, 2.84218589, 1.24419946],
        [-8.2640369, 5.79677268, 2.54733192],
        [-8.68767571, 7.18194193, 1.3350556],
        [-9.27787497, 6.09327071, 4.03498102],
        [-9.34575393, -2.54164384, 3.28062124],
        [-8.59029812, -3.46388688, 4.6567765],
        [-10.71898011, -3.58163572, 2.65211723],
        [-9.5591796, 9.66793334, -0.53212042],
        [-8.70438089, 11.29169941, -0.5990394],
        [-11.12723654, 9.8483266, -1.43755624],
        [-2.69970054, 5.55135395, 2.96084179],
        [-1.59244386, 6.50972855, 4.06699298],
        [-4.38439138, 6.18065165, 3.1939773],
    ],
    **dd
)

pos2 = torch.tensor(
    [
        [-7.67436676, 1.33433562, 0.74512468],
        [-5.75285545, 1.30220838, 0.37189432],
        [-8.23155251, -0.20308887, 1.67397231],
        [-8.15184386, 2.94589406, 1.6474141],
        [-2.96380866, 1.39739578, 0.24572676],
        [-2.14413995, 1.48993378, -1.37321106],
        [-2.39808135, 2.86614761, 1.25247646],
        [-8.26855335, 5.79452391, 2.54948621],
        [-8.69277797, 7.18061912, 1.33247046],
        [-9.28819287, 6.08797948, 4.03809906],
        [-9.3377226, -2.54245643, 3.27861813],
        [-8.59693106, -3.48501402, 4.65503795],
        [-10.72627446, -3.59514726, 2.66139579],
        [-9.55955755, 9.6716561, -0.53106973],
        [-8.7077635, 11.28708848, -0.59527696],
        [-11.12540351, 9.87000175, -1.44181568],
        [-2.70194931, 5.55490663, 2.9641866],
        [-1.60305656, 6.49854138, 4.07984311],
        [-4.39083534, 6.17898869, 3.18702311],
    ],
    **dd
)

charge1 = torch.tensor(1, **dd)
charge2 = torch.tensor(1, **dd)


##############################################################################


def main() -> int:
    numbers = torch.stack([num1, num2])
    positions = torch.stack([pos1, pos2])
    charge = torch.tensor([charge1, charge2])

    # no conformers -> batched mode 1
    opts = {"verbosity": 0, "batch_mode": 1}

    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
    result = calc.energy(positions, chrg=charge)

    ###########################################################################

    calc = dxtb.Calculator(num1, dxtb.GFN1_XTB, opts={"verbosity": 0}, **dd)
    result1 = calc.energy(pos1, chrg=charge1)

    ###########################################################################

    calc = dxtb.Calculator(num2, dxtb.GFN1_XTB, opts={"verbosity": 0}, **dd)
    result2 = calc.energy(pos2, chrg=charge2)

    ###########################################################################

    assert torch.allclose(result[0], result1)
    assert torch.allclose(result[1], result2)

    print("Issue 179 test passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
