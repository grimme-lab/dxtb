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
https://github.com/grimme-lab/dxtb/issues/194
"""
import torch

import dxtb
from dxtb.typing import DD


def main() -> int:
    if not torch.cuda.is_available():
        print("Skipping test as CUDA is not available.")
        return 0

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
    opts = {
        "verbosity": 6,
        "batch_mode": 1,
        "scf_mode": dxtb.labels.SCF_MODE_FULL,
    }

    dxtb.timer.reset()

    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
    _ = calc.energy(positions, chrg=charge)

    dxtb.timer.print(v=-999)

    print("Issue 194 test passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
