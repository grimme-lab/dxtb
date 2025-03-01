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
https://github.com/grimme-lab/dxtb/issues/123
"""
from __future__ import annotations

from pathlib import Path

import torch
from tad_mctc import read, read_chrg

import dxtb


def main() -> int:
    dd = {"device": torch.device("cpu"), "dtype": torch.double}
    dxtb.timer.cuda_sync = False

    # read molecule from file
    f = Path(__file__).parent / "coord"
    numbers, positions = read(f, **dd)
    charge = read_chrg(f, **dd)

    opts = {"verbosity": 6, "scp_mode": dxtb.labels.SCP_MODE_CHARGE}
    calc = dxtb.calculators.GFN1Calculator(numbers, opts=opts, **dd)

    ref = -35.800705002354
    energy = calc.get_energy(positions, chrg=charge)

    print("\nComparison to xtb:")
    print(f" - xtb (6.7.1) : {ref: .12f}")
    print(f" - dxtb (0.1.0): {energy: .12f}")
    print(f" - diff        : {energy - ref: .12f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
