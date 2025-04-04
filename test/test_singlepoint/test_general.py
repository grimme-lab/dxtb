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
Run tests for singlepoint calculation with read from coord file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tad_mctc import read, read_chrg

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.timing import timer

from ..conftest import DEVICE

opts = {"verbosity": 0, "int_level": 2}


@pytest.mark.filterwarnings("ignore::tad_mctc.exceptions.MoleculeWarning")
def test_uhf_fail() -> None:
    # Singlepoint starts SCF timer, but exception is thrown before the SCF
    # timer is stopped, so we must disable it here.
    status = timer._enabled
    if status is True:
        timer.disable()

    base = Path(Path(__file__).parent, "mols", "H")

    numbers, positions = read(Path(base, "coord"), device=DEVICE)
    charge = read_chrg(Path(base, ".CHRG"), device=DEVICE)

    calc = Calculator(numbers, par, opts=opts, device=DEVICE)

    with pytest.raises(ValueError):
        calc.singlepoint(positions, charge, spin=0)

    with pytest.raises(ValueError):
        calc.singlepoint(positions, charge, spin=2)

    if status is True:
        timer.enable()
