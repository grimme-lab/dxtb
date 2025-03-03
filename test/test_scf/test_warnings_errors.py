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
"""Test warnings and errors emitted by the SCF."""

from __future__ import annotations

import pytest
import torch
from tad_mctc import read

from dxtb import GFN1_XTB, Calculator, OutputHandler
from dxtb._src.constants import labels
from dxtb._src.typing import DD
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)

from ..conftest import DEVICE
from ..utils import coordfile_lih

opts = {
    "verbosity": 0,
    "maxiter": 50,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_scf_full_unconverged_warning(dtype: torch.dtype) -> None:
    # avoid pollution from previous tests
    OutputHandler.clear_warnings()

    dd: DD = {"device": DEVICE, "dtype": dtype}

    maxiter = 3
    opts = {"scf_mode": "full", "maxiter": maxiter, "verbosity": 0}

    numbers, positions = read(coordfile_lih, **dd)
    calc = Calculator(numbers, GFN1_XTB, opts=opts, **dd)

    energy = calc.get_energy(positions).sum(-1)
    assert pytest.approx(-0.8806441129803488) == energy.cpu()

    # warnings

    assert len(OutputHandler.warnings) == 2

    warning_message, warning_type = OutputHandler.warnings[0]
    assert warning_type is UserWarning
    assert "Broyden" in warning_message and "Anderson" in warning_message

    warning_message, warning_type = OutputHandler.warnings[1]
    assert warning_type is SCFConvergenceWarning
    assert f"after {maxiter} cycles" in warning_message

    # avoid pollution for other tests
    OutputHandler.clear_warnings()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_scf_full_unconverged_error(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    maxiter = 3
    opts = {
        "scf_mode": "full",
        "maxiter": maxiter,
        "verbosity": 0,
        "force_convergence": True,
    }

    numbers, positions = read(coordfile_lih, **dd)
    calc = Calculator(numbers, GFN1_XTB, opts=opts, **dd)

    with pytest.raises(SCFConvergenceError):
        calc.get_energy(positions)
