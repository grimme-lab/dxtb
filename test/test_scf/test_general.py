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
General tests for SCF setup.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, OutputHandler, labels
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.scf.implicit import SelfConsistentFieldImplicit as SCF
from dxtb._src.scf.mixer import Anderson
from dxtb._src.scf.unrolling import SelfConsistentFieldFull
from dxtb.calculators import EnergyCalculator
from dxtb.components.base import InteractionList
from dxtb.config import ConfigSCF


def test_properties() -> None:
    d = torch.randn((3, 3))  # dummy

    ints = IntegralMatrices()
    with pytest.raises(RuntimeError):
        SCF(d, d, d, d, d, d, integrals=ints)  # type: ignore

    ints.hcore = torch.randn((3, 3))
    with pytest.raises(RuntimeError):
        SCF(d, d, d, d, d, d, integrals=ints)  # type: ignore

    ints.overlap = torch.randn((3, 3))
    scf = SCF(d, d, d, d, d, d, integrals=ints)  # type: ignore
    assert scf.shape == d.shape
    assert scf.device == d.device
    assert scf.dtype == d.dtype


def test_fail() -> None:
    numbers = torch.tensor([1])
    positions = torch.tensor([[0.0, 0.0, 0.0]])

    calc = EnergyCalculator(numbers, GFN1_XTB)

    with pytest.raises(ValueError):
        calc.opts.scf.scf_mode = -1
        calc.singlepoint(positions)

    with pytest.raises(ValueError):
        calc.opts.scf.scf_mode = "fail"  # type: ignore
        calc.singlepoint(positions)


def test_full_mixer_error() -> None:
    ilist = InteractionList()
    kwargs = {
        "numbers": torch.tensor([1]),
        "occupation": torch.tensor([1]),
        "n0": torch.tensor([1]),
        "ihelp": torch.tensor([1]),
        "cache": torch.tensor([1]),
        "integrals": IntegralMatrices(),
    }

    with pytest.raises(TypeError):
        config = ConfigSCF(mixer=Anderson())  # type: ignore
        SelfConsistentFieldFull(ilist, *kwargs, config=config)


def test_full_change_scp() -> None:
    ilist = InteractionList()

    dummy = torch.tensor([1], dtype=torch.float)
    kwargs = {
        "numbers": dummy,
        "occupation": dummy,
        "n0": dummy,
        "ihelp": dummy,
        "cache": dummy,
        "integrals": IntegralMatrices(_hcore=dummy, _overlap=dummy),
    }

    config = ConfigSCF(
        method=labels.GFN2_XTB,
        batch_mode=2,
        scp_mode=labels.SCP_MODE_CHARGE,
        # Broyden mixer is not supported in full SCF
        scf_mode=labels.SCF_MODE_IMPLICIT_NON_PURE,
    )

    # Clear warnings from previous tests
    OutputHandler.clear_warnings()

    ##########################################################################

    _ = SelfConsistentFieldFull(ilist, **kwargs, config=config)

    assert len(OutputHandler.warnings) == 1

    warn_msg, warn_type = OutputHandler.warnings[0]
    assert "Changing to Fock matrix automatically." in warn_msg
    assert warn_type is UserWarning

    OutputHandler.clear_warnings()

    ##########################################################################

    config.scp_mode = labels.SCP_MODE_FOCK
    _ = SelfConsistentFieldFull(ilist, **kwargs, config=config)

    assert len(OutputHandler.warnings) == 0
