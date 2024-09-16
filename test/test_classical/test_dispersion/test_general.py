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
Testing dispersion module.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB
from dxtb._src.components.classicals.dispersion import new_dispersion
from dxtb._src.param.gfn2 import GFN2_XTB
from dxtb._src.typing.exceptions import ParameterWarning


def test_none() -> None:
    dummy = torch.tensor(0.0)
    _par1 = GFN1_XTB.model_copy(deep=True)
    _par2 = GFN2_XTB.model_copy(deep=True)

    with pytest.warns(ParameterWarning):
        _par1.dispersion = None
        assert new_dispersion(dummy, _par1) is None

        del _par1.dispersion
        assert new_dispersion(dummy, _par1) is None

        _par2.dispersion = None
        assert new_dispersion(dummy, _par2) is None

        del _par2.dispersion
        assert new_dispersion(dummy, _par2) is None


def test_fail_charge() -> None:
    with pytest.raises(ValueError):
        new_dispersion(torch.tensor(0.0), GFN2_XTB, charge=None)


def test_fail_no_dispersion() -> None:
    _par = GFN1_XTB.model_copy(deep=True)
    assert _par.dispersion is not None

    # set both to None
    _par.dispersion.d3 = None
    _par.dispersion.d4 = None
    assert new_dispersion(torch.tensor(0.0), _par) is None


def test_fail_too_many_parameters() -> None:
    _par = GFN1_XTB.model_copy(deep=True)
    _par2 = GFN2_XTB.model_copy(deep=True)

    assert _par.dispersion is not None
    assert _par2.dispersion is not None
    _par.dispersion.d4 = _par2.dispersion.d4

    with pytest.raises(ValueError):
        new_dispersion(torch.tensor(0.0), _par)
