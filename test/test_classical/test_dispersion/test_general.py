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

from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.components.classicals.dispersion import new_dispersion
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
    """Only non-self-consistent dispersion requires a total charge."""
    _par2 = GFN2_XTB.model_copy(deep=True)
    _par2.dispersion.d4.sc = False  # type: ignore

    with pytest.raises(ValueError):
        new_dispersion(torch.tensor(0.0), _par2, charge=None)


def test_fail_no_dispersion() -> None:
    _par = GFN1_XTB.model_copy(deep=True)
    assert _par.dispersion is not None

    # set both to None
    _par.dispersion.d3 = None
    _par.dispersion.d4 = None
    assert new_dispersion(torch.tensor(0.0), _par) is None


def test_fail_wrong_sc_value() -> None:
    _par = GFN2_XTB.model_copy(deep=True)
    assert _par.dispersion is not None

    _par.dispersion.d4.sc = None  # type: ignore
    with pytest.raises(ValueError):
        new_dispersion(torch.tensor(0.0), _par)


def test_fail_too_many_parameters() -> None:
    _par = GFN1_XTB.model_copy(deep=True)
    _par2 = GFN2_XTB.model_copy(deep=True)

    assert _par.dispersion is not None
    assert _par2.dispersion is not None
    _par.dispersion.d4 = _par2.dispersion.d4

    with pytest.raises(ValueError):
        new_dispersion(torch.tensor(0.0), _par)


def test_fail_d4_cache() -> None:
    numbers = torch.tensor([3, 1])

    _par = GFN2_XTB.model_copy(deep=True)

    disp = new_dispersion(numbers, _par, torch.tensor(0.0))
    assert disp is not None

    with pytest.raises(TypeError):
        _ = disp.get_cache(numbers=numbers, model=0)

    with pytest.raises(TypeError):
        _ = disp.get_cache(numbers=numbers, rcov=0)

    with pytest.raises(TypeError):
        _ = disp.get_cache(numbers=numbers, r4r2=0)

    with pytest.raises(TypeError):
        _ = disp.get_cache(numbers=numbers, cutoff=0)


def test_d4_cache() -> None:
    numbers = torch.tensor([3, 1])

    _par2 = GFN2_XTB.model_copy(deep=True)
    _par2.dispersion.d4.sc = False  # type: ignore

    disp = new_dispersion(numbers, _par2, torch.tensor(0.0))
    assert disp is not None

    _ = disp.get_cache(numbers=numbers)
    assert disp.cache_is_latest((numbers.detach().clone(),))

    _ = disp.get_cache(numbers=numbers)
    assert disp.cache_is_latest((numbers.detach().clone(),))
