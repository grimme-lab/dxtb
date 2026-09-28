# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
General Shor-range bond correction tests
=====================================

Run general tests for short range bond correction including:
 - invalid parameters
 - change of `dtype` and `device`
"""

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN0_XTB as par
from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.components.classicals import new_srb


def test_none() -> None:
    """Test that SRB is set to None if its parameters are deleted."""
    dummy = torch.tensor([0.0])
    _par = par.model_copy(deep=True)

    assert _par.short_range is not None

    _par.short_range = None
    assert new_srb(dummy, _par) is None

    del _par.short_range
    assert new_srb(dummy, _par) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    """Test changing the dtype of the srb correction."""
    cls = new_srb(torch.tensor([0.0]), par)
    assert cls is not None

    cls = cls.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    """Test changing the dtype of the srb correction."""
    cls = new_srb(torch.tensor([0.0]), par)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        cls.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    """Test changing the device of the srb correction."""
    device = str_to_device(device_str)
    cls = new_srb(torch.tensor([0.0]), par)
    assert cls is not None

    cls = cls.to(device)
    assert cls.device == device


def test_change_device_fail() -> None:
    """Test failure of changing the device of the srb correction."""
    cls = new_srb(torch.tensor([0.0]), par)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.device = "cpu"


def test_factory_selection() -> None:
    """Leave GFN1/GFN2 unchanged when no SRB block is present."""
    unique = torch.tensor([1, 6])
    assert new_srb(unique, GFN1_XTB) is None
    assert new_srb(unique, GFN2_XTB) is None


def test_fail_requires_ihelp() -> None:
    """Test failure if `ihelp` is not provided."""
    numbers = torch.tensor([3, 1])
    cls = new_srb(numbers, par)
    assert cls is not None

    with pytest.raises(ValueError):
        cls.get_cache(numbers=numbers, ihelp=None)
