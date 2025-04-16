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
General halogen bond correction tests
=====================================

Run general tests for halogen bond correction including:
 - invalid parameters
 - change of `dtype` and `device`
"""

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN1_XTB as par
from dxtb._src.components.classicals import new_halogen


def test_none() -> None:
    """Test that the HB correction is set to None if deleted."""
    dummy = torch.tensor([0.0])
    _par = par.model_copy(deep=True)

    _par.halogen = None
    assert new_halogen(dummy, _par) is None

    del _par.halogen
    assert new_halogen(dummy, _par) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    """Test changing the dtype of the halogen bond correction."""
    cls = new_halogen(torch.tensor([0.0]), par)
    assert cls is not None

    cls = cls.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    """Test changing the dtype of the halogen bond correction."""
    cls = new_halogen(torch.tensor([0.0]), par)
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
    """Test changing the device of the halogen bond correction."""
    device = str_to_device(device_str)
    cls = new_halogen(torch.tensor([0.0]), par)
    assert cls is not None

    cls = cls.to(device)
    assert cls.device == device


def test_change_device_fail() -> None:
    """Test failure of changing the device of the halogen bond correction."""
    cls = new_halogen(torch.tensor([0.0]), par)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.device = "cpu"


def test_fail_requires_ihelp() -> None:
    """Test failure if `ihelp` is not provided."""
    numbers = torch.tensor([3, 1])
    cls = new_halogen(numbers, par)
    assert cls is not None

    with pytest.raises(ValueError):
        cls.get_cache(numbers=numbers, ihelp=None)
