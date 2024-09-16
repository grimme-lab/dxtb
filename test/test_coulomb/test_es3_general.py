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
Run tests for energy contribution from on-site third-order
electrostatic energy (ES3).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.components.interactions.coulomb import thirdorder as es3


def test_none() -> None:
    dummy = torch.tensor(0.0)
    par = GFN1_XTB.model_copy(deep=True)

    par.thirdorder = None
    assert es3.new_es3(dummy, par) is None

    del par.thirdorder
    assert es3.new_es3(dummy, par) is None


def test_fail() -> None:
    dummy = torch.tensor(0.0)

    par = GFN1_XTB.model_copy(deep=True)
    assert par.thirdorder is not None

    with pytest.raises(NotImplementedError):
        par.thirdorder.shell = True
        es3.new_es3(dummy, par)


def test_fail_cache_input() -> None:
    numbers = torch.tensor([3, 1])
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    es = es3.new_es3(numbers, GFN1_XTB)
    assert es is not None

    with pytest.raises(ValueError):
        es.get_cache(numbers=None, ihelp=ihelp)

    with pytest.raises(ValueError):
        es.get_cache(numbers=numbers, ihelp=None)


def test_fail_store() -> None:
    numbers = torch.tensor([3, 1])
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    es = es3.new_es3(numbers, GFN1_XTB)
    assert es is not None

    cache = es.get_cache(numbers=numbers, ihelp=ihelp)
    with pytest.raises(RuntimeError):
        cache.restore()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    cls = es3.new_es3(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    cls = cls.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    cls = es3.new_es3(torch.tensor(0.0), GFN1_XTB)
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
    device = str_to_device(device_str)
    cls = es3.new_es3(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    cls = cls.to(device)
    assert cls.device == device


def test_change_device_fail() -> None:
    cls = es3.new_es3(torch.tensor(0.0), GFN1_XTB)
    assert cls is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.device = "cpu"
