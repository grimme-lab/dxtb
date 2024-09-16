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
Test factories for integral drivers.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, labels
from dxtb._src.integral.driver import factory
from dxtb._src.integral.driver.libcint import IntDriverLibcint
from dxtb._src.integral.driver.pytorch import (
    IntDriverPytorch,
    IntDriverPytorchLegacy,
    IntDriverPytorchNoAnalytical,
)

numbers = torch.tensor([14, 1, 1, 1, 1])


def test_fail() -> None:
    with pytest.raises(ValueError):
        factory.new_driver(-1, numbers, GFN1_XTB)


def test_driver_libcint() -> None:
    cls = factory.new_driver(labels.INTDRIVER_LIBCINT, numbers, GFN1_XTB)
    assert isinstance(cls, IntDriverLibcint)


def test_driver_pytorch() -> None:
    cls = factory.new_driver(labels.INTDRIVER_ANALYTICAL, numbers, GFN1_XTB)
    assert isinstance(cls, IntDriverPytorch)

    cls = factory.new_driver(labels.INTDRIVER_AUTOGRAD, numbers, GFN1_XTB)
    assert isinstance(cls, IntDriverPytorchNoAnalytical)

    cls = factory.new_driver(labels.INTDRIVER_LEGACY, numbers, GFN1_XTB)
    assert isinstance(cls, IntDriverPytorchLegacy)


def test_factory_libcint() -> None:
    cls = factory.new_driver_libcint(numbers, GFN1_XTB)
    assert isinstance(cls, IntDriverLibcint)


def test_factory_pytorch() -> None:
    cls = factory.new_driver_pytorch(numbers, GFN1_XTB)
    assert isinstance(cls, IntDriverPytorch)

    cls = factory.new_driver_pytorch_no_analytical(numbers, GFN1_XTB)
    assert isinstance(cls, IntDriverPytorchNoAnalytical)

    cls = factory.new_driver_legacy(numbers, GFN1_XTB)
    assert isinstance(cls, IntDriverPytorchLegacy)
