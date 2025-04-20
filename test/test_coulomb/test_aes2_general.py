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
Test general setup behavior of multipole electrostatic energy (AES2).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.exceptions import DeviceError
from tad_mctc.typing import MockTensor

from dxtb import GFN2_XTB
from dxtb._src.components.interactions.coulomb import new_aes2


def test_device_fail_numbers() -> None:
    """Test failure upon invalid device of numbers."""
    n = torch.tensor([3, 1], dtype=torch.float, device="cpu")
    numbers = MockTensor(n)
    numbers.device = "cuda"

    # works
    _ = new_aes2(n, GFN2_XTB, device=torch.device("cpu"))

    # fails
    with pytest.raises(DeviceError):
        new_aes2(numbers, GFN2_XTB, device=torch.device("cpu"))
