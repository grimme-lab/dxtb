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
Test overlap build from integral container.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, IndexHelper
from dxtb.integrals.factories import new_hcore

numbers = torch.tensor([14, 1, 1, 1, 1])


def test_fail() -> None:
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    with pytest.raises(ValueError):
        par1 = GFN1_XTB.model_copy(deep=True)
        assert par1.meta is not None

        par1.meta.name = "fail"
        new_hcore(numbers, par1, ihelp)
