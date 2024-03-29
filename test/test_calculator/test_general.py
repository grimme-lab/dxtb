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
Test Calculator setup.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.exceptions import DtypeError

from dxtb.param import GFN1_XTB as par
from dxtb.timing import timer
from dxtb.xtb import Calculator


def test_fail() -> None:
    numbers = torch.tensor([6, 1, 1, 1, 1], dtype=torch.double)

    with pytest.raises(DtypeError):
        Calculator(numbers, par)

    # because of the exception, the timer for the setup is never stopped
    timer.reset()
