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
Utility function for overlap calculation.
"""

from __future__ import annotations

from dxtb._types import DD, Literal, Tensor
from dxtb.basis import IndexHelper
from dxtb.integral.driver.pytorch import IntDriverPytorch as IntDriver
from dxtb.integral.driver.pytorch import OverlapPytorch as Overlap
from dxtb.param import Param


def calc_overlap(
    numbers: Tensor,
    positions: Tensor,
    par: Param,
    dd: DD,
    uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
) -> Tensor:
    ihelp = IndexHelper.from_numbers(numbers, par)
    driver = IntDriver(numbers, par, ihelp, **dd)
    overlap = Overlap(uplo=uplo, **dd)

    driver.setup(positions)
    return overlap.build(driver)
