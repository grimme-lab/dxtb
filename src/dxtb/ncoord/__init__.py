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
Coordination Number
===================

Functions for calculating the coordination numbers.
"""

from tad_mctc.ncoord.count import (
    derf_count,
    dexp_count,
    dgfn2_count,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.ncoord.d3 import cn_d3, cn_d3_gradient
from tad_mctc.ncoord.d4 import cn_d4

from .utils import get_dcn

__all__ = [
    "cn_d3",
    "cn_d3_gradient",
    "cn_d4",
    "erf_count",
    "derf_count",
    "exp_count",
    "dexp_count",
    "gfn2_count",
    "dgfn2_count",
    "get_dcn",
]
