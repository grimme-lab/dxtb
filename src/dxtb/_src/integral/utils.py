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
Integrals: Utility Functions
============================

Integral-related utility functions.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Tensor


def snorm(ovlp: Tensor) -> Tensor:
    d = ovlp.diagonal(dim1=-1, dim2=-2)
    zero = torch.tensor(0.0, dtype=d.dtype, device=d.device)
    return torch.where(d == 0.0, zero, torch.pow(d, -0.5))
