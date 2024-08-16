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
Integral Driver: PyTorch
========================

Pytorch-based integral implementations.
"""

from .dipole import DipolePytorch
from .driver import (
    IntDriverPytorch,
    IntDriverPytorchLegacy,
    IntDriverPytorchNoAnalytical,
)
from .overlap import OverlapPytorch
from .quadrupole import QuadrupolePytorch

__all__ = [
    "OverlapPytorch",
    "DipolePytorch",
    "QuadrupolePytorch",
    "IntDriverPytorch",
    "IntDriverPytorchLegacy",
    "IntDriverPytorchNoAnalytical",
]
