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
Components: Fields
==================

Tight-binding components for interactions with external fields.
"""

from dxtb._src.components.interactions.field import ElectricField as ElectricField
from dxtb._src.components.interactions.field import (
    ElectricFieldGrad as ElectricFieldGrad,
)
from dxtb._src.components.interactions.field import new_efield as new_efield
from dxtb._src.components.interactions.field import new_efield_grad as new_efield_grad

__all__ = [
    "ElectricField",
    "ElectricFieldGrad",
    "new_efield",
    "new_efield_grad",
]
