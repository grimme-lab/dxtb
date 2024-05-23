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
Parametrization: Electrostatics (3rd order)
===========================================

Definition of the isotropic third-order onsite correction.
"""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel

__all__ = ["ThirdOrderShell", "ThirdOrder"]


class ThirdOrderShell(BaseModel):
    """Representation of shell-resolved third-order electrostatics."""

    s: float
    """Scaling factor for s-orbitals."""

    p: float
    """Scaling factor for p-orbitals."""

    d: float
    """Scaling factor for d-orbitals."""


class ThirdOrder(BaseModel):
    """
    Representation of the isotropic third-order onsite correction.
    """

    shell: Union[bool, ThirdOrderShell] = False
    """
    Whether the third order contribution is shell-dependent or only atomwise.
    """
