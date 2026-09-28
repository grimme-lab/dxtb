# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Parametrization: Electronegativity equilibration
================================================

Definition of the non-self-consistent electronegativity-equilibration model.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["PEEQ"]


class PEEQ(BaseModel):
    """Settings for an electronegativity-equilibration (EEQ) model."""

    cn: str
    """Coordination-number counting function."""

    cutoff: float = 40.0
    """Coordination-number cutoff in Bohr."""

    cn_max: float = 8.0
    """Smooth upper bound applied to the coordination number."""

    kcn: float = 7.5
    """Steepness of the coordination-number counting function."""
