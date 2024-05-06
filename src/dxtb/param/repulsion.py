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
Parametrization: Repulsion
==========================

Definition of the repulsion contribution. The :class:`EffectiveRepulsion` is
used in GFN1-xTB and GFN2-xTB.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

__all__ = ["EffectiveRepulsion", "Repulsion"]


class EffectiveRepulsion(BaseModel):
    """
    Representation of the repulsion contribution for a parametrization.
    """

    kexp: float
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy.
    """

    klight: Optional[float] = None
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy for light elements, i.e., H and He (only GFN2).
    """


class Repulsion(BaseModel):
    """
    Possible repulsion parametrizations. Currently only the GFN1-xTB effective
    repulsion is supported.
    """

    effective: EffectiveRepulsion
    """Name of the represented method"""
