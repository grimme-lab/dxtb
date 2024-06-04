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
Parametrizations: Meta
======================

Meta data associated with a parametrization.
Mainly used for identification of data format.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

__all__ = ["Meta"]


class Meta(BaseModel):
    """
    Representation of the meta data for a parametrization.
    """

    name: Optional[str] = None
    """Name of the represented method."""

    reference: Optional[str] = None
    """References relevant for the parametrization records."""

    version: int = 0
    """Version of the represented method."""

    format: Optional[int] = None
    """Format version of the parametrization data."""
