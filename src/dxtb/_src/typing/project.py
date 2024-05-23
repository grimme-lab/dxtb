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
Typing: Project
===============

Project-specific type annotations.
"""
from __future__ import annotations

import torch

from .builtin import TypedDict
from .compat import Slicer

__all__ = ["ContainerData", "Slicers"]


class Slicers(TypedDict):
    """Collection of slicers of different resolutions for culling in SCF."""

    orbital: Slicer
    """Slicer for orbital-resolved variables."""
    shell: Slicer
    """Slicer for shell-resolved variables."""
    atom: Slicer
    """Slicer for atom-resolved variables."""


class ContainerData(TypedDict):
    """Shape and label information of Potentials."""

    mono: torch.Size | None
    """Shape of the monopolar potential."""

    dipole: torch.Size | None
    """Shape of the dipolar potential."""

    quad: torch.Size | None
    """Shape of the quadrupolar potential."""

    label: list[str] | str | None
    """Labels for the interactions contributing to the potential."""
