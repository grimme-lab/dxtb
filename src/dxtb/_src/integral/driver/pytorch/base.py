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
Implementation: Base Classes
============================

Base class for ``PyTorch``-based drivers and integral implementations.
"""

from __future__ import annotations

from dxtb._src.typing import Literal

from ...base import BaseIntegral

__all__ = ["IntegralPytorch"]


class PytorchImplementation:
    """
    Simple label for ``PyTorch``-based integral implementations.
    """

    family: Literal["PyTorch"] = "PyTorch"
    """Label for integral implementation family."""


class IntegralPytorch(PytorchImplementation, BaseIntegral):
    """
    ``PyTorch``-based integral implementation.
    """
