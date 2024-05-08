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
Overlap
=======

The GFNn-xTB overlap matrix.
"""

from __future__ import annotations

import torch

from dxtb._src.constants import labels
from dxtb._src.typing import TYPE_CHECKING, Any

from ..factory import new_overlap
from .base import BaseIntegral

if TYPE_CHECKING:
    from ..driver.libcint import OverlapLibcint
    from ..driver.pytorch import OverlapPytorch

__all__ = ["Overlap"]


class Overlap(BaseIntegral):
    """
    Overlap integral from atomic orbitals.
    """

    integral: OverlapLibcint | OverlapPytorch
    """Instance of actual overlap integral type."""

    __slots__ = ["integral"]

    def __init__(
        self,
        driver: int = labels.INTDRIVER_LIBCINT,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

        self.integral = new_overlap(driver, device=device, dtype=dtype, **kwargs)
