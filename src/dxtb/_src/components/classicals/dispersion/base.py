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
Dispersion: ABC
===============

Abstract base class for dispersion models.
"""

from __future__ import annotations

from abc import abstractmethod

import torch

from dxtb._src.typing import Any, Tensor

from ...classicals import Classical, ClassicalCache

__all__ = ["Dispersion"]


class Dispersion(Classical):
    """
    Base class for dispersion correction.
    """

    numbers: Tensor
    """Atomic numbers for all atoms in the system (shape: ``(..., nat)``)."""

    param: dict[str, Tensor]
    """Dispersion parameters."""

    charge: Tensor | None
    """Total charge of the system."""

    __slots__ = ["numbers", "param", "charge"]

    def __init__(
        self,
        numbers: Tensor,
        param: dict[str, Tensor],
        charge: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.numbers = numbers
        self.param = param
        self.charge = charge

    @abstractmethod
    def get_cache(self, numbers: Tensor, **kwargs: Any) -> ClassicalCache:
        """
        Store variables for energy calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).

        Returns
        -------
        Cache
            Cache class for storage of variables.
        """
