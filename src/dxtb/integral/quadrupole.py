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
Quadrupole integral.
"""

from __future__ import annotations

import torch

from .._types import Any
from ..constants import labels
from .base import BaseIntegral
from .driver.libcint import QuadrupoleLibcint


class Quadrupole(BaseIntegral):
    """
    Quadrupole integral from atomic orbitals.
    """

    integral: QuadrupoleLibcint
    """Instance of actual quadrupole integral type."""

    def __init__(
        self,
        driver: int = labels.INTDRIVER_LIBCINT,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ):
        super().__init__(device=device, dtype=dtype)

        # Determine which overlap class to instantiate based on the type
        if driver == labels.INTDRIVER_LIBCINT:
            self.integral = QuadrupoleLibcint(device=device, dtype=dtype, **kwargs)
        elif driver in (labels.INTDRIVER_ANALYTICAL, labels.INTDRIVER_AUTOGRAD):
            raise NotImplementedError(
                "PyTorch versions of multipole moments are not implemented. "
                "Use `libcint` as integral driver."
            )
        else:
            raise ValueError(f"Unknown integral driver '{driver}'.")
