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
Core Hamiltonian.
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.param import Param
from dxtb._src.typing import Tensor
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian
from dxtb._src.xtb.gfn2 import GFN2Hamiltonian

from .base import BaseIntegral

__all__ = ["HCore"]


class HCore(BaseIntegral):
    """
    Hamiltonian integral.
    """

    integral: GFN1Hamiltonian | GFN2Hamiltonian
    """Instance of actual GFN Hamiltonian integral."""

    __slots__ = ["integral"]

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device=device, dtype=dtype)

        if par.meta is not None:
            if par.meta.name is not None:
                if par.meta.name.casefold() == "gfn1-xtb":
                    self.integral = GFN1Hamiltonian(
                        numbers, par, ihelp, device=device, dtype=dtype
                    )
                elif par.meta.name.casefold() == "gfn2-xtb":
                    self.integral = GFN2Hamiltonian(
                        numbers, par, ihelp, device=device, dtype=dtype
                    )
                else:
                    raise ValueError(f"Unsupported Hamiltonian type: {par.meta.name}")
