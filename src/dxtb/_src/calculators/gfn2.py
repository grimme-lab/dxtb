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
Calculators: GFN2
=================

Calculator for the second generation extended tight-binding model (GFN2-xTB).
"""
from __future__ import annotations

import torch

from dxtb._src.components.classicals import Classical
from dxtb._src.components.interactions import Interaction
from dxtb._src.typing import Any, Tensor
from dxtb.config import Config

from .base import Calculator

__all__ = ["GFN2Calculator"]


class GFN2Calculator(Calculator):
    """
    Calculator for the GFN2-xTB method.

    This is a simple wrapper around the :class:`dxtb.Calculator` class with the
    :data:`GFN2-xTB <dxtb.GFN2_XTB>` parameters passed in as defaults.
    """

    def __init__(
        self,
        numbers: Tensor,
        *,
        classical: list[Classical] | tuple[Classical] | Classical | None = None,
        interaction: list[Interaction] | tuple[Interaction] | Interaction | None = None,
        opts: dict[str, Any] | Config | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # pylint: disable=import-outside-toplevel
        from dxtb import GFN2_XTB

        super().__init__(
            numbers,
            GFN2_XTB,
            classical=classical,
            interaction=interaction,
            opts=opts,
            device=device,
            dtype=dtype,
        )

        raise NotImplementedError("GFN2-xTB is not yet implemented.")
