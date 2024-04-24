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
Calculators: GFN1
=================

Calculator for the first generation extended tight-binding model (GFN1-xTB).
"""

import torch

from dxtb.components.classicals import Classical
from dxtb.components.interactions import Interaction
from dxtb.config import Config
from dxtb.typing import Any, Sequence, Tensor

from .base import BaseCalculator


class GFN1Calculator(BaseCalculator):
    """
    Calculator for the GFN1-xTB method.

    This is a simple wrapper around the
    :class:`dxtb.xtb.calculators.BaseCalculator` class with the GFN1-xTB
    parameters passed in as defaults.
    """

    def __init__(
        self,
        numbers: Tensor,
        *,
        classical: Sequence[Classical] | None = None,
        interaction: Sequence[Interaction] | None = None,
        opts: dict[str, Any] | Config | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # pylint: disable=import-outside-toplevel
        from dxtb.param import GFN1_XTB

        super().__init__(
            numbers,
            GFN1_XTB,
            classical=classical,
            interaction=interaction,
            opts=opts,
            device=device,
            dtype=dtype,
        )
