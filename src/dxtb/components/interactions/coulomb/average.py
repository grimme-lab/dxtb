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
Averaging functions for hardnesses in GFN1-xTB.
"""

from __future__ import annotations

import torch
from tad_mctc.typing import Callable, Tensor

AveragingFunction = Callable[[Tensor], Tensor]


def harmonic_average(hubbard: Tensor) -> Tensor:
    """
    Harmonic averaging function for hardnesses in GFN1-xTB.

    Parameters
    ----------
    hubbard : Tensor
        Hubbard parameters of all elements.

    Returns
    -------
    Tensor
        Harmonic average of the Hubbard parameters.
    """
    hubbard1 = 1.0 / hubbard
    return 2.0 / (hubbard1.unsqueeze(-1) + hubbard1.unsqueeze(-2))


def arithmetic_average(hubbard: Tensor) -> Tensor:
    """
    Arithmetic averaging function for hardnesses in GFN1-xTB.

    Parameters
    ----------
    hubbard : Tensor
        Hubbard parameters of all elements.

    Returns
    -------
    Tensor
        Arithmetic average of the Hubbard parameters.
    """
    return 0.5 * (hubbard.unsqueeze(-1) + hubbard.unsqueeze(-2))


def geometric_average(hubbard: Tensor) -> Tensor:
    """
    Geometric average function for hardnesses in GFN1-xTB.

    Parameters
    ----------
    hubbard : Tensor
        Hubbard parameters of all elements.

    Returns
    -------
    Tensor
        Geometric average of the Hubbard parameters.
    """
    h = hubbard.unsqueeze(-1) * hubbard.unsqueeze(-2)
    return torch.sqrt(h + torch.finfo(hubbard.dtype).eps)


averaging_function: dict[str, AveragingFunction] = {
    "arithmetic": arithmetic_average,
    "geometric": geometric_average,
    "harmonic": harmonic_average,
}
"""Available averaging functions for Hubbard parameters"""
