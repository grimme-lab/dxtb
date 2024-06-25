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
Calculation of coordination number with various counting functions.
"""

from __future__ import annotations

from tad_mctc.math import einsum

from dxtb._src.typing import Tensor

__all__ = ["get_dcn"]


def get_dcn(dcndr: Tensor, dedcn: Tensor) -> Tensor:
    """
    Calculate complete derivative for coordination number.

    Parameters
    ----------
    dcndr : Tensor
        Derivative of CN with resprect to atomic positions.
        Shape: (batch, natoms, natoms, 3)
    dedcn : Tensor
        Derivative of energy with respect to CN.
        Shape: (batch, natoms, 3)

    Returns
    -------
    Tensor
        Gradient originating from the coordination number.
    """

    # same atom terms added separately (missing due to mask)
    return einsum("...ijx, ...j -> ...ix", -dcndr, dedcn) + (
        (-dcndr).sum(-2) * dedcn.unsqueeze(-1)
    )
