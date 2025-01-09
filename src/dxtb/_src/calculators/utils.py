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
Calculators: Utility
====================

Collection of utility functions for the calculator.
"""

from __future__ import annotations

from dxtb._src.typing import Literal, NoReturn, Tensor

__all__ = ["shape_checks_chrg"]


def shape_checks_chrg(
    t: Tensor, ndims: int, name: str = "Charge"
) -> Literal[True] | NoReturn:
    """
    Check the shape of a tensor.

    Parameters
    ----------
    t : Tensor
        The tensor to check.
    ndims : int
        The number of dimensions indicating single or batched calculations.

    Raises
    ------
    ValueError
        If the tensor has not the expected number of dimensions.
    """

    if t.ndim > 1:
        raise ValueError(
            f"{name.title()} tensor has more than 1 dimension. "
            "Please use a 1D tensor for batched calculations "
            "(e.g., `torch.tensor([1.0, 0.0])`), instead of "
            "a 2D tensor (e.g., NOT `torch.tensor([[1.0], [0.0]])`)."
        )

    if t.ndim == 1 and t.numel() == 1:
        raise ValueError(
            f"{name.title()} tensor has only one element. Please use a "
            "scalar for single structures (e.g., `torch.tensor(1.0)`) and "
            "a 1D tensor for batched calculations (e.g., "
        )

    if ndims != t.ndim + 1:
        raise ValueError(
            f"{name.title()} tensor has invalid shape: {t.shape}.\n"
            "Please use a scalar for single structures (e.g., "
            "`torch.tensor(1.0)`) and a 1D tensor for batched "
            "calculations (e.g., `torch.tensor([1.0, 0.0])`)."
        )

    return True
