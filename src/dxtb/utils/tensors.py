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
Utility: Tensor Ops
===================

Collection of utility functions for matrices/tensors.
"""

from __future__ import annotations

import torch

from .._types import Tensor


def t2int(x: Tensor) -> int:
    """
    Convert tensor to int.

    Parameters
    ----------
    x : Tensor
        Tensor to convert.

    Returns
    -------
    int
        Integer value of the tensor.
    """
    return int(x.item())


def symmetrize(x: Tensor, force: bool = False) -> Tensor:
    """
    Symmetrize a tensor after checking if it is symmetric within a threshold.

    Parameters
    ----------
    x : Tensor
        Tensor to check and symmetrize.
    force : bool
        Whether symmetry should be forced. This allows switching between actual
        symmetrizing and only cleaning up numerical noise. Defaults to `False`.

    Returns
    -------
    Tensor
        Symmetrized tensor.

    Raises
    ------
    RuntimeError
        If the tensor is not symmetric within the threshold.
    """
    try:
        atol = torch.finfo(x.dtype).eps * 10
    except TypeError:
        atol = 1e-5

    if x.ndim < 2:
        raise RuntimeError("Only matrices and batches thereof allowed.")

    if force is False:
        if not torch.allclose(x, x.mT, atol=atol):
            raise RuntimeError(
                f"Matrix appears to be not symmetric (atol={atol:.3e}, "
                f"dtype={x.dtype})."
            )

    return (x + x.mT) / 2
