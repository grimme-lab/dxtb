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

from dxtb.__version__ import __tversion__
from dxtb._src.typing import Tensor

__all__ = ["t2int", "tensor_id"]


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


def tensor_id(x: Tensor) -> str:
    """
    Generate an identifier for a tensor based on its data pointer and version.
    """
    grad = int(x.requires_grad)
    v = x._version

    if __tversion__ >= (1, 13, 0) and torch._C._functorch.is_gradtrackingtensor(x):
        value = x
        while torch._C._functorch.is_gradtrackingtensor(value):
            value = torch._C._functorch.get_unwrapped(value)
        data = value.data_ptr()
    else:
        data = x.data_ptr()

    return f"tensor({data},v={v},grad={grad},dtype={x.dtype})"
