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
# NOTE: THANKS TO THE COURTESY OF https://github.com/tbmalt/tbmalt
"""
Utility: Batch
==============

Helper functions for batch operations.

This module contains classes and helper functions associated with batch
construction, handling and maintenance.
"""
from __future__ import annotations

from collections import namedtuple

import torch
from tad_mctc.batch import deflate, pack, unpack

from dxtb._src.typing import Any, Tensor, TensorOrTensors

__sort = namedtuple("sort", ("values", "indices"))

__all__ = ["pack", "pargsort", "psort", "merge", "deflate", "unpack"]


def pargsort(tensor: Tensor, mask: Tensor | None = None, dim: int = -1) -> Tensor:
    """Returns indices that sort packed tensors while ignoring padding values.

    Returns the indices that sorts the elements of ``tensor`` along ``dim`` in
    ascending order by value while ensuring padding values are shuffled to the
    end of the dimension.

    Arguments:
        tensor: the input tensor.
        mask: a boolean tensor which is True & False for "real" & padding
            values restively. [DEFAULT=None]
        dim: the dimension to sort along. [DEFAULT=-1]

    Returns:
        out: ``indices`` which along the dimension ``dim``.

    Notes:
        This will redirect to `torch.argsort` if no ``mask`` is supplied.
    """
    if mask is None:
        return torch.argsort(tensor, dim=dim)
    else:
        # A secondary sorter is used to reorder the primary sorter so that padding
        # values are moved to the end.
        n = tensor.shape[dim]
        s1 = tensor.argsort(dim)
        s2 = (
            torch.arange(n, device=tensor.device) + (~mask.gather(dim, s1) * n)
        ).argsort(dim)
        return s1.gather(dim, s2)


def psort(tensor: Tensor, mask: Tensor | None = None, dim: int = -1) -> __sort:
    """Sort a packed ``tensor`` while ignoring any padding values.

    Sorts the elements of ``tensor`` along ``dim`` in ascending order by value
    while ensuring padding values are shuffled to the end of the dimension.

    Arguments:
        tensor: the input tensor.
        mask: a boolean tensor which is True & False for "real" & padding
            values restively. [DEFAULT=None]
        dim: the dimension to sort along. [DEFAULT=-1]

    Returns:
        out: A namedtuple of (values, indices) is returned, where the values
             are the sorted values and indices are the indices of the elements
             in the original input tensor.

    Notes:
        This will redirect to `torch.sort` if no ``mask`` is supplied.
    """
    if mask is None:
        return torch.sort(tensor, dim=dim)
    else:
        indices = pargsort(tensor, mask, dim)
        return __sort(tensor.gather(dim, indices), indices)


def merge(tensors: TensorOrTensors, value: Any = 0, axis: int = 0) -> Tensor:
    """Merge two or more packed tensors into a single packed tensor.

    Arguments:
        tensors: Packed tensors which are to be merged.
        value: Value with which the tensor were/are to be padded. [DEFAULT=0]
        axis: Axis along which ``tensors`` are to be stacked. [DEFAULT=0]

    Returns:
        merged: The tensors ``tensors`` merged along the axis ``axis``.

    Warnings:
        Care must be taken to ensure the correct padding value is specified as
        erroneous behaviour may otherwise ensue. As the correct padding value
        cannot be reliably detected in situ it defaults to zero.
    """

    # Merging is performed along the 0'th axis internally. If a non-zero axis
    # is requested then tensors must be reshaped during input and output.
    if axis != 0:
        tensors = [t.transpose(0, axis) for t in tensors]

    # Tensor to merge into, filled with padding value.
    shapes = torch.tensor([i.shape for i in tensors])
    merged = torch.full(
        (shapes.sum(0)[0], *shapes.max(0).values[1:]),
        value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    n = 0  # <- batch dimension offset
    for src, size in zip(tensors, shapes):  # Assign values to tensor
        merged[(slice(n, size[0] + n), *[slice(0, s) for s in size[1:]])] = src
        n += size[0]

    # Return the merged tensor, transposing back as required
    return merged if axis == 0 else merged.transpose(0, axis)
