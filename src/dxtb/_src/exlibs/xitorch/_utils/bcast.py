# This file is part of dxtb, modified from xitorch/xitorch.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the MIT License by xitorch/xitorch.
# Modifications made by Grimme Group.
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
from typing import Tuple

import torch

__all__ = ["normalize_bcast_dims", "get_bcasted_dims", "match_dim"]


def normalize_bcast_dims(*shapes):
    """
    Normalize the lengths of the input shapes to have the same length.
    The shapes are padded at the front by 1 to make the lengths equal.
    """
    maxlens = max([len(shape) for shape in shapes])
    res = [[1] * (maxlens - len(shape)) + list(shape) for shape in shapes]
    return res


def get_bcasted_dims(*shapes):
    """
    Return the broadcasted shape of the given shapes.
    """
    shapes = normalize_bcast_dims(*shapes)
    return [max(*a) for a in zip(*shapes)]


def match_dim(*xs: torch.Tensor, contiguous: bool = False) -> Tuple[torch.Tensor, ...]:
    # match the N-1 dimensions of x and xq for searchsorted and gather with dim=-1
    orig_shapes = tuple(x.shape[:-1] for x in xs)
    shape = tuple(get_bcasted_dims(*orig_shapes))
    xs_new = tuple(x.expand(shape + (x.shape[-1],)) for x in xs)
    if contiguous:
        xs_new = tuple(x.contiguous() for x in xs_new)
    return xs_new
