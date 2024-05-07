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
Helper functions for Overlap
============================

Helper functions related to calculation and sorting of unique shells pairs.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Literal, Tensor

__all__ = ["get_pairs", "get_subblock_start"]


def get_pairs(x: Tensor, i: int) -> Tensor:
    """
    Get indices of all unqiue shells pairs with index value `i`.

    Parameters
    ----------
    x : Tensor
        Matrix of unique shell pairs.
    i : int
        Value representing all unique shells in the matrix.

    Returns
    -------
    Tensor
        Indices of all unique shells pairs with index value `i` in the matrix.
    """

    return (x == i).nonzero(as_tuple=False)


def get_subblock_start(
    umap: Tensor, i: int, norbi: int, norbj: int, uplo: Literal["n", "u", "l"] = "l"
) -> Tensor:
    """
    Filter out the top-left index of each subblock of unique shell pairs.
    This makes use of the fact that the pairs are sorted along the rows.

    Example: A "s" and "p" orbital would give the following 4x4 matrix
    of unique shell pairs:
    1 2 2 2
    3 4 4 4
    3 4 4 4
    3 4 4 4
    As the overlap routine gives back tensors of the shape `(norbi, norbj)`,
    i.e. 1x1, 1x3, 3x1 and 3x3 here, we require only the following four
    indices from the matrix of unique shell pairs: [0, 0] (1x1), [1, 0]
    (3x1), [0, 1] (1x3) and [1, 1] (3x3).


    Parameters
    ----------
    pairs : Tensor
        Indices of all unique shell pairs of one type (n, 2).
    norbi : int
        Number of orbitals per shell.
    norbj : int
        Number of orbitals per shell.

    Returns
    -------
    Tensor
        Top-left (i.e. [0, 0]) index of each subblock.
    """

    # no need to filter out a 1x1 block
    if norbi == 1 and norbj == 1:
        return get_pairs(umap, i)

    # sorting along rows allows only selecting every `norbj`th pair
    if norbi == 1:
        pairs = get_pairs(umap, i)
        return pairs[::norbj]

    if norbj == 1:
        pairs = get_pairs(umap.mT, i)

        # do the same for the transposed pairs, but switch columns
        return torch.index_select(
            pairs[::norbi], 1, torch.tensor([1, 0], device=umap.device)
        )

    # the remaining cases, i.e., if no s-orbitals are involved, are more
    # intricate because we can have variation in two dimensions...

    # If only a triangular matrix is considered, we need to take special
    # care of the diagonal because the blocks on the diagonal are cut off,
    # which leads to missing pairs for the `while` loop. Whether this is
    # the case for the unique index `i`, is checked by the trace of the
    # unique map: The trace will be zero if there are no blocks on the
    # diagonal. If there are blocks on the diagonal, we complete the
    # missing triangular matrix. This includes all unique indices `i` of
    # the unique map, and hence, introduces some redundancy for blocks that
    # are not on the diagonal.
    if uplo != "n":
        if (
            torch.where(umap == i, umap, torch.tensor(0, device=umap.device))
        ).trace() > 0.0:
            umap = torch.where(umap == -1, umap.mT, umap)

    pairs = get_pairs(umap, i)

    # remove every `norbj`th pair as before; only second dim is tricky
    pairs = pairs[::norbj]

    start = 0
    rest = pairs

    # init with dummy
    final = torch.tensor([[-1, -1]], device=umap.device)

    while True:
        # get number of blocks in a row by counting the number of same
        # indices in the first dimension
        nb = (pairs[:, 0] == pairs[start, 0]).nonzero().flatten().size(0)

        # we need to skip the amount of rows in the block
        skip = nb * norbi

        # split for the blocks in each row because there can be different
        # numbers of blocks in a row
        target, rest = torch.split(rest, [skip, rest.size(-2) - skip])

        # select only the top left index of each block
        final = torch.cat((final, target[:nb]), 0)

        start += skip
        if start >= pairs.size(-2):
            break

    # remove dummy
    return final[1:]
