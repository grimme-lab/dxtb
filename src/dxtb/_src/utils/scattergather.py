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
Utility: Scatter and Gather
===========================

Wrappers and convenience functions for `torch.scatter_reduce` and
`torch.gather` that allow for negative indices, multiple dimensions, and
batched calculations.
"""

from __future__ import annotations

import warnings
from functools import wraps

import torch

from dxtb.__version__ import __tversion__
from dxtb._src.typing import Gather, Protocol, ScatterOrGather, Tensor

from .tensors import t2int

__all__ = ["scatter_reduce", "wrap_scatter_reduce", "wrap_gather"]


class _ScatterOrGatherWrapper(Protocol):
    """
    Type annotation for wrapper function of scatter or gather.
    """

    def __call__(
        self,
        func: ScatterOrGather,
        x: Tensor,
        dim0: int,
        dim1: int,
        idx: Tensor,
        *args: str,
    ) -> Tensor: ...


def twice_remove_negative_index(
    fcn: _ScatterOrGatherWrapper,
) -> _ScatterOrGatherWrapper:
    """Wrapper for `gather_twice` function that removes negative indices."""

    @wraps(fcn)
    def wrapper(
        func: ScatterOrGather,
        x: Tensor,
        dim0: int,
        dim1: int,
        idx: Tensor,
        *args: str,
    ) -> Tensor:
        mask = idx >= 0

        if torch.all(mask):
            return fcn(func, x, dim0, dim1, idx, *args)

        # gathering in two dimensions requires expanding the mask
        return torch.where(
            mask.unsqueeze(-1) * mask.unsqueeze(-2),
            fcn(func, x, dim0, dim1, torch.where(mask, idx, 0), *args),
            torch.tensor(0.0, device=x.device, dtype=x.dtype),
        )

    return wrapper


@twice_remove_negative_index
def twice(
    func: ScatterOrGather,
    x: Tensor,
    dim0: int,
    dim1: int,
    idx: Tensor,
    *args: str,
) -> Tensor:
    """
    Spread or gather a tensor along two dimensions

    Parameters
    ----------
    f: Callable
        Function to apply (`torch.gather` or `torch.scatter_reduce`)
    x : Tensor
        Tensor to spread/gather
    index : Tensor
        Index to spread/gather along
    dim0 : int
        Dimension to spread/gather along
    dim1 : int
        Dimension to spread/gather along

    Returns
    -------
    Tensor
        Spread/Gathered tensor
    """

    shape0 = [-1] * x.dim()
    shape0[dim0] = x.shape[dim0]
    y = func(
        x,
        dim1,
        idx.unsqueeze(dim0).expand(*shape0),
        *args,
    )

    shape1 = [-1] * y.dim()
    shape1[dim1] = y.shape[dim1]
    z = func(
        y,
        dim0,
        idx.unsqueeze(dim1).expand(*shape1),
        *args,
    )
    return z


def adapt_indexer_extra(
    x: Tensor,
    dim: int | tuple[int, int],
    idx: Tensor,
) -> Tensor:
    """
    Modify the indexing tensor for the extra dimension.

    Parameters
    ----------
    x : Tensor
        Tensor to gather.
    dim : int | tuple[int, int]
        Dimension to gather over.
    idx : Tensor
        Index to gather over.

    Returns
    -------
    Tensor
        Modified indexer for gathering or spreading.

    Raises
    ------
    TypeError
        Tuple instead of integer is passed. Tuples cannot be handled.
    ValueError
        Dimension index is outside the allowed range.
    RuntimeError
        Indexing tensor has two many dimensions (max. 2 possible).
    RuntimeError
        Source tensor is two small (min. 2 dimensions).
    NotImplementedError
        Source tensor is too large (max. 4 dimension possible).
    """
    # Accounting for the extra dimension is very hacky and anything but general.
    # Therefore, I added a ton of checks to prevent the user from accidentally
    # doing something unintended, which may otherwise not be caught.
    if not isinstance(dim, int):
        raise TypeError(
            "If an extra dimension is specified, only one dimension is "
            f"allowed. You passed '{dim}'."
        )

    if dim > -2:
        raise ValueError(
            "Only negative values are allowed for indexing. Additionally, "
            "The last dimension is reserved for the extra dimension, i.e., "
            "using -1 is also prohibited."
        )

    # this should not be possible to reach
    if idx.ndim > 2:  # pragma: no cover
        raise RuntimeError(
            "The indexing tensor must be 1d or 2d for batched calculations."
        )

    if x.ndim < 2:
        raise RuntimeError(
            "The source tensor must at least have two dimension: The "
            "dimension that is reduced/gathered and the extra dimension."
        )

    if x.ndim > 4:
        raise NotImplementedError(
            "Indexing of source tensors with more than 4 dimensions is "
            "currently not implemented."
        )

    shp = [*x.shape[:dim], -1, *x.shape[(dim + 1) :]]

    # here, we assume that two dimension in `idx` mean batched mode
    if idx.ndim < 2:
        if x.ndim == 2:
            idx = idx.unsqueeze(-1).expand(*shp)
        elif x.ndim == 3:
            # Unsqueeze the indices so that there initial shape is in the
            # position specified by `dim`. Relies on negative indices and
            # exclusion of -1. So pretty much works only for -2 and -3.
            idx = idx.unsqueeze(dim + 2).unsqueeze(-1).expand(*shp)
    else:
        if x.ndim == 3:
            idx = idx.unsqueeze(-1).expand(*shp)
        elif x.ndim == 4:
            # again, only -2 and -3 are supported
            if dim == -2:
                idx = idx.unsqueeze(1).unsqueeze(-1).expand(*shp)
            elif dim == -3:
                idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*shp)

    return idx


# gather


def gather_remove_negative_index(func: Gather) -> Gather:
    """
    Wrapper for `gather` function that removes negative indices.

    Parameters
    ----------
    func : Gather
        `torch.gather`.

    Returns
    -------
    Gather
        Wrapped `torch.gather` (for use as decorator).
    """

    @wraps(func)
    def wrapper(x: Tensor, dim: int, idx: Tensor, *args: str) -> Tensor:
        mask = idx >= 0
        if torch.all(mask):
            return func(x, dim, idx, *args)

        return torch.where(
            mask,
            func(x, dim, torch.where(mask, idx, 0), *args),
            torch.tensor(0, device=x.device, dtype=x.dtype),
        )

    return wrapper


@gather_remove_negative_index
def gather(x: Tensor, dim: int, idx: Tensor) -> Tensor:
    """
    Wrapper for `torch.gather`.

    Parameters
    ----------
    x : Tensor
        Tensor to gather
    dim : int
        Dimension to gather over
    idx : Tensor
        Index to gather over

    Returns
    -------
    Tensor
        Gathered tensor
    """
    return torch.gather(x, dim, idx)


def wrap_gather(
    x: Tensor,
    dim: int | tuple[int, int],
    idx: Tensor,
    extra: bool = False,
) -> Tensor:
    """
    Wrapper for gather function. Also handles multiple dimensions.

    Parameters
    ----------
    x : Tensor
        Tensor to gather.
    dim : int | tuple[int, int]
        Dimension to gather over.
    idx : Tensor
        Index to gather over.
    extra : bool
        If the tensor to reduce contains a extra dimension of arbitrary size
        that is generally different from the size of the indexing tensor
        (e.g. gradient tensors with extra xyz dimension), the indexing tensor
        has to be modified. This feature is only tested for the aforementioned
        gradient tensors and does only work for one dimension.
        Defaults to ``False``.

    Returns
    -------
    Tensor
        Gathered tensor.
    """
    if extra is True:
        idx = adapt_indexer_extra(x, dim, idx)

    if idx.ndim > 1:
        if isinstance(dim, int):
            if x.ndim < idx.ndim:
                x = x.unsqueeze(0).expand(idx.size(0), -1)
        else:
            if x.ndim <= idx.ndim:
                x = x.unsqueeze(0).expand(idx.size(0), -1, -1)

    return (
        gather(x, dim, idx)
        if isinstance(dim, int)
        else twice(torch.gather, x, *dim, idx)
    )


# scatter


def scatter_reduce(
    x: Tensor, dim: int, idx: Tensor, *args: str, fill_value: float | int | None = 0
) -> Tensor:  # pragma: no cover
    """

    .. warning::

        `scatter_reduce` is only introduced in 1.11.1 and the API changes in
        v12.1 in a BC-breaking way. `scatter_reduce` in 1.12.1 and 1.13.0 is
        still in beta and CPU-only.

        Related links:

        - https://pytorch.org/docs/1.12/generated/torch.Tensor.scatter_reduce_.\
          html#torch.Tensor.scatter_reduce_
        - https://pytorch.org/docs/1.11/generated/torch.scatter_reduce.html
        - https://github.com/pytorch/pytorch/releases/tag/v1.12.0
          (section "Sparse")

    Thin wrapper for pytorch's `scatter_reduce` function for handling API
    changes.

    Parameters
    ----------
    x : Tensor
        Tensor to reduce.
    dim : int
        Dimension to reduce over.
    idx : Tensor
        Index to reduce over.
    fill_value : float | int | None
        Value with which the output is inititally filled (reduction units for
        indices not scattered to). Defaults to `0`.

    Returns
    -------
    Tensor
        Reduced tensor.
    """

    if (1, 11, 0) <= __tversion__ < (1, 12, 0):  # type: ignore
        actual_device = x.device

        # account for CPU-only implementation
        if "cuda" in str(actual_device):
            x = x.to(torch.device("cpu"))
            idx = idx.to(torch.device("cpu"))

        output = torch.scatter_reduce(x, dim, idx, *args)  # type: ignore
        output = output.to(actual_device)
    elif __tversion__ >= (1, 12, 0) or __tversion__ >= (2, 0, 0):  # type: ignore
        out_shape = list(x.shape)
        out_shape[dim] = t2int(idx.max()) + 1

        # filling the output is only necessary if the user wants to preserve
        # the behavior in 1.11, where indices not scattered to are filled with
        # reduction inits (sum: 0, prod: 1)
        if fill_value is None:
            out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        else:
            out = torch.full(out_shape, fill_value, device=x.device, dtype=x.dtype)

        # stop warning about beta and possible API changes in 1.12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = torch.scatter_reduce(out, dim, idx, x, *args)  # type: ignore
    else:
        raise RuntimeError(f"Unsupported PyTorch version ({__tversion__}) used.")

    return output


def wrap_scatter_reduce(
    x: Tensor,
    dim: int | tuple[int, int],
    idx: Tensor,
    reduce: str,
    extra: bool = False,
) -> Tensor:
    """
    Wrapper for `torch.scatter_reduce` that removes negative indices.

    Parameters
    ----------
    x : Tensor
        Tensor to reduce.
    dim : int | (int, int)
        Dimension to reduce over, defaults to -1.
    idx : Tensor
        Index to reduce over.
    reduce : str
        Reduction method, defaults to "sum".
    extra : bool
        If the tensor to reduce contains a extra dimension of arbitrary size
        that is generally different from the size of the indexing tensor
        (e.g. gradient tensors with extra xyz dimension), the indexing tensor
        has to be modified. This feature is only tested for the aforementioned
        gradient tensors and does only work for one dimension.
        Defaults to ``False``.

    Returns
    -------
    Tensor
        Reduced tensor.
    """
    if extra is True:
        idx = adapt_indexer_extra(x, dim, idx)

    idx = torch.where(idx >= 0, idx, 0)
    return (
        scatter_reduce(x, dim, idx, reduce)
        if isinstance(dim, int)
        else twice(scatter_reduce, x, *dim, idx, reduce)
    )
