from __future__ import annotations

from functools import wraps

import torch

from .._types import Callable, Gather, ScatterOrGather, Tensor
from .tensors import t2int

__all__ = ["scatter_reduce", "wrap_scatter_reduce", "wrap_gather"]


def twice_remove_negative_index(
    func: Callable[[ScatterOrGather, Tensor, int, int, Tensor], Tensor]
) -> Callable[[ScatterOrGather, Tensor, int, int, Tensor], Tensor]:
    """Wrapper for `gather_twice` function that removes negative indices."""

    @wraps(func)
    def wrapper(
        f: ScatterOrGather,
        x: Tensor,
        dim0: int,
        dim1: int,
        idx: Tensor,
        *args: str,
    ) -> Tensor:
        mask = idx >= 0

        if torch.all(mask):
            return func(f, x, dim0, dim1, idx, *args)

        # gathering in two dimensions requires expanding the mask
        return torch.where(
            mask.unsqueeze(-1) * mask.unsqueeze(-2),
            func(f, x, dim0, dim1, torch.where(mask, idx, 0), *args),
            x.new_tensor(0.0),
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


# gather


def gather_remove_negative_index(func: Gather) -> Gather:
    """Wrapper for `gather` function that removes negative indices."""

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
    """Wrapper for `torch.gather`.

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


def wrap_gather(x: Tensor, dim: int | tuple[int, int], idx: Tensor) -> Tensor:
    """Wrapper for gather function. Also handles multiple dimensions.

    Parameters
    ----------
    x : Tensor
        Tensor to gather
    dim : int | tuple[int, int]
        Dimension to gather over
    idx : Tensor
        Index to gather over

    Returns
    -------
    Tensor
        Gathered tensor
    """

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
        - https://pytorch.org/docs/1.12/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
        - https://pytorch.org/docs/1.11/generated/torch.scatter_reduce.html
        - https://github.com/pytorch/pytorch/releases/tag/v1.12.0 (section "Sparse")

    Thin wrapper for pytorch's `scatter_reduce` function for handling API changes.

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

    version = tuple(
        int(x) for x in torch.__version__.split("+", maxsplit=1)[0].split(".")
    )

    if (1, 11, 0) <= version < (1, 12, 0):
        output = torch.scatter_reduce(x, dim, idx, *args)  # type: ignore
    elif version >= (1, 12, 0) or version >= (2, 0, 0):
        out_shape = list(x.shape)
        out_shape[dim] = t2int(idx.max()) + 1

        # filling the output is only necessary if the user wants to preserve
        # the behavior in 1.11, where indices not scattered to are filled with
        # reduction inits (sum: 0, prod: 1)
        if fill_value is None:
            out = x.new_empty(out_shape)
        else:
            out = x.new_empty(out_shape).fill_(fill_value)

        output = torch.scatter_reduce(out, dim, idx, x, *args)  # type: ignore
    else:
        raise RuntimeError(f"Unsupported PyTorch version ({version}) used.")

    return output


def wrap_scatter_reduce(
    x: Tensor, dim: int | tuple[int, int], idx: Tensor, reduce: str
) -> Tensor:
    """Wrapper for `torch.scatter_reduce` that removes negative indices.

    Parameters
    ----------
    x : Tensor
        Tensor to reduce
    dim : int | (int, int)
        Dimension to reduce over, defaults to -1
    idx : Tensor
        Index to reduce over
    reduce : str
        Reduction method, defaults to "sum"

    Returns
    -------
    Tensor
        Reduced tensor
    """

    idx = torch.where(idx >= 0, idx, 0)
    return (
        scatter_reduce(x, dim, idx, reduce)
        if isinstance(dim, int)
        else twice(scatter_reduce, x, *dim, idx, reduce)
    )
