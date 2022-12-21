# NOTE: THANKS TO THE COURTESY OF https://github.com/tbmalt/tbmalt
"""Helper functions for batch operations.

This module contains classes and helper functions associated with batch
construction, handling and maintenance.
"""
from collections import namedtuple
from functools import partial, reduce

import torch

from . import wrap_gather
from .._types import Any, Literal, Tensor, overload

__sort = namedtuple("sort", ("values", "indices"))
Sliceable = list[Tensor] | tuple[Tensor, Tensor]
bool_like = Tensor | bool


@overload
def pack(
    tensors: Sliceable,
    axis: int = 0,
    value: Any = 0,
    size: tuple[int] | torch.Size | None = None,
    return_mask: Literal[False] = False,
) -> Tensor:
    ...


@overload
def pack(
    tensors: Sliceable,
    axis: int = 0,
    value: Any = 0,
    size: tuple[int] | torch.Size | None = None,
    return_mask: Literal[True] = True,
) -> tuple[Tensor, Tensor]:
    ...


def pack(
    tensors: Sliceable,
    axis: int = 0,
    value: Any = 0,
    size: tuple[int] | torch.Size | None = None,
    return_mask: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Pad and pack a sequence of tensors together.

    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Arguments:
        tensors: list of tensors to be packed, all with identical dtypes.
        axis: Axis along which tensors should be packed; 0 for first axis -1
            for the last axis, etc. This will be a new dimension. [DEFAULT=0]
        value: The value with which the tensor is to be padded. [DEFAULT=0]
        size: Size of each dimension to which tensors should be padded. This
            to the largest size encountered along each dimension.
        return_mask: If True, a mask identifying the padding values is
            returned. [DEFAULT=False]

    Returns:
        packed_tensors: Input tensors padded and packed into a single tensor.
        mask: A tensor that can mask out the padding values. A False value in
            ``mask`` indicates the corresponding entry in ``packed_tensor`` is
            a padding value.

    Notes:
        ``packed_tensors`` maintains the same order as ``tensors``. This
        is faster & more flexible than the internal pytorch pack & pad
        functions (at this particularly task).

        If a ``tensors`` is a `torch.tensor` it will be immedatly returned.
        This helps with batch agnostic programming.

    Examples:
        Multiple tensors can be packed into a single tensor like so:

        >>> from tbmalt.common.batch import pack
        >>> import torch
        >>> a, b, c = torch.rand(2,2), torch.rand(3,3), torch.rand(4,4)
        >>> abc_packed_a = pack([a, b, c])
        >>> print(abc_packed_a.shape)
        torch.Size([3, 4, 4])
        >>> abc_packed_b = pack([a, b, c], axis=1)
        >>> print(abc_packed_b.shape)
        torch.Size([4, 3, 4])
        >>> abc_packed_c = pack([a, b, c], axis=-1)
        >>> print(abc_packed_c.shape)
        torch.Size([4, 4, 3])

        An optional mask identifying the padding values can also be returned:

        >>> packed, mask = pack([torch.tensor([1.]),
        >>>                      torch.tensor([2., 2.]),
        >>>                      torch.tensor([3., 3., 3.])],
        >>>                     return_mask=True)
        >>> print(packed)
        tensor([[1., 0., 0.],
                [2., 2., 0.],
                [3., 3., 3.]])
        >>> print(mask)
        tensor([[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]])

    """
    # If "tensors" is already a Tensor then return it immediately as there is
    # nothing more that can be done. This helps with batch agnostic
    # programming.
    if isinstance(tensors, Tensor):
        return tensors

    # Gather some general setup info
    count, device, dtype = len(tensors), tensors[0].device, tensors[0].dtype

    # Identify the maximum size, if one was not specified.
    if size is None:
        size = torch.tensor([i.shape for i in tensors]).max(0).values

    # Tensor to pack into, filled with padding value.
    padded = torch.full((count, *size), value, dtype=dtype, device=device)

    if return_mask:  # Generate the mask if requested.
        mask = torch.full((count, *size), False, dtype=torch.bool, device=device)

    # Loop over & pack "tensors" into "padded". A proxy index "n" must be used
    # for assignments rather than a slice to prevent in-place errors.
    for n, source in enumerate(tensors):
        # Slice operations not elegant but they are dimension agnostic & fast.
        padded[(n, *[slice(0, s) for s in source.shape])] = source
        if return_mask:  # Update the mask if required.
            mask[(n, *[slice(0, s) for s in source.shape])] = True

    # If "axis" was anything other than 0, then "padded" must be permuted.
    if axis != 0:
        # Resolve relative any axes to their absolute equivalents to maintain
        # expected slicing behaviour when using the insert function.
        axis = padded.dim() + 1 + axis if axis < 0 else axis

        # Build a list of axes indices; but omit the axis on which the data
        # was concatenated (i.e. 0).
        ax = list(range(1, padded.dim()))

        ax.insert(axis, 0)  # Re-insert the concatenation axis as specified

        padded = padded.permute(ax)  # Perform the permeation

        if return_mask:  # Perform permeation on the mask is present.
            mask = mask.permute(ax)

    # Return the packed tensor, and the mask if requested.
    return (padded, mask) if return_mask else padded


def pargsort(tensor: Tensor, mask: bool_like | None = None, dim: int = -1) -> Tensor:
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


def psort(tensor: Tensor, mask: bool_like | None = None, dim: int = -1) -> __sort:
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


def merge(tensors: Sliceable, value: Any = 0, axis: int = 0) -> Tensor:
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


def deflate(tensor: Tensor, value: Any = 0, axis: int | None = None) -> Tensor:
    """Shrinks ``tensor`` to remove extraneous, trailing padding values.

    Returns a narrowed view of ``tensor`` containing no superfluous trailing
    padding values. For single systems this is equivalent to removing padding.

    All axes are deflated by default, however ``axis`` can be used to forbid
    the deflation of a specific axis. This permits excess padding to be safely
    excised from a batch without inadvertently removing a system from it. This
    is normally the value supplied to the `pack` method for ``axis``.

    Arguments:
        tensor: Tensor to be deflated.
        value: Identity of padding value. [DEFAULT=0]
        axis: Specifies which, if any, an axis exempt from deflation.
            [DEFAULT=None]

    Returns:
        deflated: ``tensor`` after deflation.

    Note:
        Only trailing padding values will be culled; i.e. columns will only be
        removed from the end of a matrix, not the start or the middle.

        Deflation cannot be performed on one dimensional systems when ``axis``
        is not `None`.

    Examples:
        `deflate` can be used to remove unessiary padding from a batch:

        >>> from tbmalt.common.batch import deflate
        >>> over_packed = torch.tensor([
        >>>     [0, 1, 2, 0, 0, 0],
        >>>     [3, 4, 5, 6, 0, 0],
        >>> ])

        >>> print(deflate(over_packed, value=0, axis=0))
        tensor([[0, 1, 2, 0],
                [3, 4, 5, 6]])

        or to remove padding from a system which was once part of a batch:

        >>> packed = torch.tensor([
        >>>     [0, 1, 0, 0],
        >>>     [3, 4, 0, 0],
        >>>     [0, 0, 0, 0],
        >>>     [0, 0, 0, 0]])

        >>> print(deflate(packed, value=0))
        tensor([[0, 1],
                [3, 4]])

    Warnings:
        Under certain circumstances "real" elements may be misidentified as
        padding values if they are equivalent. However, such complication can
        be mitigated though the selection of an appropriate padding value.

    Raises:
         ValueError: If ``tensor`` is 0 dimensional, or 1 dimensional when
            ``axis`` is not None.
    """

    # Check shape is viable.
    if axis is not None and tensor.ndim <= 1:
        raise ValueError("Tensor must be at least 2D when specifying an ``axis``.")

    mask = tensor == value
    if axis is not None:
        mask = mask.all(axis)

    slices = []
    if (ndim := mask.ndim) > 1:  # When multidimensional `all` is required
        for dim in reversed(torch.combinations(torch.arange(ndim), ndim - 1)):
            # Count Nº of trailing padding values. Reduce/partial used here as
            # torch.all cannot operate on multiple dimensions like numpy.
            v, c = (
                reduce(partial(torch.all, keepdims=True), dim, mask)
                .squeeze()
                .unique_consecutive(return_counts=True)
            )

            # Slicer will be None if there are no trailing padding values.
            slices.append(slice(None, -c[-1] if v[-1] else None))

    else:  # If mask is one dimensional, then no loop is needed
        v, c = mask.unique_consecutive(return_counts=True)
        slices.append(slice(None, -c[-1] if v[-1] else None))

    if axis is not None:
        slices.insert(axis, ...)  # <- dummy index for batch-axis

    return tensor[slices]


def unpack(tensor: Tensor, value: Any = 0, axis: int = 0) -> tuple[Tensor]:
    """Unpacks packed tensors into their constituents and removes padding.

    This acts as the inverse of the `pack` operation.

    Arguments:
        tensor: Tensor to be unpacked.
        value: Identity of padding value. [DEFAULT=0]
        axis: Axis along which ``tensor`` was packed. [DEFAULT=0]

    Returns:
        tensors: tuple of constituent tensors.
    """
    return tuple(deflate(i, value) for i in tensor.movedim(axis, 0))


# by MF
def index(inp: Tensor, idx: Tensor) -> Tensor:
    """Batched indexing using `torch.gather`.

    Parameters
    ----------
    inp : Tensor
        Input tensor.
    idx : Tensor
        Index tensor.

    Returns
    -------
    Tensor
        Output tensor.

    Examples
    --------
    Batched indexing with same dimensions of `idx` and `inp` (n_batch, x).
    >>> from xtbml.exlibs.tbmalt import batch
    >>> inp = torch.tensor([
    ...     [ 0.4800, 0.4701, 0.3405, 0.4701 ],
    ...     [ 0.4701, 0.5833, 0.7882, 0.3542 ]
    ... ])
    >>> idx = torch.tensor([
    ...     [ 0,  0,  1,  1,  2,  2,  3,  3 ],
    ...     [ 0,  1,  1,  1,  2,  2,  3,  3 ]
    ... ])
    >>> print(batch.index(inp, idx))
    tensor([[0.4800, 0.4800, 0.4701, 0.4701, 0.3405, 0.3405, 0.4701, 0.4701],
            [0.4701, 0.5833, 0.5833, 0.5833, 0.7882, 0.7882, 0.3542, 0.3542]])

    Also works for non-batched versions.
    >>> from xtbml.exlibs.tbmalt import batch
    >>> inp = torch.tensor([ 0.4800, 0.4701, 0.3405, 0.4701 ])
    >>> idx = torch.tensor([ 0,  0,  1,  1,  2,  2,  3,  3 ])
    >>> print(batch.index(inp, idx))
    tensor([0.4800, 0.4800, 0.4701, 0.4701, 0.3405, 0.3405, 0.4701, 0.4701])


    Batched indexing with `idx` having one more dimension than `inp`.
    >>> from xtbml.exlibs.tbmalt import batch
    >>> inp = torch.tensor([
    ...     [
    ...         [-3.7510, -5.8131, -1.2251],
    ...         [-1.4523, -3.0188,  2.3872],
    ...         [-1.9942, -3.5295, -1.3030],
    ...         [-4.3375, -6.6594,  0.5598],
    ...     ],
    ...     [
    ...         [ 3.3579,  2.5251, -3.4608],
    ...         [ 2.7920,  1.0176, -2.5924],
    ...         [ 3.0536,  7.1525,  1.8216],
    ...         [ 1.2930,  0.7893,  0.9190]
    ...     ],
    ... ])
    >>> idx = torch.tensor([
    ...     [ 0,  0,  1,  1,  2,  2,  3,  3 ],
    ...     [ 0,  1,  1,  1,  2,  2,  3,  3 ]
    ... ])
    >>> print(batch.index(inp, idx))
    tensor([[[-3.7510, -5.8131, -1.2251],
             [-3.7510, -5.8131, -1.2251],
             [-1.4523, -3.0188,  2.3872],
             [-1.4523, -3.0188,  2.3872],
             [-1.9942, -3.5295, -1.3030],
             [-1.9942, -3.5295, -1.3030],
             [-4.3375, -6.6594,  0.5598],
             [-4.3375, -6.6594,  0.5598]],

            [[ 3.3579,  2.5251, -3.4608],
             [ 2.7920,  1.0176, -2.5924],
             [ 2.7920,  1.0176, -2.5924],
             [ 2.7920,  1.0176, -2.5924],
             [ 3.0536,  7.1525,  1.8216],
             [ 3.0536,  7.1525,  1.8216],
             [ 1.2930,  0.7893,  0.9190],
             [ 1.2930,  0.7893,  0.9190]]])

    Also works for non-batched version.
    """

    if len(inp.shape) == len(idx.shape):
        return torch.gather(inp, -1, idx)

    if len(inp.shape) == (len(idx.shape) + 1):
        # also support non-batched by unpacking idx
        size = [*idx.size(), inp.size(-1)]

        dummy = idx.unsqueeze(-1).expand(*size)

        return wrap_gather(
            inp,
            -2,
            torch.where(dummy >= 0, dummy, dummy.new_tensor(0)),
        )

    if len(inp.shape) == (len(idx.shape) - 1):
        dummy = inp.unsqueeze(-2).expand(idx.size(0), -1)
        return torch.where(
            idx >= 0,
            torch.gather(dummy, -1, torch.where(idx >= 0, idx, 0)),
            torch.tensor(-999.0, device=inp.device, dtype=inp.dtype),
        )

    raise NotImplementedError(
        f"Indexing with input size '{len(inp.shape)}' and index size "
        f"'{len(idx.shape)}' not implemented."
    )
