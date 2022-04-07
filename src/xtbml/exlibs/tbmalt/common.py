# -*- coding: utf-8 -*-
# NOTE: THANKS TO THE COURTESY OF https://github.com/tbmalt/tbmalt
"""A collection of useful code abstractions.

All modules that are not specifically associated with any one component of the
code, such as generally mathematical operations, are located here.
"""
from typing import Tuple, Union, List
import torch
from torch import Tensor

# Types
float_like = Union[Tensor, float]
bool_like = Union[Tensor, bool]


def split_by_size(
    tensor: Tensor, sizes: Union[Tensor, List[int]], dim: int = 0
) -> Tuple[Tensor]:
    """Splits a tensor into chunks of specified length.

    This function takes a tensor & splits it into `n` chunks, where `n` is the
    number of entries in ``sizes``. The length of the `i'th` chunk is defined
    by the `i'th` element of ``sizes``.

    Arguments:
        tensor: Tensor to be split.
        sizes: Size of each chunk.
        dim: Dimension along which to split ``tensor``.

    Returns:
        chunked: Tuple of tensors viewing the original ``tensor``.

    Examples:
        Tensors can be sequentially split into multiple sub-tensors like so:

        >>> from tbmalt.common import split_by_size
        >>> a = torch.arange(10)
        >>> print(split_by_size(a, [2, 2, 2, 2, 2]))
        (tensor([0, 1]), tensor([2, 3]), tensor([4, 5]), tensor([6, 7]), tensor([8, 9]))
        >>> print(split_by_size(a, [5, 5]))
        tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9]))
        >>> print(split_by_size(a, [1, 2, 3, 4]))
        (tensor([0]), tensor([1, 2]), tensor([3, 4, 5]), tensor([6, 7, 8, 9]))

    Notes:
        The resulting tensors ``chunked`` are views of the original tensor and
        not copies. This was created as no analog existed natively within the
        pytorch framework. However, this will eventually be removed once the
        pytorch function `split_with_sizes` becomes operational.

    Raises:
        AssertionError: If number of elements requested via ``split_sizes``
            exceeds the number of elements present in ``tensor``.
    """
    # Looks like it returns a tuple rather than a list
    if dim < 0:  # Shift dim to be compatible with torch.narrow
        dim += tensor.dim()

    # Ensure the tensor is large enough to satisfy the chunk declaration.
    size_match = tensor.shape[dim] == sum(sizes)
    assert size_match, (
        "Sum of split sizes fails to match tensor length " "along specified dim"
    )

    # Identify the slice positions
    splits = torch.cumsum(torch.tensor([0, *sizes]), dim=0)[:-1]

    # Return the sliced tensor. use torch.narrow to avoid data duplication
    return tuple(
        torch.narrow(tensor, dim, start, length) for start, length in zip(splits, sizes)
    )
