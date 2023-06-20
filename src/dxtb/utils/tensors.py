"""
Collection of utility functions for matrices/tensors.
"""
from __future__ import annotations

import torch

from .._types import Size, Tensor


@torch.jit.script
def real_atoms(numbers: Tensor) -> Tensor:
    return numbers != 0


@torch.jit.script
def real_pairs(numbers: Tensor, diagonal: bool = False) -> Tensor:
    """
    Generates mask that differentiates real atom pairs and padding.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers
    diagonal : bool, optional
        Whether the diagonal should be masked, i.e. filled with `False`.
        Defaults to `False`, i.e., `True` remains on the diagonal for real atoms.

    Returns
    -------
    Tensor
        Mask for real atom pairs.
    """
    real = real_atoms(numbers)
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)

    if diagonal is True:
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask


@torch.jit.script
def real_triples(numbers: Tensor, diagonal: bool = False) -> Tensor:
    """
    Generates mask that differentiates real atom triples and padding.
    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    diagonal : bool, optional
        Whether the diagonal should be masked, i.e. filled with `False`.
        Defaults to `False`, i.e., `True` remains on the diagonal for real atoms.
    Returns
    -------
    Tensor
        Mask for real atom triples.
    """
    real = real_pairs(numbers, diagonal=False)
    mask = real.unsqueeze(-3) * real.unsqueeze(-2) * real.unsqueeze(-1)
    if diagonal is True:
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask


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


def symmetrize(x: Tensor) -> Tensor:
    """
    Symmetrize a tensor after checking if it is symmetric within a threshold.

    Parameters
    ----------
    x : Tensor
        Tensor to check and symmetrize.

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

    if not torch.allclose(x, x.mT, atol=atol):
        raise RuntimeError(
            f"Matrix appears to be not symmetric (atol={atol:.3e}, "
            f"dtype={x.dtype})."
        )

    return (x + x.mT) / 2


def reshape_fortran(x: Tensor, shape: Size) -> Tensor:
    """
    Implements Fortran's `reshape` function (column-major).

    Parameters
    ----------
    x : Tensor
        Input tensor
    shape : Size
        Output size to which `x` is reshaped.

    Returns
    -------
    Tensor
        Reshaped tensor of size `shape`.
    """
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def euclidean_dist_quadratic_expansion(x: Tensor, y: Tensor) -> Tensor:
    """
    Computation of euclidean distance matrix via quadratic expansion (sum of
    squared differences or L2-norm of differences).

    While this is significantly faster than the "direct expansion" or
    "broadcast" approach, it only works for euclidean (p=2) distances.
    Additionally, it has issues with numerical stability (the diagonal slightly
    deviates from zero for `x=y`). The numerical stability should not pose
    problems, since we must remove zeros anyway for batched calculations.

    For more information, see `https://github.com/eth-cscs/PythonHPC/blob/master/numpy/03-euclidean-distance-matrix-numpy.ipynb`__ or
    `https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065`__.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor (with same shape as first tensor).

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    eps = torch.tensor(
        torch.finfo(x.dtype).eps,
        device=x.device,
        dtype=x.dtype,
    )

    # using einsum is slightly faster than `torch.pow(x, 2).sum(-1)`
    xnorm = torch.einsum("...ij,...ij->...i", x, x)
    ynorm = torch.einsum("...ij,...ij->...i", y, y)

    n = xnorm.unsqueeze(-1) + ynorm.unsqueeze(-2)

    # x @ y.mT
    prod = torch.einsum("...ik,...jk->...ij", x, y)

    # important: remove negative values that give NaN in backward
    return torch.sqrt(torch.clamp(n - 2.0 * prod, min=eps))


def cdist_direct_expansion(x: Tensor, y: Tensor, p: int = 2) -> Tensor:
    """
    Computation of cartesian distance matrix.

    This currently replaces the use of `torch.cdist`, which does not handle
    zeros well and produces nan's in the backward pass.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor (with same shape as first tensor).
    p : int, optional
        Power used in the distance evaluation (p-norm). Defaults to 2.

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    eps = torch.tensor(
        torch.finfo(x.dtype).eps,
        device=x.device,
        dtype=x.dtype,
    )

    # unsqueeze different dimension to create matrix
    diff = torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3))

    # einsum is nearly twice as fast!
    if p == 2:
        distances = torch.einsum("...ijk,...ijk->...ij", diff, diff)
    else:
        distances = torch.sum(torch.pow(diff, p), -1)

    return torch.pow(torch.clamp(distances, min=eps), 1.0 / p)


def cdist(x: Tensor, y: Tensor | None = None, p: int = 2) -> Tensor:
    """
    Wrapper for cartesian distance computation.

    This currently replaces the use of `torch.cdist`, which does not handle
    zeros well and produces nan's in the backward pass.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Optional[Tensor], optional
        Second tensor. If no second tensor is given (default), the first tensor
        is used as the second tensor, too.
    p : int, optional
        Power used in the distance evaluation (p-norm). Defaults to 2.

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    if y is None:
        y = x

    # faster
    if p == 2:
        return euclidean_dist_quadratic_expansion(x, y)

    return cdist_direct_expansion(x, y, p=p)
