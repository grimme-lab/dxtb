"""
Collection of utility functions for testing.
"""

import torch

from dxtb.typing import Any, Tensor


def combinations(x: Tensor, r: int = 2) -> Tensor:
    """
    Generate all combinations of matrix elements.

    This is required for the comparision of overlap and Hmailtonian for
    larger systems because these matrices do not coincide with tblite.
    This is possibly due to switched indices, which were introduced in
    the initial Fortran-to-Python port.

    Parameters
    ----------
    x : Tensor
        Matrix to generate combinations from.

    Returns
    -------
    Tensor
        Matrix of combinations (n, 2).
    """
    return torch.combinations(torch.sort(x.flatten())[0], r)


def load_from_npz(npzfile: Any, name: str, dtype: torch.dtype) -> Tensor:
    """Get torch tensor from npz file

    Parameters
    ----------
    npzfile : Any
        Loaded npz file.
    name : str
        Name of the tensor in the npz file.
    dtype : torch.dtype
        Data type of the tensor.

    Returns
    -------
    Tensor
        Tensor from the npz file.
    """
    name = name.replace("-", "").replace("+", "").lower()
    return torch.from_numpy(npzfile[name]).type(dtype)


def nth_derivative(f: Tensor, x: Tensor, n: int = 1) -> Tensor:
    """
    Calculate the *n*th derivative of a tensor.

    Parameters
    ----------
    f : Tensor
        Input tensor of which the gradient should be calculated.
    x : Tensor
        Dependent variable (must have `requires_grad`)
    n : int, optional
        Order of the derivative. Defaults to 1.

    Returns
    -------
    Tensor
        The *n*th order derivative of `f` w.r.t. `x`.

    Raises
    ------
    ValueError
        Order of derivative is smaller than 1 or not an integer.
    """
    if n < 1 or not isinstance(n, int):
        raise ValueError("Order of derivative must be an integer and larger 1.")

    grads = None
    for _ in range(n):
        grads = torch.autograd.grad(f, x, create_graph=True)[0]
        f = grads.sum()

    assert grads is not None
    return grads
