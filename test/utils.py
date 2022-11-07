"""
Collection of utility functions for testing.
"""

import torch

from dxtb.typing import Tensor


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
