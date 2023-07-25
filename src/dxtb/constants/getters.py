"""
Getter functions for constants
==============================

This module only contains some convenience functions for collecting constants.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from .chemistry import ATOMIC_MASSES
from .defaults import TORCH_DTYPE
from .units import GMOL2AU

__all__ = ["get_atomic_masses"]


def get_atomic_masses(
    numbers: Tensor,
    atomic_units: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = TORCH_DTYPE,
) -> Tensor:
    """
    Get isotope-averaged atomic masses for all `numbers`.
    Parameters
    ----------
    numbers : Tensor
        Atomic numbers in the system.
    atomic_units : bool, optional
        Flag for unit conversion. If `True` (default), the atomic masses will
        be returned in atomic units. If `False`, the unit remains g/mol.
    device : torch.device | None, optional
        Device to store the tensor. If `None` (default), the default device is used.
    dtype : torch.dtype, optional
        Data type of the tensor. If none is given, it defaults to float32.
    Returns
    -------
    Tensor
        Atomic masses.
    """
    m = ATOMIC_MASSES[numbers].to(device).type(dtype)
    return m * GMOL2AU if atomic_units is True else m
