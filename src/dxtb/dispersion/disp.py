"""
Function for creating a new instance of a Dispersion.
"""

from __future__ import annotations

import warnings

import torch

from .._types import Tensor
from ..exceptions import ParameterWarning
from ..param import Param
from ..utils import convert_float_tensor
from .base import Dispersion
from .d3 import DispersionD3
from .d4 import DispersionD4


def new_dispersion(
    numbers: Tensor,
    par: Param,
    charge: Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Dispersion | None:
    """
    Create new instance of the Dispersion class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Dispersion | None
        Instance of the Dispersion class or `None` if no dispersion is used.

    Raises
    ------
    ValueError
        Parametrization does not contain a dispersion correction.
    ValueError
        D4 parametrization is requested but no charge given.
    """
    if hasattr(par, "dispersion") is False or par.dispersion is None:
        warnings.warn("No dispersion scheme found.", ParameterWarning)
        return None

    if par.dispersion.d3 is not None and par.dispersion.d4 is None:
        param = convert_float_tensor(
            {
                "a1": par.dispersion.d3.a1,
                "a2": par.dispersion.d3.a2,
                "s6": par.dispersion.d3.s6,
                "s8": par.dispersion.d3.s8,
                "s9": par.dispersion.d3.s9,
            },
            device=device,
            dtype=dtype,
        )
        return DispersionD3(numbers, param, device=device, dtype=dtype)

    if par.dispersion.d4 is not None and par.dispersion.d3 is None:
        if charge is None:
            raise ValueError("The total charge is required for DFT-D4.")
        param = convert_float_tensor(
            {
                "a1": par.dispersion.d4.a1,
                "a2": par.dispersion.d4.a2,
                "s6": par.dispersion.d4.s6,
                "s8": par.dispersion.d4.s8,
                "s9": par.dispersion.d4.s9,
                "s10": par.dispersion.d4.s10,
            },
            device=device,
            dtype=dtype,
        )
        return DispersionD4(numbers, param, charge, device=device, dtype=dtype)

    if par.dispersion.d3 is not None and par.dispersion.d4 is not None:
        raise ValueError("Parameters for both D3 and D4 found. Please decide.")

    return None
