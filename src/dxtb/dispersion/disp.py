"""
Function for creating a new instance of a Dispersion.
"""
from __future__ import annotations

import warnings

import torch

from .._types import NoReturn, Tensor
from ..param import Param
from ..utils import ParameterWarning
from .base import Dispersion
from .d3 import DispersionD3
from .d4 import DispersionD4


def new_dispersion(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Dispersion | None | NoReturn:
    """
    Create new instance of the Dispersion class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Dispersion | None
        Instance of the Dispersion class or `None` if no dispersion is used.

    Raises
    ------
    ValueError
        If parametrization does not contain a halogen bond correction.
    """

    if hasattr(par, "dispersion") is False or par.dispersion is None:
        warnings.warn("No dispersion scheme found.", ParameterWarning)
        return None

    if par.dispersion.d3 is not None and par.dispersion.d4 is None:
        param = _convert_float_tensor(
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
        param = _convert_float_tensor(
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
        return DispersionD4(numbers, param, device=device, dtype=dtype)

    if par.dispersion.d3 is not None and par.dispersion.d4 is not None:
        raise ValueError("Parameters for both D3 and D4 found. Please decide.")

    return None


def _convert_float_tensor(
    d: dict[str, float | Tensor],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    for key, value in d.items():
        if isinstance(value, float):
            d[key] = torch.tensor(value, device=device, dtype=dtype)

    return d  # type: ignore
