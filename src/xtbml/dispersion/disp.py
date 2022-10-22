"""
Function for creating a new instance of a Dispersion.
"""

import warnings

from .abc import Dispersion
from .d3 import DispersionD3
from .d4 import DispersionD4
from ..exceptions import ParameterWarning
from ..param import Param
from ..typing import Tensor


def new_dispersion(numbers: Tensor, positions: Tensor, par: Param) -> Dispersion | None:
    """
    Create new instance of the Dispersion class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    positions : Tensor
        Cartesian coordinates of all atoms.
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
        # TODO: Dispersion is used in all models, so error or just warning?
        warnings.warn("No dispersion scheme found.", ParameterWarning)
        return None

    if par.dispersion.d3 is not None and par.dispersion.d4 is None:
        # FIXME: TypeError: rational_damping() got an unexpected keyword argument 's9'
        param = {
            "a1": par.dispersion.d3.a1,
            "a2": par.dispersion.d3.a2,
            "s6": par.dispersion.d3.s6,
            "s8": par.dispersion.d3.s8,
            # "s9": par.dispersion.d3.s9,
        }
        return DispersionD3(numbers, positions, param)

    if par.dispersion.d4 is not None and par.dispersion.d3 is None:
        param = {
            "a1": par.dispersion.d4.a1,
            "a2": par.dispersion.d4.a2,
            "s6": par.dispersion.d4.s6,
            "s8": par.dispersion.d4.s8,
            "s9": par.dispersion.d4.s9,
        }
        return DispersionD4(numbers, positions, param)

    if par.dispersion.d3 is not None and par.dispersion.d4 is not None:
        raise ValueError("Parameters for both D3 and D4 found. Please decide.")
