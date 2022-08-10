from __future__ import annotations

from .d3 import DispersionD3
from .d4 import DispersionD4
from .type import Dispersion
from ..param import Param
from ..typing import Tensor


def new_dispersion(numbers: Tensor, positions: Tensor, par: Param) -> Dispersion:
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
    Dispersion
        Instance of the Dispersion class.

    Raises
    ------
    ValueError
        If parametrization does not contain a halogen bond correction.
    """

    if par.dispersion is None:
        raise ValueError("No dispersion schemes provided.")

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

    raise ValueError("No parameters for D3 or D4 found. Or for both (please decide).")
