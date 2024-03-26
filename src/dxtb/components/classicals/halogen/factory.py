"""
Halogen Bond Correction: Factory
================================

A factory function to create instances of the HBC class.
"""

import torch
from tad_mctc.typing import DD, Tensor, get_default_dtype

from dxtb.constants import xtb
from dxtb.param import Param, get_elem_param

from .hal import Halogen

__all__ = ["new_halogen"]


def new_halogen(
    numbers: Tensor,
    par: Param,
    cutoff: Tensor = torch.tensor(xtb.DEFAULT_XB_CUTOFF),
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Halogen | None:
    """
    Create new instance of Halogen class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    par : Param
        Representation of an extended tight-binding model.
    cutoff : Tensor
        Real space cutoff for halogen bonding interactions (default: 20.0).

    Returns
    -------
    Halogen | None
        Instance of the Halogen class or `None` if no halogen bond correction is used.

    Raises
    ------
    ValueError
        If parametrization does not contain a halogen bond correction.
    """

    if hasattr(par, "halogen") is False or par.halogen is None:
        return None

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    damp = torch.tensor(par.halogen.classical.damping, **dd)
    rscale = torch.tensor(par.halogen.classical.rscale, **dd)

    unique = torch.unique(numbers)
    bond_strength = get_elem_param(unique, par.element, "xbond", pad_val=0, **dd)

    return Halogen(damp, rscale, bond_strength, cutoff=cutoff, **dd)