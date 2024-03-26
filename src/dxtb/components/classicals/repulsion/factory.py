"""
Repulsion: Factory
==================

A factory function to create instances of the Repulsion class.
"""

from __future__ import annotations

import warnings

import torch
from tad_mctc.typing import DD, Tensor, get_default_dtype

from dxtb.constants import xtb
from dxtb.exceptions import ParameterWarning
from dxtb.param import Param, get_elem_param

from .rep import Repulsion, RepulsionAnalytical

__all__ = ["new_repulsion"]


def new_repulsion(
    numbers: Tensor,
    par: Param,
    cutoff: float = xtb.DEFAULT_REPULSION_CUTOFF,
    with_analytical_gradient: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Repulsion | None:
    """
    Create new instance of Repulsion class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    par : Param
        Representation of an extended tight-binding model.
    cutoff : float
        Real space cutoff for repulsion interactions (default: 25.0).
    with_analytical_gradient : bool, optional
        Whether to instantiate a repulsion class that implements a custom
        backward function with an analytical nuclear gradient, i.e., the first
        derivative w.r.t. positions is computed with an analytical formula
        instead of the AD engine. Defaults to `False`.

    Returns
    -------
    Repulsion | None
        Instance of the Repulsion class or `None` if no repulsion is used.

    Raises
    ------
    ValueError
        If parametrization does not contain a halogen bond correction.
    """

    if hasattr(par, "repulsion") is False or par.repulsion is None:
        # TODO: Repulsion is used in all models, so error or just warning?
        warnings.warn("No repulsion scheme found.", ParameterWarning)
        return None

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    kexp = torch.tensor(par.repulsion.effective.kexp, **dd)
    klight = (
        torch.tensor(par.repulsion.effective.klight, **dd)
        if par.repulsion.effective.klight
        else None
    )

    # get parameters for unique species
    unique = torch.unique(numbers)
    arep = get_elem_param(unique, par.element, "arep", pad_val=0, **dd)
    zeff = get_elem_param(unique, par.element, "zeff", pad_val=0, **dd)

    if with_analytical_gradient is True:
        return RepulsionAnalytical(arep, zeff, kexp, klight, cutoff, **dd)
    return Repulsion(arep, zeff, kexp, klight, cutoff, **dd)
