"""
Models for the initial charge guess for the SCF.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from ..basis import IndexHelper
from ..charges import ChargeModel, solve
from ..ncoord import exp_count, get_coordination_number


def get_guess(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    ihelp: IndexHelper,
    name: str = "eeq",
) -> Tensor:
    """
    Obtain initial guess for charges.
    Currently the following methods are supported:
     - electronegativity equilibration charge model ("eeq")
     - superposition of atomic densities ("sad"), i.e. zero charges


    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system.
    chrg : Tensor
        Total charge of system.
    ihelp : IndexHelper
        Helper class for indexing.
    name : str, optional
        Name of guess method, by default "eeq".

    Returns
    -------
    Tensor
        Orbital-resolved charges.

    Raises
    ------
    ValueError
        Name of guess method is unknown.
    """
    if name == "eeq":
        charges = get_eeq_guess(numbers, positions, chrg)
    elif name == "sad":
        charges = torch.zeros_like(positions[..., -1])
    else:
        raise ValueError(f"Unknown guess method '{name}'.")

    return spread_charges_atomic_to_orbital(charges, ihelp)


def get_eeq_guess(numbers: Tensor, positions: Tensor, chrg: Tensor) -> Tensor:
    """
    Calculate atomic EEQ charges.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system.
    chrg : Tensor
        Total charge of system.

    Returns
    -------
    Tensor
        Atomic charges.
    """
    eeq = ChargeModel.param2019().to(positions.device).type(positions.dtype)
    cn = get_coordination_number(numbers, positions, exp_count)
    _, qat = solve(numbers, positions, chrg, eeq, cn)

    return qat


def spread_charges_atomic_to_orbital(charges: Tensor, ihelp: IndexHelper) -> Tensor:
    """
    Spread atomic charges to orbital charges, while conserving total charge.

    Parameters
    ----------
    charges : Tensor
        Atomic charges.
    ihelp : IndexHelper
        Helper class for indexing.

    Returns
    -------
    Tensor
        Orbital-resolved charges.

    Raises
    ------
    RuntimeError
        Total charge is not conserved.
    """
    shells_per_atom = ihelp.spread_atom_to_shell(ihelp.shells_per_atom)
    orbs_per_shell = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)

    shell_charges = torch.where(
        shells_per_atom > 0,
        ihelp.spread_atom_to_shell(charges) / shells_per_atom,
        charges.new_tensor(0.0),
    )
    orb_charges = torch.where(
        orbs_per_shell > 0,
        ihelp.spread_shell_to_orbital(shell_charges) / orbs_per_shell,
        charges.new_tensor(0.0),
    )

    tot_chrg_old = charges.sum(-1)
    tot_chrg_new = orb_charges.sum(-1)
    if torch.any(
        tot_chrg_old - tot_chrg_new
        > torch.sqrt(charges.new_tensor(torch.finfo(charges.dtype).eps))
    ):
        raise RuntimeError(
            "Total charge changed during spreading from atomic to orbital "
            f"charges ({tot_chrg_old:.6f} -> {tot_chrg_new:.6f})."
        )

    return orb_charges
