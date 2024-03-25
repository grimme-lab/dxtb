"""
SCF: Guess
==========

Models for the initial charge guess for the SCF.
"""

from __future__ import annotations

import torch

from .._types import Tensor
from ..basis import IndexHelper
from ..constants import labels


def get_guess(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    ihelp: IndexHelper,
    name: int | str = labels.GUESS_EEQ,
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
        Cartesian coordinates of all atoms in the system (nat, 3).
    chrg : Tensor
        Total charge of system.
    ihelp : IndexHelper
        Helper class for indexing.
    name : str | int, optional
        Name of guess method, by default EEQ (`labels.GUESS_EEQ`).

    Returns
    -------
    Tensor
        Orbital-resolved charges.

    Raises
    ------
    ValueError
        Name of guess method is unknown.
    """
    if isinstance(name, str):
        if name.casefold() in labels.GUESS_EEQ_STRS:
            name = labels.GUESS_EEQ
        elif name.casefold() in labels.GUESS_SAD_STRS:
            name = labels.GUESS_SAD
        else:
            raise ValueError(f"Unknown guess method '{name}'.")

    if name == labels.GUESS_EEQ:
        charges = get_eeq_guess(numbers, positions, chrg)
    elif name == labels.GUESS_SAD:
        charges = torch.zeros_like(
            positions[..., -1], requires_grad=positions.requires_grad
        )
    else:
        raise ValueError(f"Unknown guess method '{name}'.")

    return spread_charges_atomic_to_orbital(charges, ihelp)


def get_eeq_guess(
    numbers: Tensor, positions: Tensor, chrg: Tensor, cutoff: Tensor | None = None
) -> Tensor:
    """
    Calculate atomic EEQ charges.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    chrg : Tensor
        Total charge of system.
    cutoff : Tensor, optional
        Cutoff radius for the EEQ model. Defaults to `None`.

    Returns
    -------
    Tensor
        Atomic charges.
    """
    # pylint: disable=import-outside-toplevel
    from tad_multicharge import get_eeq_charges

    return get_eeq_charges(numbers, positions, chrg, cutoff=cutoff)


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
    eps = torch.tensor(
        torch.finfo(charges.dtype).eps,
        device=charges.device,
        dtype=charges.dtype,
    )
    zero = torch.tensor(
        0.0,
        device=charges.device,
        dtype=charges.dtype,
    )

    shells_per_atom = ihelp.spread_atom_to_shell(ihelp.shells_per_atom)
    orbs_per_shell = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)

    shell_charges = torch.where(
        shells_per_atom > 0,
        ihelp.spread_atom_to_shell(charges)
        / torch.clamp(shells_per_atom.type(charges.dtype), min=eps),
        zero,
    )

    orb_charges = torch.where(
        orbs_per_shell > 0,
        ihelp.spread_shell_to_orbital(shell_charges)
        / torch.clamp(orbs_per_shell.type(charges.dtype), min=eps),
        zero,
    )

    tot_chrg_old = charges.sum(-1)
    tot_chrg_new = orb_charges.sum(-1)
    if torch.any(torch.abs(tot_chrg_new - tot_chrg_old) > torch.sqrt(eps)):
        raise RuntimeError(
            "Total charge changed during spreading from atomic to orbital "
            f"charges ({tot_chrg_old:.6f} -> {tot_chrg_new:.6f})."
        )

    return orb_charges
