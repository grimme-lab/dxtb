"""
Iterations
==========

Iterations of physical properties implemented as pure functions
in order to avoid RAM leak due to circular references.
"""

from __future__ import annotations

from dxtb._src.components.interactions import (
    Charges,
    InteractionList,
    Potential,
)
from dxtb._src.constants import labels
from dxtb._src.typing import Tensor
from dxtb.config import ConfigSCF

from .conversions import (
    charges_to_potential,
    density_to_charges,
    hamiltonian_to_density,
    potential_to_charges,
    potential_to_hamiltonian,
)
from .data import _Data

__all__ = [
    "iterate_charges",
    "iterate_potential",
    "iterate_fockian",
    "iter_options",
]


def iterate_charges(
    charges: Tensor, data: _Data, cfg: ConfigSCF, interactions: InteractionList
) -> Tensor:
    """
    Perform single self-consistent iteration.

    Parameters
    ----------
    charges : Tensor
        Orbital-resolved partial charges vector.
    data: _Data
        Storage for tensors which become part of autograd graph within SCF cycles.
    cfg: SCFConfig
        Configuration for SCF settings.
    interactions : InteractionList
        Collection of `Interation` objects.

    Returns
    -------
    Tensor
        New orbital-resolved partial charges vector.
    """
    q = Charges.from_tensor(charges, data.charges, batch_mode=cfg.batch_mode)

    potential = charges_to_potential(q, interactions, data)
    new_charges = potential_to_charges(potential, data, cfg)

    return new_charges.as_tensor()


def iterate_potential(
    potential: Tensor,
    data: _Data,
    cfg: ConfigSCF,
    interactions: InteractionList,
) -> Tensor:
    """
    Perform single self-consistent iteration.

    Parameters
    ----------
    potential: Tensor
        Potential vector for each orbital partial charge.
    data: _Data
        Storage for tensors which become part of autograd graph within SCF cycles.
    cfg: SCFConfig
        Configuration for SCF settings.
    interactions : InteractionList
        Collection of `Interation` objects.


    Returns
    -------
    Tensor
        New potential vector for each orbital partial charge.
    """
    pot = Potential.from_tensor(
        potential, data.potential, batch_mode=cfg.batch_mode
    )

    charges = potential_to_charges(pot, data, cfg)
    new_potential = charges_to_potential(charges, interactions, data)

    return new_potential.as_tensor()


def iterate_fockian(
    fockian: Tensor, data: _Data, cfg: ConfigSCF, interactions: InteractionList
) -> Tensor:
    """
    Perform single self-consistent iteration using the Fock matrix.

    Parameters
    ----------
    fockian : Tensor
        Fock matrix.
    data: _Data
        Storage for tensors which become part of autograd graph within SCF cycles.
    cfg: SCFConfig
        Configuration for SCF settings.
    interactions : InteractionList
        Collection of `Interation` objects.

    Returns
    -------
    Tensor
        New Fock matrix.
    """
    data.density = hamiltonian_to_density(fockian, data, cfg)
    charges = density_to_charges(data.density, data, cfg)
    potential = charges_to_potential(charges, interactions, data)
    data.hamiltonian = potential_to_hamiltonian(potential, data)

    return data.hamiltonian


iter_options = {
    labels.SCP_MODE_CHARGE: iterate_charges,
    labels.SCP_MODE_POTENTIAL: iterate_potential,
    labels.SCP_MODE_FOCK: iterate_fockian,
}
"""Possible physical values to be iterated during SCF procedure."""
