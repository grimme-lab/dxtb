"""
Iterations
==========

Iterations of physical properties implemented as pure functions
in order to avoid RAM leak due to circular references.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from ..constants import defaults
from ..interaction import InteractionList
from .config import SCFConfig
from .conversions import (
    charges_to_potential,
    density_to_charges,
    hamiltonian_to_density,
    potential_to_charges,
    potential_to_hamiltonian,
)
from .data import _Data
from .energies import get_energy


def iterate_charges(
    charges: Tensor, data: _Data, cfg: SCFConfig, interactions: InteractionList
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
    if cfg.scf_options.get("verbosity", defaults.VERBOSITY) > 0:
        if charges.ndim < 2:  # pragma: no cover
            energy = get_energy(charges, data, interactions).sum(-1).detach().clone()
            ediff = torch.linalg.vector_norm(data.old_energy - energy)

            density = data.density.detach().clone()
            pnorm = torch.linalg.matrix_norm(data.old_density - density)

            q = charges.detach().clone()
            qdiff = torch.linalg.vector_norm(data.old_charges - q)

            print(
                f"{data.iter:3}   {energy: .16E}  {ediff: .6E} "
                f"{pnorm: .6E}   {qdiff: .6E}"
            )

            data.old_energy = energy
            data.old_charges = q
            data.old_density = density
            data.iter += 1
        else:
            energy = get_energy(charges, data, interactions).detach().clone()
            ediff = torch.linalg.norm(data.old_energy - energy)

            density = data.density.detach().clone()
            pnorm = torch.linalg.norm(data.old_density - density)

            q = charges.detach().clone()
            qdiff = torch.linalg.norm(data.old_charges - q)

            print(
                f"{data.iter:3}   {energy.sum(): .16E}  {ediff: .6E} " f"{qdiff: .6E}"
            )

            data.old_energy = energy
            data.old_charges = q
            data.old_density = density
            data.iter += 1

    if cfg.fwd_options["verbose"] > 1:  # pragma: no cover
        print(f"energy: {get_energy(charges, data, interactions).sum(-1)}")
    potential = charges_to_potential(charges, interactions, data)
    return potential_to_charges(potential, data, cfg)


def iterate_potential(
    potential: Tensor, data: _Data, cfg: SCFConfig, interactions: InteractionList
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

    charges = potential_to_charges(potential, data, cfg)
    if cfg.scf_options["verbosity"] > 0:
        if charges.ndim < 2:  # pragma: no cover
            energy = get_energy(charges, data, interactions).sum(-1).detach().clone()
            ediff = torch.linalg.vector_norm(data.old_energy - energy)

            density = data.density.detach().clone()
            pnorm = torch.linalg.matrix_norm(data.old_density - density)

            q = charges.detach().clone()
            qdiff = torch.linalg.vector_norm(data.old_charges - q)

            print(
                f"{data.iter:3}   {energy: .16E}  {ediff: .6E} "
                f"{pnorm: .6E}   {qdiff: .6E}"
            )

            data.old_energy = energy
            data.old_charges = q
            data.old_density = density
            data.iter += 1
        else:
            energy = get_energy(charges, data, interactions).detach().clone()
            ediff = torch.linalg.norm(data.old_energy - energy)

            density = data.density.detach().clone()
            pnorm = torch.linalg.norm(data.old_density - density)

            q = charges.detach().clone()
            qdiff = torch.linalg.norm(data.old_charges - q)

            print(
                f"{data.iter:3}   {energy.sum(): .16E}  {ediff: .6E} " f"{qdiff: .6E}"
            )

            data.old_energy = energy
            data.old_charges = q
            data.old_density = density
            data.iter += 1

    return charges_to_potential(charges, interactions, data)


def iterate_fockian(
    fockian: Tensor, data: _Data, cfg: SCFConfig, interactions: InteractionList
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
    charges = density_to_charges(data.density, data)
    potential = charges_to_potential(charges, interactions, data)
    data.hamiltonian = potential_to_hamiltonian(potential, data)

    return data.hamiltonian


iter_options = {
    "charge": iterate_charges,
    "charges": iterate_charges,
    "potential": iterate_potential,
    "fock": iterate_fockian,
}
"""Possible physical values to be iterated during SCF procedure."""
