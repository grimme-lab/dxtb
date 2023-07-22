"""
Conversions
===========

Conversion of charges, potential, hamiltonian, ... implemented as pure
functions in order to avoid RAM leak due to circular references.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from ..constants import defaults
from ..interaction import InteractionList
from ..wavefunction import filling
from .base import get_density
from .config import SCFConfig
from .data import _Data
from .ovlp_diag import diagonalize

# NOTE:
# Conversion methods are designed as semi-pure functions (i.e. contain
# `data.attr = x`). Therefore, make sure to delete attributes manually at end
# of scope (i.e. `del data.attr`).


def converged_to_charges(x: Tensor, data: _Data, cfg: SCFConfig) -> Tensor:
    """
    Convert the converged property to charges.

    Parameters
    ----------
    x : Tensor
        Converged property (scp).
    data: _Data
        Object holding SCF data.
    cfg: SCFConfig
        Configuration for SCF settings.

    Returns
    -------
    Tensor
        Orbital-resolved partial charges

    Raises
    ------
    ValueError
        Unknown `scp_mode` given.
    """

    if cfg.scp_mode in ("charge", "charges"):
        return x

    if cfg.scp_mode == "potential":
        return potential_to_charges(x, data, cfg)

    if cfg.scp_mode == "fock":
        data.density = hamiltonian_to_density(x, data, cfg)
        return density_to_charges(data.density, data)

    raise ValueError(f"Unknown convergence target (SCP mode) '{cfg.scp_mode}'.")


def charges_to_potential(
    charges: Tensor, interactions: InteractionList, data: _Data
) -> Tensor:
    """
    Compute the potential from the orbital charges.

    Parameters
    ----------
    charges : Tensor
        Orbital-resolved partial charges vector.
    interactions : InteractionList
        Collection of `Interation` objects.
    data: _Data
        Storage for tensors which become part of autograd graph within SCF cycles.

    Returns
    -------
    Tensor
        Potential vector for each orbital partial charge.
    """
    return interactions.get_potential(charges, data.cache, data.ihelp)


def potential_to_charges(potential: Tensor, data: _Data, cfg: SCFConfig) -> Tensor:
    """
    Compute the orbital charges from the potential.

    Parameters
    ----------
    potential : Tensor
        Potential vector for each orbital partial charge.
    data: _Data
        Data cache for intermediary storage during self-consistency.
    cfg: SCFConfig
        Configuration for SCF settings.

    Returns
    -------
    Tensor
        Orbital-resolved partial charges vector.
    """

    data.density = potential_to_density(potential, data, cfg)
    return density_to_charges(data.density, data)


def potential_to_density(potential: Tensor, data: _Data, cfg: SCFConfig) -> Tensor:
    """
    Obtain the density matrix from the potential.

    Parameters
    ----------
    potential : Tensor
        Potential vector for each orbital partial charge.
    data: _Data
        Data cache for intermediary storage during self-consistency.
    cfg: SCFConfig
        Configuration for SCF settings.

    Returns
    -------
    Tensor
        Density matrix.
    """

    data.hamiltonian = potential_to_hamiltonian(potential, data)
    return hamiltonian_to_density(data.hamiltonian, data, cfg)


def density_to_charges(density: Tensor, data: _Data) -> Tensor:
    """
    Compute the orbital charges from the density matrix.

    Parameters
    ----------
    density : Tensor
        Density matrix.
    data: _Data
        Data cache for intermediary storage during self-consistency.

    Returns
    -------
    Tensor
        Orbital-resolved partial charges vector.
    """

    data.energy = torch.diagonal(
        torch.einsum("...ik,...kj->...ij", density, data.hcore),
        dim1=-2,
        dim2=-1,
    )

    populations = torch.diagonal(
        torch.einsum("...ik,...kj->...ij", density, data.overlap),
        dim1=-2,
        dim2=-1,
    )
    return data.n0 - populations


def potential_to_hamiltonian(potential: Tensor, data: _Data) -> Tensor:
    """
    Compute the Hamiltonian from the potential.

    Parameters
    ----------
    potential : Tensor
        Potential vector for each orbital partial charge.
    data: _Data
        Data cache for intermediary storage during self-consistency.

    Returns
    -------
    Tensor
        Hamiltonian matrix.
    """

    return data.hcore - 0.5 * data.overlap * (
        potential.unsqueeze(-1) + potential.unsqueeze(-2)
    )


def hamiltonian_to_density(hamiltonian: Tensor, data: _Data, cfg: SCFConfig) -> Tensor:
    """
    Compute the density matrix from the Hamiltonian.

    Parameters
    ----------
    hamiltonian : Tensor
        Hamiltonian matrix.
    data: _Data
        Data cache for intermediary storage during self-consistency.
    cfg: SCFConfig
        Configuration for SCF settings.

    Returns
    -------
    Tensor
        Density matrix.
    """

    data.evals, data.evecs = diagonalize(hamiltonian, data.overlap, cfg.eigen_options)

    # round to integers to avoid numerical errors
    nel = data.occupation.sum(-1).round()

    # expand emo/mask to second dim (for alpha/beta electrons)
    emo = data.evals.unsqueeze(-2).expand([*nel.shape, -1])
    mask = data.ihelp.spread_shell_to_orbital(data.ihelp.orbitals_per_shell)
    mask = mask.unsqueeze(-2).expand([*nel.shape, -1])

    # Fermi smearing only for non-zero electronic temperature
    if cfg.kt is not None and not torch.all(cfg.kt < 3e-7):  # 0.1 Kelvin * K2AU
        data.occupation = filling.get_fermi_occupation(
            nel,
            emo,
            kt=cfg.kt,
            mask=mask,
            maxiter=cfg.scf_options.get("fermi_maxiter", defaults.FERMI_MAXITER),
            thr=cfg.scf_options.get("fermi_thresh", defaults.THRESH),
        )

        # check if number of electrons is still correct
        _nel = data.occupation.sum(-1)
        if torch.any(torch.abs(nel - _nel.round(decimals=3)) > 1e-4):
            raise RuntimeError(
                f"Number of electrons changed during Fermi smearing "
                f"({nel} -> {_nel})."
            )

    return get_density(data.evecs, data.occupation.sum(-2))
