"""
Conversions
===========

Conversion of charges, potential, hamiltonian, ... implemented as pure
functions in order to avoid RAM leak due to circular references.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum
from tad_mctc.units.energy import KELVIN2AU

from dxtb.components.interactions import InteractionList
from dxtb.components.interactions.container import Charges, Potential
from dxtb.config import ConfigSCF
from dxtb.constants import defaults, labels
from dxtb.typing import Tensor
from dxtb.wavefunction import filling

from ..utils import get_density
from .data import _Data
from .ovlp_diag import diagonalize

# NOTE:
# Conversion methods are designed as semi-pure functions (i.e. contain
# `data.attr = x`). Therefore, make sure to delete attributes manually at end
# of scope (i.e. `del data.attr`).


def converged_to_charges(x: Tensor, data: _Data, config: ConfigSCF) -> Charges:
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

    if config.scp_mode == labels.SCP_MODE_CHARGE:
        return Charges.from_tensor(x, data.charges, batch_mode=config.batch_mode)

    if config.scp_mode == labels.SCP_MODE_POTENTIAL:
        pot = Potential.from_tensor(x, data.potential, batch_mode=config.batch_mode)
        return potential_to_charges(pot, data, cfg=config)

    if config.scp_mode == labels.SCP_MODE_FOCK:
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        x = torch.where(x != defaults.PADNZ, x, zero)

        data.density = hamiltonian_to_density(x, data, config)
        return density_to_charges(data.density, data)

    raise ValueError(f"Unknown convergence target (SCP mode) '{config.scp_mode}'.")


def charges_to_potential(
    charges: Charges, interactions: InteractionList, data: _Data
) -> Potential:
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

    potential = interactions.get_potential(charges, data.cache, data.ihelp)
    data.potential = {
        "mono": potential.mono_shape,
        "dipole": potential.dipole_shape,
        "quad": potential.quad_shape,
        "label": potential.label,
    }

    return potential


def potential_to_charges(potential: Potential, data: _Data, cfg: ConfigSCF) -> Charges:
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


def potential_to_density(potential: Potential, data: _Data, cfg: ConfigSCF) -> Tensor:
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


def density_to_charges(density: Tensor, data: _Data) -> Charges:
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
        torch.einsum("...ik,...kj->...ij", density, data.ints.hcore),
        dim1=-2,
        dim2=-1,
    )

    populations = torch.diagonal(
        torch.einsum("...ik,...kj->...ij", density, data.ints.overlap),
        dim1=-2,
        dim2=-1,
    )

    charges = Charges(mono=data.n0 - populations, batch_mode=data.ihelp.batch_mode)
    return charges


def potential_to_hamiltonian(potential: Potential, data: _Data) -> Tensor:
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
    h1 = data.ints.hcore

    if potential.mono is not None:
        v = potential.mono.unsqueeze(-1) + potential.mono.unsqueeze(-2)
        h1 = h1 - (0.5 * data.ints.overlap * v)

    def add_vmp_to_h1(h1: Tensor, mpint: Tensor, vmp: Tensor) -> Tensor:
        # spread potential to orbitals
        v = data.ihelp.spread_atom_to_orbital(vmp, dim=-2, extra=True)

        # Form dot product over the the multipolar components.
        #  - shape multipole integral: (..., x, norb, norb)
        #  - shape multipole potential: (..., norb, x)
        tmp = 0.5 * einsum("...kij,...ik->...ij", mpint, v)
        return h1 - (tmp + tmp.mT)

    if potential.dipole is not None:
        dpint = data.ints.dipole
        if dpint is not None:
            h1 = add_vmp_to_h1(h1, dpint, potential.dipole)

    if potential.quad is not None:
        qpint = data.ints.quadrupole
        if qpint is not None:
            h1 = add_vmp_to_h1(h1, qpint, potential.quad)

    return h1


def hamiltonian_to_density(hamiltonian: Tensor, data: _Data, cfg: ConfigSCF) -> Tensor:
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

    data.evals, data.evecs = diagonalize(
        hamiltonian, data.ints.overlap, cfg.eigen_options
    )

    # round to integers to avoid numerical errors
    nel = data.occupation.sum(-1).round()

    # expand emo/mask to second dim (for alpha/beta electrons)
    emo = data.evals.unsqueeze(-2).expand([*nel.shape, -1])
    mask = data.ihelp.spread_shell_to_orbital(data.ihelp.orbitals_per_shell)
    mask = mask.unsqueeze(-2).expand([*nel.shape, -1])

    # Fermi smearing only for non-zero electronic temperature
    kt = data.ints.hcore.new_tensor(cfg.fermi.etemp * KELVIN2AU)
    if not torch.all(kt < 3e-7):  # 0.1 Kelvin * K2AU
        data.occupation = filling.get_fermi_occupation(
            nel,
            emo,
            kt=kt,
            mask=mask,
            maxiter=cfg.fermi.maxiter,
            thr=cfg.fermi.thresh,
        )

        # check if number of electrons is still correct
        _nel = data.occupation.sum(-1)
        if torch.any(torch.abs(nel - _nel.round(decimals=3)) > 1e-4):
            raise RuntimeError(
                f"Number of electrons changed during Fermi smearing "
                f"({nel} -> {_nel})."
            )

    return get_density(data.evecs, data.occupation.sum(-2))
