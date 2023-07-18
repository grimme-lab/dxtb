
from __future__ import annotations

from abc import abstractmethod

import torch

from .._types import Any, SCFResult, Slicers, Tensor
from ..basis import IndexHelper
from ..constants import K2AU, defaults
from ..exlibs.xitorch import EditableModule, LinearOperator
from ..exlibs.xitorch import linalg as xtl
from ..exlibs.xitorch import optimize as xto
from ..interaction import InteractionList
from ..utils import eighb, real_atoms
from ..wavefunction import filling, mulliken

from .base import get_density
from .data import _Data
from .config import SCF_Config
from .ovlp_diag import diagonalize

"""
Conversion of charges, potential, hamiltonian, ... implemented as pure 
functions in order to avoid RAM leak due to circular references.

NOTE: Conversion methods are designed as semi-pure functions (i.e. contain `data.attr = x`).
      Therefore, make sure to delete attributes manually at end of scope (i.e. `del data.attr`).
"""


def converged_to_charges(x: Tensor, data: _Data, cfg: SCF_Config) -> Tensor:
    """
        Convert the converged property to charges.

        Parameters
        ----------
        x : Tensor
            Converged property (scp).
        data: _Data
            Object holding SCF data.
        cfg: SCF_Config
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
        charges =  potential_to_charges(x, data, cfg.kt, cfg.scf_options)
        # data.density = density # TODO: causes RAM leak (when not )
        # data.density_new = density
        # del data.density_new # works
        return charges
        # return potential_to_charges(x, data, cfg.kt, cfg.scf_options)

    if cfg.scp_mode == "fock":
        data.density = hamiltonian_to_density(x)
        return density_to_charges(data.density)

    raise ValueError(f"Unknown convergence target (SCP mode) '{cfg.scp_mode}'.")
    

def charges_to_potential(charges: Tensor, interactions, data) -> Tensor:
    return interactions.get_potential(
        charges, data.cache, data.ihelp
    )

def potential_to_charges(potential: Tensor, data, kt, scf_options) -> Tensor:
    """
    Compute the orbital charges from the potential.

    Parameters
    ----------
    potential : Tensor
        Potential vector for each orbital partial charge.

    Returns
    -------
    Tensor
        Orbital-resolved partial charges vector.
    """

    data.density = potential_to_density(potential, data, kt, scf_options)
    return density_to_charges(data.density, data)
   
   
    # TODO: writing back to data object
    # density = potential_to_density(potential, data, kt, scf_options)
    data.density = potential_to_density(potential, data, kt, scf_options)
    
    # data.density = density # TODO: causes RAM leak
    # Do I need to write it back to object here?? (or only at the end of the whole conversion)
    # update function of data needed

    return density_to_charges(data.density, data)#, density
    # data.density = potential_to_density(potential, data, kt, scf_options)
    # return density_to_charges(data.density, data)

def potential_to_density(potential: Tensor, data, kt, scf_options) -> Tensor:
    # TODO: writing back to data object
    hamiltonian = potential_to_hamiltonian(potential, data)

    # hamiltonian = potential.unsqueeze(-1) + potential.unsqueeze(-2) # does not work
    # hamiltonian = torch.eye(len(potential)) * potential # does not work

    # symmetric random tensor
    tensor = torch.rand(hamiltonian.shape)
    # hamiltonian = (tensor + tensor.transpose(-2, -1)) / 2 # works
    # NOTE: requires no grad during xto.equilibrium but during converged_to_charges
    #       during converged_to_charges is when the RAM leak happens

    return hamiltonian_to_density(hamiltonian, data, kt, scf_options)
    # data.hamiltonian = potential_to_hamiltonian(potential, data)
    # return hamiltonian_to_density(data.hamiltonian, data, kt, scf_options)

def density_to_charges(density: Tensor, data) -> Tensor:
    # TODO: writing back to data object

    # data.energy = torch.diagonal(
    #     torch.einsum("...ik,...kj->...ij", density, data.hcore),
    #     dim1=-2,
    #     dim2=-1,
    # )

    populations = torch.diagonal(
        torch.einsum("...ik,...kj->...ij", density, data.overlap),
        dim1=-2,
        dim2=-1,
    )
    return data.n0 - populations

def potential_to_hamiltonian(potential: Tensor, data) -> Tensor:
    """
    Compute the Hamiltonian from the potential.

    Parameters
    ----------
    potential : Tensor
        Potential vector for each orbital partial charge.

    Returns
    -------
    Tensor
        Hamiltonian matrix.
    """
    return data.hcore - 0.5 * data.overlap * (
        potential.unsqueeze(-1) + potential.unsqueeze(-2)
    )


def hamiltonian_to_density(hamiltonian: Tensor, data, kt, scf_options: dict, ) -> Tensor:

    # TODO: diagonalize from SCF object required

    # TODO:
    eigen_options = {"method": "exacteig"}

    data_evals, data_evecs = diagonalize(hamiltonian, data.overlap, eigen_options)
    # data.evals, data.evecs = diagonalize(hamiltonian, data)

    # round to integers to avoid numerical errors
    nel = data.occupation.sum(-1).round()

    # expand emo/mask to second dim (for alpha/beta electrons)
    emo = data_evals.unsqueeze(-2).expand([*nel.shape, -1])
    mask = data.ihelp.spread_shell_to_orbital(
        data.ihelp.orbitals_per_shell
    )
    mask = mask.unsqueeze(-2).expand([*nel.shape, -1])

    # Fermi smearing only for non-zero electronic temperature
    if kt is not None and not torch.all(kt < 3e-7):  # 0.1 Kelvin * K2AU
        data_occupation = filling.get_fermi_occupation(
            nel,
            emo,
            kt=kt,
            mask=mask,
            maxiter=scf_options.get("fermi_maxiter", defaults.FERMI_MAXITER),
            thr=scf_options.get("fermi_thresh", defaults.THRESH),
        )

        # check if number of electrons is still correct
        _nel = data_occupation.sum(-1)
        if torch.any(torch.abs(nel - _nel.round(decimals=3)) > 1e-4):
            raise RuntimeError(
                f"Number of electrons changed during Fermi smearing "
                f"({nel} -> {_nel})."
            )
    
    return get_density(data_evecs, data_occupation.sum(-2)) # works
    return get_density(data.evecs, data.occupation.sum(-2))