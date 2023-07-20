from __future__ import annotations

import torch

from .._types import Tensor
from ..constants import defaults
from ..utils import real_atoms
from ..wavefunction import mulliken
from ..interaction import InteractionList

from .data import _Data
from .config import SCF_Config


def get_energy(charges: Tensor, data: _Data, interactions: InteractionList) -> Tensor:
    """
    Get the energy of the system with the given charges.

    Parameters
    ----------
    charges : Tensor
        Orbital charges vector.
    data: _Data
        Storage for tensors which become part of autograd graph within SCF cycles.
    interactions : InteractionList
        Collection of `Interation` objects.

    Returns
    -------
    Tensor
        Energy of the system.
    """
    energy = data.ihelp.reduce_orbital_to_atom(data.energy)
    return energy + interactions.get_energy(charges, data.cache, data.ihelp)


def get_energy_as_dict(
    charges: Tensor, data: _Data, interactions: InteractionList
) -> dict[str, Tensor]:
    """
    Get the energy of the system with the given charges.

    Parameters
    ----------
    charges : Tensor
        Orbital charges vector.
    data: _Data
        Storage for tensors which become part of autograd graph within SCF cycles.
    interactions : InteractionList
        Collection of `Interation` objects.

    Returns
    -------
    Tensor
        Energy of the system.
    """
    energy_h0 = {"h0": data.energy}

    energy_interactions = interactions.get_energy_as_dict(
        charges, data.cache, data.ihelp
    )

    return {**energy_h0, **energy_interactions}


def get_electronic_free_energy(data: _Data, cfg: SCF_Config) -> Tensor:
    r"""
    Calculate electronic free energy from entropy.

    .. math::

        G = -TS = k_B\sum_{i}f_i \; ln(f_i) + (1 - f_i)\; ln(1 - f_i))

    The atomic partitioning can be performed by means of Mulliken population
    analysis using an "electronic entropy" density matrix.

    .. math::

        E_\kappa^\text{TS} = (\mathbf P^\text{TS} \mathbf S)_{\kappa\kappa}
        \qquad\text{with}\quad \mathbf P^\text{TS} = \mathbf C^T \cdot
        \text{diag}(g) \cdot \mathbf C

    Returns
    -------
    Tensor
        Orbital-resolved electronic free energy (G = -TS).

    Note
    ----
    Partitioning scheme is set through SCF options
    (`scf_options["fermi_fenergy_partition"]`).
    Defaults to an equal partitioning to all atoms (`"equal"`).
    """
    eps = torch.tensor(
        torch.finfo(data.occupation.dtype).eps,
        device=cfg.device,
        dtype=cfg.dtype,
    )

    occ = torch.clamp(data.occupation, min=eps)
    occ1 = torch.clamp(1 - data.occupation, min=eps)
    g = torch.log(occ**occ * occ1**occ1).sum(-2) * cfg.kt

    mode = cfg.scf_options.get(
        "fermi_fenergy_partition", defaults.FERMI_FENERGY_PARTITION
    )

    # partition to atoms equally
    if mode == "equal":
        real = real_atoms(data.numbers)

        count = real.count_nonzero(dim=-1).unsqueeze(-1)
        g_atomic = torch.sum(g, dim=-1, keepdim=True) / count

        return torch.where(real, g_atomic.expand(*real.shape), g.new_tensor(0.0))

    # partition to atoms via Mulliken population analysis
    if mode == "atomic":
        # "electronic entropy" density matrix
        density = torch.einsum(
            "...ik,...k,...jk->...ij",
            data.evecs,  # sorted by energy, starting with lowest
            g,
            data.evecs,  # transposed
        )

        return mulliken.get_atomic_populations(data.overlap, density, data.ihelp)

    raise ValueError(f"Unknown partitioning mode '{mode}'.")