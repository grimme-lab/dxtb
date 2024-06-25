"""
Energy
======

Functions for energy calculations within the SCF.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import real_atoms
from tad_mctc.math import einsum
from tad_mctc.units.energy import KELVIN2AU

from dxtb._src.components.interactions import Charges, InteractionList
from dxtb._src.constants import labels
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction import mulliken
from dxtb.config import ConfigSCF

from .data import _Data

__all__ = [
    "get_energy",
    "get_energy_as_dict",
    "get_electronic_free_energy",
]


def get_energy(charges: Charges, data: _Data, interactions: InteractionList) -> Tensor:
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
    charges: Charges, data: _Data, interactions: InteractionList
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


def get_electronic_free_energy(data: _Data, cfg: ConfigSCF) -> Tensor:
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

    kt = data.ints.hcore.new_tensor(cfg.fermi.etemp * KELVIN2AU)
    occ = torch.clamp(data.occupation, min=eps)
    occ1 = torch.clamp(1 - data.occupation, min=eps)
    g = torch.log(occ**occ * occ1**occ1).sum(-2) * kt

    # partition to atoms equally
    if cfg.fermi.partition == labels.FERMI_PARTITION_EQUAL:
        real = real_atoms(data.numbers)

        count = real.count_nonzero(dim=-1).unsqueeze(-1)
        g_atomic = torch.sum(g, dim=-1, keepdim=True) / count

        return torch.where(real, g_atomic.expand(*real.shape), g.new_tensor(0.0))

    # partition to atoms via Mulliken population analysis
    if cfg.fermi.partition == labels.FERMI_PARTITION_ATOMIC:
        # "electronic entropy" density matrix
        density = einsum(
            "...ik,...k,...jk->...ij",
            data.evecs,  # sorted by energy, starting with lowest
            g,
            data.evecs,  # transposed
        )

        return mulliken.get_atomic_populations(data.ints.overlap, density, data.ihelp)

    part = labels.FERMI_PARTITION_MAP[cfg.fermi.partition]
    raise ValueError(f"Unknown partitioning mode '{part}'.")
