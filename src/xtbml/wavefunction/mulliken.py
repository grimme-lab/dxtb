"""
Wavefunction analysis via Mulliken populations.
"""

import torch
from ..typing import Tensor
from ..basis import IndexHelper


def get_orbital_populations(
    overlap: Tensor,
    density: Tensor,
) -> Tensor:
    """
    Compute orbital-resolved populations using Mulliken population analysis.

    Parameters
    ----------
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.

    Returns
    -------
    Tensor
        Orbital populations.
    """

    return torch.diagonal(density @ overlap, dim1=-2, dim2=-1)


def get_shell_populations(
    indexhelper: IndexHelper,
    overlap: Tensor,
    density: Tensor,
) -> Tensor:
    """
    Compute shell-resolved populations using Mulliken population analysis.

    Parameters
    ----------
    indexhelper : IndexHelper
        Index mapping for the basis set.
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.

    Returns
    -------
    Tensor
        Shell populations.
    """

    return torch.scatter_reduce(
        get_orbital_populations(overlap, density),
        -1,
        indexhelper.orbitals_to_shell,
        reduce="sum",
    )


def get_atomic_populations(
    indexhelper: IndexHelper,
    overlap: Tensor,
    density: Tensor,
) -> Tensor:
    """
    Compute atom-resolved populations.

    Parameters
    ----------
    indexhelper : IndexHelper
        Index mapping for the basis set.
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.

    Returns
    -------
    Tensor
        Atom populations.
    """

    return torch.scatter_reduce(
        get_shell_populations(indexhelper, overlap, density),
        -1,
        indexhelper.shells_to_atom,
        reduce="sum",
    )


def get_mulliken_shell_charges(
    indexhelper: IndexHelper,
    overlap: Tensor,
    density: Tensor,
    n0: Tensor,
) -> Tensor:
    """
    Compute shell-resolved Mulliken partial charges using Mulliken population analysis.

    Parameters
    ----------
    indexhelper : IndexHelper
        Index mapping for the basis set.
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.
    n0 : Tensor
        Reference occupancy numbers.

    Returns
    -------
    Tensor
        Shell-resolved Mulliken partial charges.
    """

    return n0 - get_shell_populations(indexhelper, overlap, density)


def get_mulliken_atomic_charges(
    indexhelper: IndexHelper,
    overlap: Tensor,
    density: Tensor,
    n0: Tensor,
) -> Tensor:
    """
    Compute atom-resolved Mulliken partial charges.

    Parameters
    ----------
    indexhelper : IndexHelper
        Index mapping for the basis set.
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.
    n0 : Tensor
        Reference occupancy numbers.

    Returns
    -------
    Tensor
        Atom-resolved Mulliken partial charges.
    """

    return n0 - get_atomic_populations(indexhelper, overlap, density)
