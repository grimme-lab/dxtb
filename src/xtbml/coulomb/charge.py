from __future__ import annotations
import torch

from ..typing import Tensor
from xtbml.param.element import Element


def get_second_order(
    numbers: Tensor, positions: Tensor, qat: Tensor, hubbard: Tensor, gexp: Tensor
) -> Tensor:
    """Calculate the second-order Coulomb interaction.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms.
    positions : Tensor
        Cartesian coordinates of all atoms in the system.
    qat : Tensor
        Atomic charges of all atoms.
    hubbard : Tensor
        Hubbard parameters of all elements.
    gexp : Tensor
        Exponent of the second-order Coulomb interaction.

    Returns
    -------
    Tensor
        Atomwise second-order Coulomb interaction energies.
    """

    # masks
    real = numbers != 0
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    mask.diagonal(dim1=-2, dim2=-1).fill_(False)

    # all distances to the power of "gexp"
    dist_gexp = torch.where(
        mask,
        torch.pow(torch.cdist(positions, positions, p=2), gexp),
        torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype),
    )

    # averaging function for hardnesses (Hubbard parameter)
    avg = harmonic_average(hubbard[numbers])

    # Coulomb matrix
    mat = 1.0 / torch.pow(dist_gexp + torch.pow(avg, -gexp), 1.0 / gexp)

    # single and batched matrix-vector multiplication
    mv = 0.5 * torch.einsum("...ik, ...k -> ...i", mat, qat)

    return mv * qat


def get_hubbard_params(par_element: dict[str, Element]) -> Tensor:
    """Obtain hubbard parameters from parametrization of elements.

    Parameters
    ----------
    par : dict[str, Element]
        Parametrization of elements.
    Returns
    -------
    Tensor
        Hubbard parameters of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
    """

    # dummy for indexing with atomic numbers
    g = [0.0]

    for item in par_element.values():
        g.append(item.gam)

    return torch.tensor(g)


def harmonic_average(hubbard: Tensor) -> Tensor:
    """Harmonic averaging function for hardnesses in GFN1-xTB.

    Parameters
    ----------
    hubbard : Tensor
        Hubbard parameters of all elements.

    Returns
    -------
    Tensor
        Harmonic average of the Hubbard parameters.
    """

    hubbard1 = 1.0 / (hubbard)
    return 2.0 / (hubbard1.unsqueeze(-1) + hubbard1.unsqueeze(-2))
