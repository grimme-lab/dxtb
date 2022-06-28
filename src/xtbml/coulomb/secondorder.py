"""
Isotropic second-order electrostatic energy
===========================================

This module implements the second-order electrostatic energy for GFN1-xTB.

Example
-------
>>> import torch
>>> import xtbml.coulomb.secondorder as es2
>>> from xtbml.coulomb.average import harmonic_average as average
>>> from xtbml.param.gfn1 import GFN1_XTB
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [0.00000000000000, -0.00000000000000, 0.00000000000000],
...     [1.61768389755830, 1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [1.61768389755830, -1.61768389755830, 1.61768389755830],
...     [-1.61768389755830, 1.61768389755830, 1.61768389755830],
... ])
>>> qat = torch.tensor([
...     -8.41282505804719e-2,
...     2.10320626451180e-2,
...     2.10320626451178e-2,
...     2.10320626451179e-2,
...     2.10320626451179e-2,
... ])
>>> # get parametrization
>>> gexp = torch.tensor(GFN1_XTB.charge.effective.gexp)
>>> hubbard = es2.get_hubbard_params(GFN1_XTB.element)
>>> # calculate energy
>>> e = es2.get_energy(numbers, positions, qat, hubbard, average, gexp)
>>> torch.set_printoptions(precision=7)
>>> print(torch.sum(e, dim=-1))
tensor(0.0005078)
"""


from __future__ import annotations
import torch

from ..param import Element
from ..typing import Tensor
from .average import AveragingFunction, harmonic_average


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


def get_energy(
    numbers: Tensor,
    positions: Tensor,
    qat: Tensor,
    hubbard: Tensor,
    average: AveragingFunction = harmonic_average,
    gexp: Tensor = torch.tensor(2.0),
) -> Tensor:
    """
    Calculate the second-order Coulomb interaction.

    Implements Eq.25 of the following paper:
    - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
    S. Spicher and S. Grimme, *WIREs Computational Molecular Science*, **2020**, 11, e1493. DOI: `10.1002/wcms.1493 <https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493>`__

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
    average : AveragingFunction
        Function to use for averaging the Hubbard parameters (default: harmonic_average).
    gexp : Tensor
        Exponent of the second-order Coulomb interaction (default: 2.0).

    Returns
    -------
    Tensor
        Atomwise second-order Coulomb interaction energies.
    """

    # masks
    real = numbers != 0
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    mask.diagonal(dim1=-2, dim2=-1).fill_(False)

    # all distances to the power of "gexp" (R^2_AB from Eq.26)
    dist_gexp = torch.where(
        mask,
        torch.pow(torch.cdist(positions, positions, p=2), gexp),
        torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype),
    )

    # Eq.30: averaging function for hardnesses (Hubbard parameter)
    avg = average(hubbard[numbers])

    # Eq.26: Coulomb matrix
    mat = 1.0 / torch.pow(dist_gexp + torch.pow(avg, -gexp), 1.0 / gexp)

    # Eq.25: single and batched matrix-vector multiplication
    mv = 0.5 * torch.einsum("...ik, ...k -> ...i", mat, qat)

    return mv * qat
