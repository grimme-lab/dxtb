import math
from typing import Dict
import torch

from xtbml.exlibs.tbmalt import Geometry
from xtbml.constants import KCN

# TODO: differentiate GFN1 and GFN2
# from xtbml.constants import KCN, KA, KB, R_SHIFT


Tensor = torch.Tensor

def get_coordination_number_d3(
    geometry: Geometry,
    rcov: Tensor,
    cutoff: float = 25.0,
    kcn: float = 16.0,
    symmetric: bool = True,
) -> Tensor:
    """
    Compute fractional coordination number using an exponential counting function.
    """

    numbers = geometry.atomic_numbers
    distances = geometry.distances
    real = numbers != 0
    mask = ~(real.unsqueeze(-2) * real.unsqueeze(-1))
    torch.diagonal(mask, dim1=-2, dim2=-1)[:] = True

    rc = rcov[numbers].unsqueeze(-2) + rcov[numbers].unsqueeze(-1)
    cf = 1.0 / (1.0 + torch.exp(-kcn * (rc / distances - 1.0)))
    cf[mask] = 0
    return torch.sum(cf, dim=-1)


def get_coordination_number(
    mol: Geometry,
    trans: torch.Tensor,
    cutoff: float,
    rcov: Dict[str, float],
    dcndr=None,
    dcndL=None,
):
    """Geometric fractional coordination number, supports exponential counting
    functions."""

    """# Molecular structure data
    type(structure_type), intent(in) :: mol
    # Lattice points
    real(wp), intent(in) :: trans(:, :)
    # Real space cutoff
    real(wp), intent(in) :: cutoff
    # Covalent radius
    real(wp), intent(in) :: rcov(:)
    # Error function coordination number.
    real(wp), intent(out) :: cn(:)
    # Derivative of the CN with respect to the Cartesian coordinates.
    real(wp), intent(out), optional :: dcndr(:, :, :)
    # Derivative of the CN with respect to strain deformations.
    real(wp), intent(out), optional :: dcndL(:, :, :)"""
    # TODO: docstring

    # TODO:
    if (dcndr is not None) and (dcndL is not None):
        return ncoord_dexp(mol, trans, cutoff, rcov, dcndr, dcndL)
    else:
        return ncoord_exp(mol, trans, cutoff, rcov)


def ncoord_exp(
    mol: Geometry, trans: torch.Tensor, cutoff: float, rcov: Dict[str, float]
) -> torch.Tensor:
    """Geometric fractional coordination number from exponential counting function.

    Args:
        mol (Geometry): Molecular structure data
        trans (torch.Tensor): Lattice points
        cutoff (float): Real space cutoff
        rcov (dict): Covalent radius

    Returns:
        torch.Tensor: Error function coordination number
    """

    cn = torch.zeros(mol.get_length())
    cutoff2 = cutoff**2

    for i, el_1 in enumerate(mol.chemical_symbols):
        for j in range(i + 1):
            el_2 = mol.chemical_symbols[j]
            for itr in range(trans.size(dim=1)):
                rij = torch.sub(mol.positions[i, :], mol.positions[j, :])
                rij = torch.sub(rij, trans[itr, :])
                r2 = sum(rij**2)

                if (r2 > cutoff2) or (r2 < 1.0e-12):
                    continue
                r1 = math.sqrt(r2)

                rc = rcov[el_1] + rcov[el_2]

                countf = exp_count(KCN, r1, rc)

                cn[i] += countf
                if i != j:
                    cn[j] += countf

    return cn


# FIXME: same as ncoord_exp
def ncoord_dexp(mol, trans, cutoff, rcov, dcndr, dcndL):

    """# Molecular structure data
    type(structure_type), intent(in) :: mol
    # Lattice points
    real(wp), intent(in) :: trans(:, :)
    # Real space cutoff
    real(wp), intent(in) :: cutoff
    # Covalent radius
    real(wp), intent(in) :: rcov(:)
    # Error function coordination number.
    real(wp), intent(out) :: cn(:)
    # Derivative of the CN with respect to the Cartesian coordinates.
    real(wp), intent(out) :: dcndr(:, :, :)
    # Derivative of the CN with respect to strain deformations.
    real(wp), intent(out) :: dcndL(:, :, :)"""
    # TODO: docstring

    # integer :: iat, jat, izp, jzp, itr
    # real(wp) :: r2, r1, rc, rij(3), countf, countd(3), sigma(3, 3), cutoff2

    cn = torch.zeros(cn.size())
    dcndr = torch.zeros(dcndr.size())
    dcndL = torch.zeros(dcndL.size())
    cutoff2 = cutoff**2

    for iat in range(mol.get_length()):
        izp = mol.id[iat]
        for jat in range(iat):
            jzp = mol.id[jat]

            for itr in range(trans.size()[1]):
                rij = mol.xyz[:, iat] - (mol.xyz[:, jat] + trans[:, itr])
                r2 = sum(rij**2)
                if (r2 > cutoff2) or (r2 < 1.0e-12):
                    continue
                r1 = math.sqrt(r2)
                rc = rcov[izp] + rcov[jzp]

                countf = exp_count(KCN, r1, rc)
                countd = dexp_count(KCN, r1, rc) * rij / r1

                cn[iat] += countf
                if iat != jat:
                    cn[jat] += countf

                dcndr[:, iat, iat] += countd
                dcndr[:, jat, jat] -= countd
                dcndr[:, iat, jat] += countd
                dcndr[:, jat, iat] -= countd

                # TODO: check correct spreading to form 3x3 sigma
                # sigma = spread(countd, 1, 3) * spread(rij, 2, 3)
                sigma = countd.tile((1, 3)) * rij.tile((2, 3))

                dcndL[:, :, iat] += sigma
                if iat != jat:
                    dcndL[:, :, jat] += sigma

    return cn, dcndr, dcndL


def add_coordination_number_derivs(mol, trans, cutoff, rcov, dEdcn, gradient, sigma):

    """# Molecular structure data
    type(structure_type), intent(in) :: mol
    # Lattice points
    real(wp), intent(in) :: trans[:, :]
    # Real space cutoff
    real(wp), intent(in) :: cutoff
    # Covalent radius
    real(wp), intent(in) :: rcov(:)
    # Derivative of expression with respect to the coordination number
    real(wp), intent(in) :: dEdcn(:)
    # Derivative of the CN with respect to the Cartesian coordinates
    real(wp), intent(inout) :: gradient(:, :)
    # Derivative of the CN with respect to strain deformations
    real(wp), intent(inout) :: sigma(:, :)"""

    # integer :: iat, jat, izp, jzp, itr
    # real(wp) :: r2, r1, rc, rij(3), countf, countd(3), ds(3, 3), cutoff2

    cutoff2 = cutoff**2

    for iat in range(mol.get_length()):
        izp = mol.id[iat]
        for jat in range(iat):
            jzp = mol.id[jat]

            for itr in range(trans.size()[1]):
                rij = mol.xyz[:, iat] - (mol.xyz[:, jat] + trans[:, itr])
                r2 = sum(rij**2)
                if (r2 > cutoff2) or (r2 < 1.0e-12):
                    continue
                r1 = math.sqrt(r2)
                rc = rcov[izp] + rcov[jzp]

                countd = dexp_count(KCN, r1, rc) * rij / r1

                gradient[:, iat] += countd * (dEdcn[iat] + dEdcn[jat])
                gradient[:, jat] -= countd * (dEdcn[iat] + dEdcn[jat])

                # TODO: check correct spreading to form 3x3 ds
                # ds = spread(countd, 1, 3) * spread(rij, 2, 3)
                ds = countd.tile((1, 3)) * rij.tile((2, 3))

                # TODO: correct merging -- cf. http://www.lahey.com/docs/lfpro78help/F95ARMERGEFn.htm
                # sigma[:, :] += ds * (dEdcn[iat] + merge(dEdcn[jat], 0.0, jat != iat))
                sigma[:, :] += ds * (
                    dEdcn[iat] + torch.where(jat != iat, dEdcn[jat], 0.0)
                )

    return gradient, sigma


def exp_count(k: float, r: float, r0: float) -> float:
    """Exponential counting function for coordination number contributions.

    Args:
        k (float): Steepness of the counting function.
        r (float): Current distance.
        r0 (float): Cutoff radius.

    Returns:
        float: Count of coordination number contribution.
    """
    return 1.0 / (1.0 + math.exp(-k * (r0 / r - 1.0)))


def dexp_count(k: float, r: float, r0: float) -> float:
    """Derivative of the counting function w.r.t. the distance.

    Args:
        k (float): Steepness of the counting function.
        r (float): Current distance.
        r0 (float): Cutoff radius.

    Returns:
        float: Derivative of count of coordination number contribution.
    """
    expterm = math.exp(-k * (r0 / r - 1.0))
    return (-k * r0 * expterm) / (r**2 * ((expterm + 1.0) ** 2))
