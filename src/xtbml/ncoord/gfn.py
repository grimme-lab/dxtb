import math
import torch


from .type import NcoordType
from ..data.covrad import get_covalent_rad
from ..cutoff import get_lattice_points
from .ncoord import exp_count, dexp_count

from xtbml.constants import KCN, KA, KB, R_SHIFT, NCOORD_DEFAULT_CUTOFF
from xtbml.exlibs.tbmalt import Geometry

# TODO: differentiate between GFN1 and GFN2
# TODO: GfnNcoordType is currently not used...


class GfnNcoordType(NcoordType):
    def __init__(self, mol, cutoff, rcov):
        """# Coordination number container
        type(gfn_ncoord_type), intent(out) :: self
        # Molecular structure data
        type(structure_type), intent(in) :: mol
        # Real space cutoff
        real(wp), intent(in), optional :: cutoff
        # Covalent radii
        real(wp), intent(in), optional :: rcov(:)"""
        # TODO: docstring

        # TODO: can be refactored as constructor via __init__?

        if cutoff is not None:
            self.cutoff = cutoff
        else:
            self.cutoff = NCOORD_DEFAULT_CUTOFF

        if rcov is not None:
            self.rcov = rcov
        else:
            # FIXME: no mol.num in geometry
            self.rcov = get_covalent_rad(mol.num)

    def get_cn(self, mol, cn, dcndr, dcndL):
        """# Coordination number container
        class(gfn_ncoord_type), intent(in) :: self
        # Molecular structure data
        type(structure_type), intent(in) :: mol
        # Error function coordination number.
        real(wp), intent(out) :: cn(:)
        # Derivative of the CN with respect to the Cartesian coordinates.
        real(wp), intent(out), optional :: dcndr(:, :, :)
        # Derivative of the CN with respect to strain deformations.
        real(wp), intent(out), optional :: dcndL(:, :, :)"""

        # real(wp), allocatable :: lattr(:, :)

        lattr = get_lattice_points(mol.periodic, mol.lattice, self.cutoff)
        cn, dcndr, dcndL = self.get_coordination_number(
            mol, lattr, self.cutoff, self.rcov, cn, dcndr, dcndL
        )
        return cn, dcndr, dcndL

    def get_coordination_number(
        self, mol: Geometry, trans, cutoff, rcov, cn, dcndr, dcndL
    ):

        """Geometric fractional coordination number, supports exponential counting functions.

        Args:
            mol (Geometry): Molecular structure data
            trans (_type_): Lattice points
            cutoff (_type_): Real space cutoff
            rcov (_type_): Covalent radius
            cn (_type_): Error function coordination number
            dcndr (_type_): Derivative of the CN with respect to the Cartesian coordinates
            dcndL (_type_): Derivative of the CN with respect to strain deformations
        """

        if (dcndr is not None) and (dcndL is not None):
            cn, dcndr, dcndL = self.ncoord_dexp(
                mol, trans, cutoff, rcov, cn, dcndr, dcndL
            )
        else:
            cn = self.ncoord_exp(mol, trans, cutoff, rcov, cn)

        return cn, dcndr, dcndL

    def ncoord_exp(self, mol, trans, cutoff, rcov, cn):
        """# Molecular structure data
        type(structure_type), intent(in) :: mol
        # Lattice points
        real(wp), intent(in) :: trans(:, :)
        # Real space cutoff
        real(wp), intent(in) :: cutoff
        # Covalent radius
        real(wp), intent(in) :: rcov(:)
        # Error function coordination number.
        real(wp), intent(out) :: cn(:)"""
        # TODO: docstring

        # integer :: iat, jat, izp, jzp, itr
        # real(wp) :: r2, r1, rc, rij(3), countf, cutoff2

        cn = torch.zeros(cn.size())  # TODO: torch.zeros(mol.nat)
        cutoff2 = cutoff**2

        for iat in range(mol.nat):
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

                    countf = exp_count(KA, r1, rc) * exp_count(KB, r1, rc + R_SHIFT)

                    cn[iat] = cn[iat] + countf
                    if iat != jat:
                        cn[jat] = cn[jat] + countf

        return cn

    def ncoord_dexp(self, mol, trans, cutoff, rcov, cn, dcndr, dcndL):

        # TODO: similar to import ncoord.ncoord.ncoord_dexp() --> only final function inside differs! --> why, only because of different cn schemes?

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

        for iat in range(mol.nat):
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

                    # TODO: only difference
                    countf = exp_count(KA, r1, rc) * exp_count(KB, r1, rc + R_SHIFT)
                    countd = (
                        (
                            dexp_count(KA, r1, rc) * exp_count(KB, r1, rc + R_SHIFT)
                            + exp_count(KA, r1, rc) * dexp_count(KB, r1, rc + R_SHIFT)
                        )
                        * rij
                        / r1
                    )

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
