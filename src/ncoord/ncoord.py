
import math
import torch

# Steepness of counting function
kcn = 16.0


def get_coordination_number(mol, trans, cutoff, rcov, cn, dcndr, dcndL):
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
        cn, dcndr, dcndL = ncoord_dexp(mol, trans, cutoff, rcov, cn, dcndr, dcndL)
    else:
        cn = ncoord_exp(mol, trans, cutoff, rcov, cn)

    return cn, dcndr, dcndL


def ncoord_exp(mol, trans, cutoff, rcov, cn):

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

    #integer :: iat, jat, izp, jzp, itr
    #real(wp) :: r2, r1, rc, rij(3), countf, cutoff2

    cn = torch.zeros(cn.size()) #TODO: torch.zeros(mol.nat)
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

                countf = exp_count(kcn, r1, rc)

                cn[iat] = cn[iat] + countf
                if (iat != jat):
                    cn[jat] = cn[jat] + countf
    
    return cn


def ncoord_dexp(mol, trans, cutoff, rcov, cn, dcndr, dcndL):

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

                countf = exp_count(kcn, r1, rc)
                countd = dexp_count(kcn, r1, rc) * rij/r1

                cn[iat] += countf
                if (iat != jat):
                    cn[jat] += countf

                dcndr[:, iat, iat] += countd
                dcndr[:, jat, jat] -= countd
                dcndr[:, iat, jat] += countd
                dcndr[:, jat, iat] -= countd

                # TODO: check correct spreading to form 3x3 sigma 
                #sigma = spread(countd, 1, 3) * spread(rij, 2, 3)
                sigma = countd.tile((1,3)) * rij.tile((2,3))

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

    #integer :: iat, jat, izp, jzp, itr
    #real(wp) :: r2, r1, rc, rij(3), countf, countd(3), ds(3, 3), cutoff2

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

                countd = dexp_count(kcn, r1, rc) * rij/r1

                gradient[:, iat] += countd * (dEdcn[iat] + dEdcn[jat])
                gradient[:, jat] -= countd * (dEdcn[iat] + dEdcn[jat])

                # TODO: check correct spreading to form 3x3 ds 
                #ds = spread(countd, 1, 3) * spread(rij, 2, 3)
                ds = countd.tile((1,3)) * rij.tile((2,3))

                # TODO: correct merging -- cf. http://www.lahey.com/docs/lfpro78help/F95ARMERGEFn.htm
                #sigma[:, :] += ds * (dEdcn[iat] + merge(dEdcn[jat], 0.0, jat != iat))
                sigma[:, :] += ds * (dEdcn[iat] + torch.where(jat != iat, dEdcn[jat], 0.0))

    return gradient, sigma


def exp_count(k: float, r: float, r0: float) -> float:
    """ Exponential counting function for coordination number contributions.

    Args:
        k (float): Steepness of the counting function.
        r (float): Current distance.
        r0 (float): Cutoff radius.

    Returns:
        float: Count of coordination number contribution.
    """
    return 1.0/(1.0+math.exp(-k*(r0/r-1.0)))

def dexp_count(k: float, r: float, r0: float) -> float:
    """ Derivative of the counting function w.r.t. the distance.

    Args:
        k (float): Steepness of the counting function.
        r (float): Current distance.
        r0 (float): Cutoff radius.

    Returns:
        float: Derivative of count of coordination number contribution.
    """
    expterm = math.exp(-k*(r0/r-1.))
    return (-k*r0*expterm)/(r**2*((expterm+1.)**2))