

import math
import torch

from .overlap import msao, mlao, maxl, maxl2, sqrtpi3, lx, lmap, _overlap_1d, _horizontal_shift, _form_product
from ..exlibs.tblite import dimDipole, dimQuadrupole # TODO: merge these configs into single file
from .trafo import transform0, transform1, transform2


def multipole_cgto(cgtoj, cgtoi, r2, vec, intcut, overlap, dpint, qpint):
    # !> Description of contracted Gaussian function on center i
    # type(cgto_type), intent(in) :: cgtoi
    # !> Description of contracted Gaussian function on center j
    # type(cgto_type), intent(in) :: cgtoj
    # !> Square distance between center i and j
    # real(wp), intent(in) :: r2
    # !> Distance vector between center i and j, ri - rj
    # real(wp), intent(in) :: vec(3)
    # !> Maximum value of integral prefactor to consider
    # real(wp), intent(in) :: intcut
    # !> Overlap integrals for the given pair i  and j
    # real(wp), intent(out) :: overlap(msao(cgtoj.ang), msao(cgtoi.ang))
    # !> Dipole moment integrals for the given pair i  and j
    # real(wp), intent(out) :: dpint(3, msao(cgtoj.ang), msao(cgtoi.ang))
    # !> Quadrupole moment integrals for the given pair i  and j
    # real(wp), intent(out) :: qpint(6, msao(cgtoj.ang), msao(cgtoi.ang))
    # TODO: docstring


    s1d = torch.zeros((maxl2))
    s3d = torch.zeros((mlao[cgtoj.ang],mlao[cgtoi.ang]))
    d3d = torch.zeros((dimDipole, mlao[cgtoj.ang],mlao[cgtoi.ang]))
    q3d = torch.zeros((dimQuadrupole, mlao[cgtoj.ang],mlao[cgtoi.ang]))


    for ip in range(cgtoi.nprim):
        for jp in range(cgtoj.nprim):
            eab = cgtoi.alpha[ip] + cgtoj.alpha[jp]
            oab = 1.0/eab
            est = cgtoi.alpha[ip] * cgtoj.alpha[jp] * r2 * oab
            if (est > intcut):
                continue
            pre = math.exp(-est) * sqrtpi3*math.sqrt(oab)**3
            rpi = -vec * cgtoj.alpha[jp] * oab
            rpj = +vec * cgtoi.alpha[ip] * oab
            for l in range(cgtoi.ang + cgtoj.ang + 2 + 1): # TODO
              s1d[l] = _overlap_1d(l, eab)
            cc = cgtoi.coeff[ip] * cgtoj.coeff[jp] * pre
            for mli in range(mlao[cgtoi.ang]):
                for mlj in range(mlao[cgtoj.ang]):
                    # TODO
                    val, dip, quad = multipole_3d(rpj, rpi, cgtoj.alpha[jp], cgtoi.alpha[ip], lx[:, mlj+lmap[cgtoj.ang]], lx[:, mli+lmap[cgtoi.ang]], s1d, val, dip, quad)
                    s3d[mlj, mli] = s3d[mlj, mli] + cc*val
                    d3d[:, mlj, mli] = d3d[:, mlj, mli] + cc*dip
                    q3d[:, mlj, mli] = q3d[:, mlj, mli] + cc*quad

    # TODO
    overlap = transform0(cgtoj.ang, cgtoi.ang, s3d, overlap)
    dpint = transform1(cgtoj.ang, cgtoi.ang, d3d, dpint)
    qpint = transform1(cgtoj.ang, cgtoi.ang, q3d, qpint)

    #remove trace from quadrupole integrals (transfrom to spherical harmonics and back)
    for mli in range(msao[cgtoi.ang]):
        for mlj in range(msao[cgtoj.ang]):
            tr = 0.5 * (qpint[1, mlj, mli] + qpint[dimDipole, mlj, mli] + qpint[dimQuadrupole, mlj, mli])
            qpint[0, mlj, mli] = 1.5 * qpint[0, mlj, mli] - tr
            qpint[1, mlj, mli] = 1.5 * qpint[1, mlj, mli]
            qpint[2, mlj, mli] = 1.5 * qpint[2, mlj, mli] - tr
            qpint[3, mlj, mli] = 1.5 * qpint[3, mlj, mli]
            qpint[4, mlj, mli] = 1.5 * qpint[4, mlj, mli]
            qpint[5, mlj, mli] = 1.5 * qpint[5, mlj, mli] - tr

    return overlap, dpint, qpint


def multipole_3d(rpj, rpi, aj: float, ai: float, lj, li, s1d, s3d, d3d, q3d):

    """real(wp), intent(in) :: rpi(3)
    real(wp), intent(in) :: rpj(3)
    integer, intent(in) :: li(3)
    integer, intent(in) :: lj(3)
    real(wp), intent(in) :: s1d(0:)
    real(wp), intent(out) :: s3d
    real(wp), intent(out) :: d3d(3)
    real(wp), intent(out) :: q3d(6)"""

    # TODO: pretty identical setup to overlap._overlap_3d 

    v1d = torch.zeros((3,3))

    for k in range(3):
        vi = torch.zeros((maxl+1))
        vj = torch.zeros((maxl+1))
        vv = torch.zeros((maxl2+1))
        vi[li[k]] = 1.0
        vj[lj[k]] = 1.0

        # shift of gto to centers
        _horizontal_shift(rpi[k], li[k], vi)
        _horizontal_shift(rpj[k], lj[k], vj)

        # calc gto product
        _form_product(vi, vj, li[k], lj[k], vv)


        for l in range(li[k] + lj[k] + 1):
            v1d[k, 0] += s1d[l] * vv[l]
            v1d[k, 1] += (s1d[l+1] + rpi[k]*s1d[l]) * vv[l]
            v1d[k, 2] += (s1d[l+2] + 2*rpi[k]*s1d[l+1] + rpi[k]*rpi[k]*s1d[l]) * vv[l]
            # TODO: check l+1 indices
    
    s3d = v1d[0, 0] * v1d[1, 0] * v1d[2, 0]
    d3d[0] = v1d[0, 0] * v1d[0, 0] * v1d[2, 0]
    d3d[1] = v1d[0, 0] * v1d[1, 1] * v1d[2, 0]
    d3d[2] = v1d[0, 0] * v1d[1, 0] * v1d[2, 1]

    q3d[0] = v1d[0, 2] * v1d[1, 0] * v1d[2, 0]
    q3d[1] = v1d[0, 1] * v1d[1, 1] * v1d[2, 0]
    q3d[2] = v1d[0, 0] * v1d[1, 2] * v1d[2, 0]
    q3d[3] = v1d[0, 1] * v1d[1, 0] * v1d[2, 1]
    q3d[4] = v1d[0, 0] * v1d[1, 1] * v1d[2, 1]
    q3d[5] = v1d[0, 0] * v1d[1, 0] * v1d[2, 2]

    return s3d, d3d, q3d
