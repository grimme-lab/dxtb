#!/usr/bin/python


import math
import torch


from ..integral.trafo import transform0, transform1
from ..constants import FLOAT64 as DTYPE_FLOAT
from ..constants import UINT8 as DTYPE_INT


# TODO: define interface (or just ducktyping)
"""interface get_overlap
    module procedure :: get_overlap_lat
end interface get_overlap"""

maxl = 6
maxl2 = 2 * maxl
msao = [1, 3, 5, 7, 9, 11, 13]
mlao = [1, 3, 6, 10, 15, 21, 28]
lmap = [0, 1, 4, 10, 20, 35, 56]
assert all([len(l) == maxl + 1 for l in [msao, mlao, lmap]])

sqrtpi = math.sqrt(math.pi)
sqrtpi3 = sqrtpi**3

# fmt: off
lx = torch.tensor([
    0,
    1,0,0,
    2,0,0,1,1,0,
    3,0,0,2,2,1,0,1,0,1,
    4,0,0,3,3,1,0,1,0,2,2,0,2,1,1,
    5,0,0,3,3,2,2,0,0,4,4,1,0,0,1,1,3,1,2,2,1,
    6,0,0,3,3,0,5,5,1,0,0,1,4,4,2,0,2,0,3,3,1,2,2,1,4,1,1,2,
    0,
    0,1,0,
    0,2,0,1,0,1,
    0,3,0,1,0,2,2,0,1,1,
    0,4,0,1,0,3,3,0,1,2,0,2,1,2,1,
    0,5,0,2,0,3,0,3,2,1,0,4,4,1,0,1,1,3,2,1,2,
    0,6,0,3,0,3,1,0,0,1,5,5,2,0,0,2,4,4,2,1,3,1,3,2,1,4,1,2,
    0,
    0,0,1,
    0,0,2,0,1,1,
    0,0,3,0,1,0,1,2,2,1,
    0,0,4,0,1,0,1,3,3,0,2,2,1,1,2,
    0,0,5,0,2,0,3,2,3,0,1,0,1,4,4,3,1,1,1,2,2,
    0,0,6,0,3,3,0,1,5,5,1,0,0,2,4,4,0,2,1,2,2,3,1,3,1,1,4,2
    ], dtype=DTYPE_INT).reshape(3, 84)
# fmt: on


def _overlap_1d(moment: int, alpha: float):
    """Calculate one-dimensional overlap

    Args:
        moment (int): Angular momentum
        alpha (float): Exponent of gaussian

    Returns:
        float: Overlap in one dimension
    """

    # see OEIS A001147
    dfactorial = [
        1.0,
        1.0,
        3.0,
        15.0,
        105.0,
        945.0,
        10395.0,
        135135.0,
    ]

    if moment % 2 == 0:
        overlap = (0.5 / alpha) ** (moment / 2) * dfactorial[int(moment / 2)]
    else:
        overlap = 0.0

    return overlap


def _horizontal_shift(ae: float, l: int, cfs):
    """Shift center of gaussian function

    Args:
        ae (float): [description]
        l (int): [description]
        cfs ([type]): [description]
    """  # TODO: add docstring

    if l == 0:  # s
        pass
    elif l == 1:  # p
        cfs[0] += ae * cfs[1]
    elif l == 2:  # d
        cfs[0] += ae * ae * cfs[2]
        cfs[1] += 2 * ae * cfs[2]
    elif l == 3:  # f
        cfs[0] += ae * ae * ae * cfs[3]
        cfs[1] += 3 * ae * ae * cfs[3]
        cfs[2] += 3 * ae * cfs[3]
    elif l == 4:  # g
        cfs[0] += ae * ae * ae * ae * cfs[4]
        cfs[1] += 4 * ae * ae * ae * cfs[4]
        cfs[2] += 6 * ae * ae * cfs[4]
        cfs[3] += 4 * ae * cfs[4]

    return


def _form_product(a, b, la: int, lb: int, d):
    """Form product of two gto

    Args:
        a ([type]): [description]
        b ([type]): [description]
        la (int): [description]
        lb (int): [description]
        d ([type]): [description]
    """  # TODO: add docstring

    # real(wp), intent(in) :: a(*), b(*)
    # real(wp), intent(inout) :: d(*)
    if (la >= 4) or (lb >= 4):
        # <s|g> = <s|*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g>
        d[0] = a[0] * b[0]
        d[1] = a[0] * b[1] + a[1] * b[0]
        d[2] = a[0] * b[2] + a[2] * b[0]
        d[3] = a[0] * b[3] + a[3] * b[0]
        d[4] = a[0] * b[4] + a[4] * b[0]
        if (la == 0) or (lb == 0):
            return
        # <p|g> = (<s|+<p|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h>
        d[2] = d[2] + a[1] * b[1]
        d[3] = d[3] + a[1] * b[2] + a[2] * b[1]
        d[4] = d[4] + a[1] * b[3] + a[3] * b[1]
        d[5] = a[1] * b[4] + a[4] * b[1]
        if (la <= 1) or (lb <= 1):
            return
        # <d|g> = (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i>
        d[4] = d[4] + a[2] * b[2]
        d[5] = d[4] + a[2] * b[3] + a[3] * b[2]
        d[6] = a[2] * b[4] + a[4] * b[2]
        if (la <= 2) or (lb <= 2):
            return
        # <f|g> = (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k>
        d[6] = d[6] + a[3] * b[3]
        d[7] = a[3] * b[4] + a[4] * b[3]
        if (la <= 3) or (lb <= 3):
            return
        # <g|g> = (<s|+<p|+<d|+<f|+<g|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k> + <l>
        d[8] = a[4] * b[4]
    elif (la >= 3) or (lb >= 3):
        # <s|f> = <s|*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f>
        d[0] = a[0] * b[0]
        d[1] = a[0] * b[1] + a[1] * b[0]
        d[2] = a[0] * b[2] + a[2] * b[0]
        d[3] = a[0] * b[3] + a[3] * b[0]
        if (la == 0) or (lb == 0):
            return
        # <p|f> = (<s|+<p|)*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f> + <g>
        d[2] = d[2] + a[1] * b[1]
        d[3] = d[3] + a[1] * b[2] + a[2] * b[1]
        d[4] = a[1] * b[3] + a[3] * b[1]
        if (la <= 1) or (lb <= 1):
            return
        # <d|f> = (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f> + <g> + <h>
        d[4] = d[4] + a[2] * b[2]
        d[5] = a[2] * b[3] + a[3] * b[2]
        if (la <= 2) or (lb <= 2):
            return
        # <f|f> = (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i>
        d[6] = a[3] * b[3]
    elif (la >= 2) or (lb >= 2):
        # <s|d> = <s|*(|s>+|p>+|d>)
        #       = <s> + <p> + <d>
        d[0] = a[0] * b[0]
        d[1] = a[0] * b[1] + a[1] * b[0]
        d[2] = a[0] * b[2] + a[2] * b[0]
        if (la == 0) or (lb == 0):
            return
        # <p|d> = (<s|+<p|)*(|s>+|p>+|d>)
        #       = <s> + <p> + <d> + <f>
        d[2] = d[2] + a[1] * b[1]
        d[3] = a[1] * b[2] + a[2] * b[1]
        if (la <= 1) or (lb <= 1):
            return
        # <d|d> = (<s|+<p|+<d|)*(|s>+|p>+|d>)
        #       = <s> + <p> + <d> + <f> + <g>
        d[4] = a[2] * b[2]
    else:
        # <s|s> = <s>
        d[0] = a[0] * b[0]
        if (la == 0) and (lb == 0):
            return
        # <s|p> = <s|*(|s>+|p>)
        #       = <s> + <p>
        d[1] = a[0] * b[1] + a[1] * b[0]
        if (la == 0) or (lb == 0):
            return
        # <p|p> = (<s|+<p|)*(|s>+|p>)
        #       = <s> + <p> + <d>
        d[2] = a[1] * b[1]
    return


def _overlap_3d(rpj, rpi, lj, li, s1d):
    """Calculate three-dimensional overlap

    Args:
        rpj (torch.tensor): Scaled distance vector for gaussian i
        rpi (torch.tensor): Scaled distance vector for gaussian i
        lj (torch.tensor): [description]
        li (torch.tensor): [description]
        s1d (torch.tensor): One-dimensional overlap contributions

    Returns:
        float: Overlap in three dimensions
    """  # TODO: add docstring

    v1d = [0.0 for _ in range(3)]  # TODO: change to tensor?

    for k in range(3):
        vi = [0.0 for _ in range(maxl + 1)]  # TODO: change to tensor?
        vj = [0.0 for _ in range(maxl + 1)]
        vv = [0.0 for _ in range(maxl2 + 1)]  # combined gto
        vi[li[k]] = 1.0
        vj[lj[k]] = 1.0

        with torch.profiler.record_function("calc horizontal_shift"):
            # shift of gto to centers
            _horizontal_shift(rpi[k], li[k], vi)
            _horizontal_shift(rpj[k], lj[k], vj)

        with torch.profiler.record_function("calc form_product"):
            # calc gto product
            _form_product(vi, vj, li[k], lj[k], vv)
            # TODO: vv = _form_product(vi, vj, li[k], lj[k]) --> change return value inside

        # sum over momenta of product gto
        for l in range(li[k] + lj[k] + 1):
            v1d[k] += s1d[l] * vv[l]

    s3d = v1d[0] * v1d[1] * v1d[2]

    return s3d


def overlap_cgto(cgtoj, cgtoi, r2, vec, intcut):
    """Calculate overlap integral for two cgto representations

    Args:
        cgtoj (cgto_type): Description of contracted gaussian function on center i
        cgtoi (cgto_type): Description of contracted gaussian function on center j
        r2 (float): Square distance between center i and j
        vec (torch.tensor): Distance vector between center i and j, ri - rj
        intcut (float): Maximum value of integral prefactor to consider

    Returns:
        torch.tensor: Overlap integrals for the given pair i  and j (overlap(msao(cgtoj.ang), msao(cgtoi.ang))
    """

    s1d = torch.zeros((maxl2), dtype=DTYPE_FLOAT)
    s3d = torch.zeros((mlao[cgtoi.ang], mlao[cgtoj.ang]), dtype=DTYPE_FLOAT)

    # with torch.profiler.record_function("calc overlap"):
    if True:
        # all contraction combinations
        for ip in range(cgtoi.nprim):
            for jp in range(cgtoj.nprim):

                eab = cgtoi.alpha[ip] + cgtoj.alpha[jp]
                oab = 1.0 / eab
                est = cgtoi.alpha[ip] * cgtoj.alpha[jp] * r2 * oab
                if est > intcut:
                    continue

                pre = math.exp(-est) * sqrtpi3 * math.sqrt(oab) ** 3
                rpi = -vec * cgtoj.alpha[jp] * oab
                rpj = +vec * cgtoi.alpha[ip] * oab
                assert len(rpi) == 3
                assert len(rpj) == 3

                with torch.profiler.record_function("calc 1D"):
                    # calc pairwise 1D overlap
                    for l in range(cgtoi.ang + cgtoj.ang + 1):
                        s1d[l] = _overlap_1d(l, eab)

                # scaling constant
                cc = cgtoi.coeff[ip] * cgtoj.coeff[jp] * pre

                # with torch.profiler.record_function("calc 3D"):
                if True:
                    # calc pairwise 3D overlap
                    for mli in range(mlao[cgtoi.ang]):
                        for mlj in range(mlao[cgtoj.ang]):
                            val = _overlap_3d(
                                rpj,
                                rpi,
                                lx[:, mlj + lmap[cgtoj.ang]],
                                lx[:, mli + lmap[cgtoi.ang]],
                                s1d,
                            )
                            s3d[mli, mlj] += cc * val

    with torch.profiler.record_function("transform overlap"):
        # transform overlap matrix
        overlap = transform0(cgtoj.ang, cgtoi.ang, s3d)

    return overlap


def overlap_grad_cgto(cgtoj, cgtoi, r2, vec, intcut, overlap, doverlap):
    """Calculate gradient for overlap cgto.

    Args:
        cgtoj (cgto_type): Description of contracted Gaussian function on center j
        cgtoi (cgto_type): Description of contracted Gaussian function on center i
        r2 (float): Square distance between center i and j
        vec (int[3]): Distance vector between center i and j, ri - rj
        intcut (float): Maximum value of integral prefactor to consider
        overlap (float[msao(cgtoj.ang), msao(cgtoi.ang)]): Overlap integrals for the given pair i and j
        doverlap (float[3, msao(cgtoj.ang), msao(cgtoi.ang)]): Overlap integral gradient for the given pair i and j
    """

    raise NotImplementedError

    # real(wp) :: eab, oab, est, rpi(3), rpj(3), cc, val, grad(3), pre

    s1d = torch.zeros((maxl2), dtype=DTYPE_FLOAT)
    s3d = torch.zeros((mlao[cgtoi.ang], mlao[cgtoj.ang]), dtype=DTYPE_FLOAT)
    ds3d = torch.zeros((3, mlao[cgtoi.ang], mlao[cgtoj.ang]), dtype=DTYPE_FLOAT)

    for ip in range(cgtoi.nprim):
        for jp in range(cgtoj.nprim):
            eab = cgtoi.alpha(ip) + cgtoj.alpha(jp)
            oab = 1.0 / eab
            est = cgtoi.alpha(ip) * cgtoj.alpha(jp) * r2 * oab
            if est > intcut:
                continue
            pre = math.exp(-est) * sqrtpi3 * math.sqrt(oab) ** 3
            rpi = -vec * cgtoj.alpha(jp) * oab
            rpj = +vec * cgtoi.alpha(ip) * oab

            for l in range(0, cgtoi.ang + cgtoj.ang + 1):
                s1d[l] = _overlap_1d(l, eab)

            cc = cgtoi.coeff(ip) * cgtoj.coeff(jp) * pre
            for mli in range(1, mlao(cgtoi.ang)):
                for mlj in range(1, mlao(cgtoj.ang)):
                    val, grad = _overlap_grad_3d(
                        rpj,
                        rpi,
                        cgtoj.alpha(jp),
                        cgtoi.alpha(ip),
                        lx[:, mlj + lmap(cgtoj.ang)],
                        lx[:, mli + lmap(cgtoi.ang)],
                        s1d,
                        val,
                        grad,
                    )
                    s3d[mlj, mli] += cc * val
                    ds3d[:, mlj, mli] += +cc * grad

    transform0(cgtoj.ang, cgtoi.ang, s3d, overlap)
    transform1(cgtoj.ang, cgtoi.ang, ds3d, doverlap)

    return


def get_overlap_lat(mol, trans, cutoff: float, bas, overlap):
    """!> Evaluate overlap for a molecular structure
    !> Molecular structure data
    type(structure_type), intent(in) :: mol
    !> Lattice points within a given realspace cutoff
    real(wp), intent(in) :: trans(:, :)
    !> Realspace cutoff
    real(wp), intent(in) :: cutoff
    !> Basis set information
    type(basis_type), intent(in) :: bas
    !> Overlap matrix
    real(wp), intent(out) :: overlap(:, :)"""

    raise NotImplementedError

    overlap = torch.zeros_like(overlap)
    stmp = torch.zeros((msao[bas.maxl] ** 2))
    cutoff2 = cutoff**2

    # $omp parallel do schedule(runtime) default(none) &
    # $omp shared(mol, bas, trans, cutoff2, overlap) private(r2, vec, stmp) &
    # $omp private(iat, jat, izp, jzp, itr, is, js, ish, jsh, ii, jj, iao, jao, nao)
    for iat in range(1, mol.nat):
        izp = mol.id[iat]
        iss = bas.ish_at[iat]
        for jat in range(1, mol.nat):
            jzp = mol.id[jat]
            js = bas.ish_at[jat]
            for itr in range(1, torch.size(trans, 2)):
                vec = mol.xyz[:, iat] - mol.xyz[:, jat] - trans[:, itr]
                r2 = vec[1] ** 2 + vec[2] ** 2 + vec[3] ** 2
                if r2 > cutoff2:
                    continue
                for ish in range(1, bas.nsh_id[izp]):
                    ii = bas.iao_sh[iss + ish]
                    for jsh in range(1, bas.nsh_id[jzp]):
                        jj = bas.iao_sh[js + jsh]
                        overlap_cgto(
                            bas.cgto[jsh, jzp],
                            bas.cgto[ish, izp],
                            r2,
                            vec,
                            bas.intcut,
                            stmp,
                        )

                        nao = msao(bas.cgto(jsh, jzp).ang)
                        # $omp simd collapse(2)
                        for iao in range(1, msao[bas.cgto[ish, izp].ang]):
                            for jao in range(1, nao):
                                overlap[jj + jao, ii + iao] += stmp[
                                    jao + nao * (iao - 1)
                                ]

    return overlap


def get_overlap(mol, trans, cutoff: float, bas, overlap):
    # TODO: nicer aliasing
    get_overlap_lat(mol, trans, cutoff, bas, overlap)
