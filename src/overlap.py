#!/usr/bin/python


import math
import numpy as np


from trafo import transform0, transform1
"""use mctc_env, only : wp
use mctc_io, only : structure_type
use mctc_io_constants, only : pi
use tblite_basis_type, only : basis_type, cgto_type"""

# TODO: rewrite into pytorch! (not only numpy)

# public :: overlap_cgto, overlap_grad_cgto
# public :: get_overlap
# public :: maxl, msao

"""interface get_overlap
    module procedure :: get_overlap_lat
end interface get_overlap""" #TODO: maybe just ducktyping

maxl = 6
maxl2 = 2*maxl
msao = [1, 3, 5, 7, 9, 11, 13]
mlao = [1, 3, 6, 10, 15, 21, 28]
lmap = [0, 1, 4, 10, 20, 35, 56]
assert all([len(l) == maxl+1 for l in [msao, mlao, lmap]])

sqrtpi = math.sqrt(math.pi)
sqrtpi3 = sqrtpi**3

lx = np.array([
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
    ]).reshape(3, 84)

def _overlap_1d(moment: int, alpha: float):
    """ Calculate one-dimensional overlap

    Args:
        moment (int): Angular momentum
        alpha (float): Exponent of gaussian

    Returns:
        float: Overlap in one dimension
    """

    dfactorial = [1.,1.,3.,15.,105.,945.,10395.,135135.] # see OEIS A001147
    if moment % 2 == 0 :
        overlap = (0.5/alpha)**(moment/2) * dfactorial[int(moment/2)]
    else:
        overlap = 0.0

    return overlap

def _horizontal_shift(ae: float, l: int, cfs):
    """ Shift center of gaussian function

    Args:
        ae (float): [description]
        l (int): [description]
        cfs ([type]): [description]
    """ # TODO: add docstring

    if l == 0: #s
        pass
    elif l == 1: #p
        cfs[0] += ae*cfs[1]
    elif l == 2: #d
        cfs[0] += ae*ae*cfs[2]
        cfs[1] +=  2*ae*cfs[2]
    elif l == 3: #f
        cfs[0] += ae*ae*ae*cfs[3]
        cfs[1] +=  3*ae*ae*cfs[3]
        cfs[2] +=  3*ae*cfs[3]
    elif l == 4: #g
        cfs[0] += ae*ae*ae*ae*cfs[4]
        cfs[1] +=  4*ae*ae*ae*cfs[4]
        cfs[2] +=  6*ae*ae*cfs[4]
        cfs[3] +=  4*ae*cfs[4]

    return

def _form_product(a, b, la: int, lb: int, d):
    #real(wp), intent(in) :: a(*), b(*)
    #real(wp), intent(inout) :: d(*)
    if (la >= 4) or (lb >= 4):
        # <s|g> = <s|*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g>
        d[0]=a[0]*b[0]
        d[1]=a[0]*b[1]+a[1]*b[0]
        d[2]=a[0]*b[2]+a[2]*b[0]
        d[3]=a[0]*b[3]+a[3]*b[0]
        d[4]=a[0]*b[4]+a[4]*b[0]
        if (la == 0) or (lb == 0):
            return
        # <p|g> = (<s|+<p|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h>
        d[2]=d[2]+a[1]*b[1]
        d[3]=d[3]+a[1]*b[2]+a[2]*b[1]
        d[4]=d[4]+a[1]*b[3]+a[3]*b[1]
        d[5]=a[1]*b[4]+a[4]*b[1]
        if (la <= 1) or (lb <= 1):
            return
        # <d|g> = (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i>
        d[4]=d[4]+a[2]*b[2]
        d[5]=d[4]+a[2]*b[3]+a[3]*b[2]
        d[6]=a[2]*b[4]+a[4]*b[2]
        if (la <= 2) or (lb <= 2):
            return
        # <f|g> = (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k>
        d[6]=d[6]+a[3]*b[3]
        d[7]=a[3]*b[4]+a[4]*b[3]
        if (la <= 3) or (lb <= 3):
            return
        # <g|g> = (<s|+<p|+<d|+<f|+<g|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k> + <l>
        d[8]=a[4]*b[4]
    elif (la >= 3) or (lb >= 3):
        # <s|f> = <s|*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f>
        d[0]=a[0]*b[0]
        d[1]=a[0]*b[1]+a[1]*b[0]
        d[2]=a[0]*b[2]+a[2]*b[0]
        d[3]=a[0]*b[3]+a[3]*b[0]
        if (la == 0) or (lb == 0):
            return
        # <p|f> = (<s|+<p|)*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f> + <g>
        d[2]=d[2]+a[1]*b[1]
        d[3]=d[3]+a[1]*b[2]+a[2]*b[1]
        d[4]=a[1]*b[3]+a[3]*b[1]
        if (la <= 1) or (lb <= 1):
            return
        # <d|f> = (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f> + <g> + <h>
        d[4]=d[4]+a[2]*b[2]
        d[5]=a[2]*b[3]+a[3]*b[2]
        if (la <= 2) or (lb <= 2):
            return
        # <f|f> = (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i>
        d[6]=a[3]*b[3]
    elif (la >= 2) or (lb >= 2):
        # <s|d> = <s|*(|s>+|p>+|d>)
        #       = <s> + <p> + <d>
        d[0]=a[0]*b[0]
        d[1]=a[0]*b[1]+a[1]*b[0]
        d[2]=a[0]*b[2]+a[2]*b[0]
        if (la == 0) or (lb == 0):
            return
        # <p|d> = (<s|+<p|)*(|s>+|p>+|d>)
        #       = <s> + <p> + <d> + <f>
        d[2]=d[2]+a[1]*b[1]
        d[3]=a[1]*b[2]+a[2]*b[1]
        if (la <= 1) or (lb <= 1):
            return
        # <d|d> = (<s|+<p|+<d|)*(|s>+|p>+|d>)
        #       = <s> + <p> + <d> + <f> + <g>
        d[4]=a[2]*b[2]
    else:
        print("here we are")
        print(la, lb)
        # <s|s> = <s>
        d[0]=a[0]*b[0]
        print("wrapping3")
        if (la == 0) and (lb == 0):
            return
        # <s|p> = <s|*(|s>+|p>)
        #       = <s> + <p>
        print("wrapping2")
        d[1]=a[0]*b[1]+a[1]*b[0]
        if (la == 0) or (lb == 0):
            return
        # <p|p> = (<s|+<p|)*(|s>+|p>)
        #       = <s> + <p> + <d>
        d[2]=a[1]*b[1]
        print("wrapping")
    return

def _overlap_3d(rpj, rpi, lj, li, s1d):
    """ Calculate three-dimensional overlap

    Args:
        rpj (np.array): Scaled distance vector for gaussian i
        rpi (np.array): Scaled distance vector for gaussian i
        lj (np.array): [description]
        li (np.array): [description]
        s1d (np.array): One-dimensional overlap contributions

    Returns:
        float: Overlap in three dimensions
    """   # TODO: add docstring

    v1d = [0.0 for _ in range(3)]

    for k in range(3):
        vi = [0.0 for _ in range(maxl+1)]
        vj = [0.0 for _ in range(maxl+1)]            
        vv = [0.0 for _ in range(maxl2+1)] # combined gto
        vi[li[k]] = 1.0
        vj[lj[k]] = 1.0

        # shift of gto to centers
        _horizontal_shift(rpi[k], li[k], vi)
        _horizontal_shift(rpj[k], lj[k], vj)
       
        # calc gto product
        _form_product(vi, vj, li[k], lj[k], vv)
        # TODO: vv = _form_product(vi, vj, li[k], lj[k]) --> change return value inside

        # sum over momenta of product gto
        for l in range(li[k] + lj[k] + 1):
            v1d[k] += s1d[l] * vv[l]

    s3d = v1d[0] * v1d[1] * v1d[2]
    
    return s3d

def overlap_cgto(cgtoj, cgtoi, r2, vec, intcut):
    """ Calculate overlap integral for two cgto representations

    Args:
        cgtoj (cgto_type): Description of contracted gaussian function on center i
        cgtoi (cgto_type): Description of contracted gaussian function on center j
        r2 (float): Square distance between center i and j
        vec (np.array): Distance vector between center i and j, ri - rj
        intcut (float): Maximum value of integral prefactor to consider

    Returns:
        np.array: Overlap integrals for the given pair i  and j (overlap(msao(cgtoj.ang), msao(cgtoi.ang))
    """

    s1d = np.zeros((maxl2))
    s3d = np.zeros((mlao[cgtoi.ang],mlao[cgtoj.ang]))

    # all contraction combinations
    for ip in range(cgtoi.nprim):
        for jp in range(cgtoj.nprim):

            eab = cgtoi.alpha[ip] + cgtoj.alpha[jp]
            oab = 1.0/eab
            est = cgtoi.alpha[ip] * cgtoj.alpha[jp] * r2 * oab
            if est > intcut:
                continue
        
            pre = math.exp(-est) * sqrtpi3*math.sqrt(oab)**3
            rpi = -vec * cgtoj.alpha[jp] * oab
            rpj = +vec * cgtoi.alpha[ip] * oab
            assert len(rpi) == 3
            assert len(rpj) == 3

            # calc pairwise 1D overlap
            for l in range(cgtoi.ang + cgtoj.ang + 1):
                s1d[l] = _overlap_1d(l, eab)

            # scaling constant
            cc = cgtoi.coeff[ip] * cgtoj.coeff[jp] * pre

            # calc pairwise 3D overlap
            for mli in range(mlao[cgtoi.ang]):
                for mlj in range(mlao[cgtoj.ang]):
                    val = _overlap_3d(rpj, rpi, lx[:, mlj+lmap[cgtoj.ang]], lx[:, mli+lmap[cgtoi.ang]], s1d)
                    s3d[mlj, mli] += cc*val

    # transform overlap matrix
    overlap = transform0(cgtoj.ang, cgtoi.ang, s3d)

    return overlap
