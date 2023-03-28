"""
Overlap Transformation
======================

This module contains the transformation matrices for cartesian basis functions
to spherical harmonics as well as the cartesian orbital ordering used in the
overlap integral.


https://theochem.github.io/horton/2.0.1/tech_ref_gaussian_basis.html
"""

from __future__ import annotations

from math import sqrt

import torch

s3 = sqrt(3.0)
s3_4 = s3 * 0.5

d32 = 3.0 / 2.0
s3_8 = sqrt(3.0 / 8.0)
s5_8 = sqrt(5.0 / 8.0)
s6 = sqrt(6.0)
s15 = sqrt(15.0)
s15_4 = sqrt(15.0 / 4.0)
s45 = sqrt(45.0)
s45_8 = sqrt(45.0 / 8.0)

d38 = 3.0 / 8.0
d34 = 3.0 / 4.0
s5_16 = sqrt(5.0 / 16.0)
s10 = sqrt(10.0)
s10_8 = sqrt(10.0 / 8.0)
s35_4 = sqrt(35.0 / 4.0)
s35_8 = sqrt(35.0 / 8.0)
s35_64 = sqrt(35.0 / 64.0)
s45_4 = sqrt(45.0 / 4.0)
s315_8 = sqrt(315.0 / 8.0)
s315_16 = sqrt(315.0 / 16.0)


TRAFO = (
    torch.tensor([[1.0]]),
    torch.tensor(
        [
            [1.0, 0.0, 0.0],  # x
            [0.0, 1.0, 0.0],  # y
            [0.0, 0.0, 1.0],  # z
        ]
    ),
    # fmt: off
    torch.tensor([
        [-0.5,  -0.5, 1.0, 0.0, 0.0, 0.0],
        [ 0.0,   0.0, 0.0, 0.0,  s3, 0.0],
        [ 0.0,   0.0, 0.0, 0.0, 0.0,  s3],
        [s3_4, -s3_4, 0.0, 0.0, 0.0, 0.0],
        [ 0.0,   0.0, 0.0,  s3, 0.0, 0.0],
    ]),
    # FIXME: This transformation matrix is the possibly wrong as it uses the
    # [-l, ..., 0, ..., l] ordering (copied from tblite) and not the ordering
    # that the d-orbitals use ([0, ..., l, -l]). However, I do not have a
    # suitable reference yet and we do not need f-orbitals currently...
    torch.tensor([
        [  0.0, -s5_8, 0.0, s45_8,   0.0,    0.0,    0.0, 0.0, 0.0, 0.0],
        [  0.0,   0.0, 0.0,   0.0,   0.0,    0.0,    0.0, 0.0, 0.0, s15],
        [  0.0, -s3_8, 0.0, -s3_8,   0.0,    0.0,    0.0, 0.0,  s6, 0.0],
        [  0.0,   0.0, 1.0,   0.0,  -d32,    0.0,   -d32, 0.0, 0.0, 0.0],
        [-s3_8,   0.0, 0.0,   0.0,   0.0,  -s3_8,    0.0,  s6, 0.0, 0.0],
        [  0.0,   0.0, 0.0,   0.0, s15_4,    0.0, -s15_4, 0.0, 0.0, 0.0],
        [ s5_8,   0.0, 0.0,   0.0,   0.0, -s45_8,    0.0, 0.0, 0.0, 0.0],
    ]),
    # fmt: on
)
"""
Transformation from cartesian basis functions to spherical harmonics.
The convention for spherial harmonics ordering is [0, ..., l, -l].
"""

NLM_CART = (
    torch.tensor(
        [
            [0, 0, 0],  # s
        ]
    ),
    torch.tensor(
        [
            # tblite order: x (+1), y (-1), z (0) in [-1, 0, 1] sorting
            [0, 1, 0],  # py
            [0, 0, 1],  # pz
            [1, 0, 0],  # px
        ]
    ),
    torch.tensor(
        [
            [2, 0, 0],  # dxx
            [0, 2, 0],  # dyy
            [0, 0, 2],  # dzz
            [1, 1, 0],  # dxy
            [1, 0, 1],  # dxz
            [0, 1, 1],  # dyz
        ]
    ),
    torch.tensor(
        [
            [3, 0, 0],  # fxxx
            [0, 3, 0],  # fyyy
            [0, 0, 3],  # fzzz
            [2, 1, 0],  # fxxy
            [2, 0, 1],  # fxxz
            [1, 2, 0],  # fxyy
            [0, 2, 1],  # fyyz
            [1, 0, 2],  # fxzz
            [0, 1, 2],  # fyzz
            [1, 1, 1],  # fxyz
        ]
    ),
    # FIXME: RE-SORT (should be similar to d and f)
    torch.tensor(
        [
            [4, 0, 0],  # gxxxx
            [3, 1, 0],  # gxxxy
            [3, 0, 1],  # gxxxz
            [2, 2, 0],  # gxxyy
            [2, 1, 1],  # gxxyz
            [2, 0, 2],  # gxxzz
            [1, 3, 0],  # gxyyy
            [1, 2, 1],  # gxyyz
            [1, 1, 2],  # gxyzz
            [1, 0, 3],  # gxzzz
            [0, 4, 0],  # gyyyy
            [0, 3, 1],  # gyyyz
            [0, 2, 2],  # gyyzz
            [0, 1, 3],  # gyzzz
            [0, 0, 4],  # gzzzz
        ]
    ),
)
"""Cartesian components of Gaussian orbitals."""

# Transformation matrices taken from tblite and reshaped accordingly.
#
# dtrafo
# reshape_fortran(torch.tensor(
#     [ #   0    1   -1     2     -2
#         -0.5, 0.0, 0.0,  s3_4, 0.0,  # xx
#         -0.5, 0.0, 0.0, -s3_4, 0.0,  # yy
#          1.0, 0.0, 0.0,   0.0, 0.0,  # zz
#          0.0, 0.0, 0.0,   0.0,  s3,  # xy
#          0.0,  s3, 0.0,   0.0, 0.0,  # xz
#          0.0, 0.0,  s3,   0.0, 0.0,  # yz
#     ]
# ), (5, 6))
#
#
# THE TRAFO FOR F-ORBITALS IS VERY LIKELY WRONG!
# The explicit transformation in tblite differs from the transformation with
# the transformation matrix that uses [-l, ..., 0, ..., l] ordering. Changing
# the ordering to [0, ..., l-1, -(l-1), l, -l] reproduces the explicit
# transformation. Since we do not use f-orbitals I did not fix this yet
#
# ftrafo
# reshape_fortran(torch.tensor(
#     [  #  -3    -2     -1     0      1       2       3
#           0.0,  0.0,   0.0,  0.0, -s3_8,    0.0,   s5_8, # xxx
#         -s5_8,  0.0, -s3_8,  0.0,   0.0,    0.0,    0.0, # yyy
#           0.0,  0.0,   0.0,  1.0,   0.0,    0.0,    0.0, # zzz
#         s45_8,  0.0, -s3_8,  0.0,   0.0,    0.0,    0.0, # xxy
#           0.0,  0.0,   0.0, -d32,   0.0,  s15_4,    0.0, # xxz
#           0.0,  0.0,   0.0,  0.0, -s3_8,    0.0, -s45_8, # xyy
#           0.0,  0.0,   0.0, -d32,   0.0, -s15_4,    0.0, # yyz
#           0.0,  0.0,   0.0,  0.0,    s6,    0.0,    0.0, # xzz
#           0.0,  0.0,    s6,  0.0,   0.0,    0.0,    0.0, # yzz
#           0.0,  s15,   0.0,  0.0,   0.0,    0.0,    0.0, # xyz
#     ]
# ), (7, 10))
