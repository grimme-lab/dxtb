# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

__all__ = ["TRAFO", "NLM_CART"]


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
    torch.tensor(
        [
            [4, 0, 0],  # gxxxx
            [0, 4, 0],  # gyyyy
            [0, 0, 4],  # gzzzz
            [3, 1, 0],  # gxxxy
            [3, 0, 1],  # gxxxz
            [1, 3, 0],  # gxyyy
            [0, 3, 1],  # gyyyz
            [1, 0, 3],  # gxzzz
            [0, 1, 3],  # gyzzz
            [2, 2, 0],  # gxxyy
            [2, 0, 2],  # gxxzz
            [0, 2, 2],  # gyyzz
            [2, 1, 1],  # gxxyz
            [1, 2, 1],  # gxyyz
            [1, 1, 2],  # gxyzz
        ]
    ),
)
"""Cartesian components of Gaussian orbitals."""

# Cartesion components taken from tblite.
#
# integer, parameter :: lx(3, 84) = reshape([&
#     & 0, & ! s
#     & 0,0,1, & ! p
#     & 2,0,0,1,1,0, & ! d
#     & 3,0,0,2,2,1,0,1,0,1, & ! f
#     & 4,0,0,3,3,1,0,1,0,2,2,0,2,1,1, & ! g
#     & 5,0,0,3,3,2,2,0,0,4,4,1,0,0,1,1,3,1,2,2,1, & ! h
#     & 6,0,0,3,3,0,5,5,1,0,0,1,4,4,2,0,2,0,3,3,1,2,2,1,4,1,1,2, & ! i
#     & 0, &
#     & 1,0,0, &
#     & 0,2,0,1,0,1, &
#     & 0,3,0,1,0,2,2,0,1,1, &
#     & 0,4,0,1,0,3,3,0,1,2,0,2,1,2,1, &
#     & 0,5,0,2,0,3,0,3,2,1,0,4,4,1,0,1,1,3,2,1,2, &
#     & 0,6,0,3,0,3,1,0,0,1,5,5,2,0,0,2,4,4,2,1,3,1,3,2,1,4,1,2, &
#     & 0, &
#     & 0,1,0, &
#     & 0,0,2,0,1,1, &
#     & 0,0,3,0,1,0,1,2,2,1, &
#     & 0,0,4,0,1,0,1,3,3,0,2,2,1,1,2, &
#     & 0,0,5,0,2,0,3,2,3,0,1,0,1,4,4,3,1,1,1,2,2, &
#     & 0,0,6,0,3,3,0,1,5,5,1,0,0,2,4,4,0,2,1,2,2,3,1,3,1,1,4,2], &
#     & shape(lx), order=[2, 1])
#
# For the ordering here, take all rows for one angular momentum from `lx` and
# stack them. Then use the columns for the Python ordering. For the p-orbitals,
# this would look the following way:
#
# Fortran  ->   Python
# 0, 0, 1     [[0, 1, 0]
# 1, 0, 0      [0, 0, 1]
# 0, 1, 0      [1, 0, 0]]
