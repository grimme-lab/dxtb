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

import numpy as np

__all__ = ["TRAFO", "NLM_CART"]

# DEVNOTE: The tables are numpy arrays on purpose. Module-level tensors would
# be created whenever the (lazily imported) driver is first used, which looks
# like a memory leak of the first calculation, and they would be fixed to one
# device and dtype.


s3 = sqrt(3.0)
s3_4 = s3 * 0.5

d32 = 3.0 / 2.0
s3_8 = sqrt(3.0 / 8.0)
s5_8 = sqrt(5.0 / 8.0)
s6 = sqrt(6.0)
s15 = sqrt(15.0)
s15_4 = sqrt(15.0 / 4.0)
# s45 = sqrt(45.0)
s45_8 = sqrt(45.0 / 8.0)

# d38 = 3.0 / 8.0
# d34 = 3.0 / 4.0
# s5_16 = sqrt(5.0 / 16.0)
# s10 = sqrt(10.0)
# s10_8 = sqrt(10.0 / 8.0)
# s35_4 = sqrt(35.0 / 4.0)
# s35_8 = sqrt(35.0 / 8.0)
# s35_64 = sqrt(35.0 / 64.0)
# s45_4 = sqrt(45.0 / 4.0)
# s315_8 = sqrt(315.0 / 8.0)
# s315_16 = sqrt(315.0 / 16.0)


TRAFO = (
    np.array([[1.0]], dtype=np.float64),
    np.array(
        [
            [1.0, 0.0, 0.0],  # y
            [0.0, 1.0, 0.0],  # z
            [0.0, 0.0, 1.0],  # x
        ],
        dtype=np.float64,
    ),
    # fmt: off
    # Cartesian columns: xx, xy, xz, yy, yz, zz; rows: m = -2, ..., 2
    np.array([
        [ 0.0,  s3, 0.0,   0.0, 0.0, 0.0],  # m = -2
        [ 0.0, 0.0, 0.0,   0.0,  s3, 0.0],  # m = -1
        [-0.5, 0.0, 0.0,  -0.5, 0.0, 1.0],  # m =  0
        [ 0.0, 0.0,  s3,   0.0, 0.0, 0.0],  # m = +1
        [s3_4, 0.0, 0.0, -s3_4, 0.0, 0.0],  # m = +2
    ], dtype=np.float64),
    # Cartesian columns: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
    # rows: m = -3, ..., 3
    np.array([
        [  0.0, s45_8,    0.0,    0.0, 0.0, 0.0, -s5_8,    0.0, 0.0, 0.0],  # -3
        [  0.0,   0.0,    0.0,    0.0, s15, 0.0,   0.0,    0.0, 0.0, 0.0],  # -2
        [  0.0, -s3_8,    0.0,    0.0, 0.0, 0.0, -s3_8,    0.0,  s6, 0.0],  # -1
        [  0.0,   0.0,   -d32,    0.0, 0.0, 0.0,   0.0,   -d32, 0.0, 1.0],  #  0
        [-s3_8,   0.0,    0.0,  -s3_8, 0.0,  s6,   0.0,    0.0, 0.0, 0.0],  # +1
        [  0.0,   0.0,  s15_4,    0.0, 0.0, 0.0,   0.0, -s15_4, 0.0, 0.0],  # +2
        [ s5_8,   0.0,    0.0, -s45_8, 0.0, 0.0,   0.0,    0.0, 0.0, 0.0],  # +3
    ], dtype=np.float64),
    # fmt: on
)
"""
Transformation from cartesian basis functions to spherical harmonics.

Follows the CCA ordering of tblite (tblite/tblite#371): cartesian functions
are in lexicographic order (xx, xy, xz, yy, yz, zz, ...) and spherical
harmonics in ascending order of m, i.e., [-l, ..., 0, ..., l]. The p-orbitals
are the exception (identity transformation): they are in the order y, z, x,
which is the spherical order [-1, 0, 1] of tblite.
"""


# The d and f matrices above are `dtrafo` and `ftrafo` of
# `src/tblite/integral/trafo.f90` in tblite (after tblite/tblite#371), both of
# shape (2l+1, ncart). The Fortran source lists them column by column, i.e.,
# one cartesian function per line.

NLM_CART = (
    np.array(
        [
            [0, 0, 0],  # s
        ]
    ),
    np.array(
        [
            # tblite order: y (-1), z (0), x (+1) in [-1, 0, 1] sorting
            [0, 1, 0],  # py
            [0, 0, 1],  # pz
            [1, 0, 0],  # px
        ]
    ),
    np.array(
        [
            [2, 0, 0],  # dxx
            [1, 1, 0],  # dxy
            [1, 0, 1],  # dxz
            [0, 2, 0],  # dyy
            [0, 1, 1],  # dyz
            [0, 0, 2],  # dzz
        ]
    ),
    np.array(
        [
            [3, 0, 0],  # fxxx
            [2, 1, 0],  # fxxy
            [2, 0, 1],  # fxxz
            [1, 2, 0],  # fxyy
            [1, 1, 1],  # fxyz
            [1, 0, 2],  # fxzz
            [0, 3, 0],  # fyyy
            [0, 2, 1],  # fyyz
            [0, 1, 2],  # fyzz
            [0, 0, 3],  # fzzz
        ]
    ),
    np.array(
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

# Cartesian components taken from `lx` in
# `src/tblite/integral/native/integrals.f90` of tblite (CCA ordering, after
# tblite/tblite#371): for each l, components are ordered by decreasing lx and,
# for equal lx, by decreasing ly (lz = l - lx - ly). This is the ordering used
# for d, f and g above. For p, tblite also lists x, y, z, but its transform
# then permutes the result with [2, 3, 1] to y, z, x, which is the ordering
# used here.
