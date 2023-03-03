"""
Routines for transformation from cartesian basis functions to spherical harmonics.
The convention for spherial harmonics ordering is [0, ..., l, -l].
"""
from __future__ import annotations

import math

import torch

# For Latex equations see: https://theochem.github.io/horton/2.0.1/tech_ref_gaussian_basis.html


s3 = math.sqrt(3.0)
s3_4 = s3 * 0.5

d32 = 3.0 / 2.0
s3_8 = math.sqrt(3.0 / 8.0)
s5_8 = math.sqrt(5.0 / 8.0)
s6 = math.sqrt(6.0)
s15 = math.sqrt(15.0)
s15_4 = math.sqrt(15.0 / 4.0)
s45 = math.sqrt(45.0)
s45_8 = math.sqrt(45.0 / 8.0)

d38 = 3.0 / 8.0
d34 = 3.0 / 4.0
s5_16 = math.sqrt(5.0 / 16.0)
s10 = math.sqrt(10.0)
s10_8 = math.sqrt(10.0 / 8.0)
s35_4 = math.sqrt(35.0 / 4.0)
s35_8 = math.sqrt(35.0 / 8.0)
s35_64 = math.sqrt(35.0 / 64.0)
s45_4 = math.sqrt(45.0 / 4.0)
s315_8 = math.sqrt(315.0 / 8.0)
s315_16 = math.sqrt(315.0 / 16.0)


trafo = (
    torch.tensor([[1.0]]),
    torch.tensor(
        [
            [1.0, 0.0, 0.0],  # x
            [0.0, 1.0, 0.0],  # y
            [0.0, 0.0, 1.0],  # z
        ]
    ),
    torch.tensor(  # fmt: off
        [  #    0    1   -1      2   -2
            [-0.5, 0.0, 0.0, s3_4, 0.0],  # xx
            [0.0, 0.0, 0.0, 0.0, s3],  # xy
            [0.0, s3, 0.0, 0.0, 0.0],  # xz
            [-0.5, 0.0, 0.0, -s3_4, 0.0],  # yy
            [0.0, 0.0, s3, 0.0, 0.0],  # yz
            [1.0, 0.0, 0.0, 0.0, 0.0],  # zz
        ]
    ),  # fmt: on
    torch.tensor(  # fmt: off
        [  #     0      1     -1       2   -2       3     -3
            [0.0, -s3_8, 0.0, 0.0, 0.0, s5_8, 0.0],  # xxx
            [0.0, 0.0, -s3_8, 0.0, 0.0, 0.0, s45_8],  # xxy
            [-d32, 0.0, 0.0, s15_4, 0.0, 0.0, 0.0],  # xxz
            [0.0, -s3_8, 0.0, 0.0, 0.0, -s45_8, 0.0],  # xyy
            [0.0, 0.0, 0.0, 0.0, s15, 0.0, 0.0],  # xyz
            [0.0, s6, 0.0, 0.0, 0.0, 0.0, 0.0],  # xzz
            [0.0, 0.0, -s3_8, 0.0, 0.0, 0.0, -s5_8],  # yyy
            [-d32, 0.0, 0.0, -s15_4, 0.0, 0.0, 0.0],  # yyz
            [0.0, 0.0, s6, 0.0, 0.0, 0.0, 0.0],  # yzz
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # zzz
        ]
    ),  # fmt: on
    torch.tensor(  # fmt: off
        [  #    0       1      -1       2      -2        3      -3         4      -4
            [d38, 0.0, 0.0, -s5_16, 0.0, 0.0, 0.0, s35_64, 0.0],  # xxxx
            [0.0, 0.0, 0.0, 0.0, -s10_8, 0.0, 0.0, 0.0, s35_4],  # xxxy
            [0.0, -s45_8, 0.0, 0.0, 0.0, s35_8, 0.0, 0.0, 0.0],  # xxxz
            [d34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -s315_16, 0.0],  # xxyy
            [0.0, 0.0, -s45_8, 0.0, 0.0, 0.0, s315_8, 0.0, 0.0],  # xxyz
            [-3.0, 0.0, 0.0, s45_4, 0.0, 0.0, 0.0, 0.0, 0.0],  # xxzz
            [0.0, 0.0, 0.0, 0.0, -s10_8, 0.0, 0.0, 0.0, -s35_4],  # xyyy
            [0.0, -s45_8, 0.0, 0.0, 0.0, -s315_8, 0.0, 0.0, 0.0],  # xyyz
            [0.0, 0.0, 0.0, 0.0, s45, 0.0, 0.0, 0.0, 0.0],  # xyzz
            [0.0, s10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # xzzz
            [d38, 0.0, 0.0, s5_16, 0.0, 0.0, 0.0, s35_64, 0.0],  # yyyy
            [0.0, 0.0, -s45_8, 0.0, 0.0, 0.0, -s35_8, 0.0, 0.0],  # yyyz
            [-3.0, 0.0, 0.0, -s45_4, 0.0, 0.0, 0.0, 0.0, 0.0],  # yyzz
            [0.0, 0.0, s10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # yzzz
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # zzzz
        ]
    ),  # fmt: on
)
