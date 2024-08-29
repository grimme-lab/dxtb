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
Reference data for property calculations.
"""

from __future__ import annotations

import torch
from tad_mctc.data.molecules import merge_nested_dicts, mols

from dxtb._src.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """Format of reference records."""

    dipole: Tensor
    """Dipole moment of molecule with (0, 0, 0) field."""

    dipole2: Tensor
    """Dipole moment of molecule with (-2, 1, 0.5) field."""

    quadrupole: Tensor
    """Quadrupole moment of molecule with (0, 0, 0) field."""

    quadrupole2: Tensor
    """Quadrupole moment of molecule with (2, 3, 5) field."""

    freqs: Tensor
    """Frequencies for GFN1-xTB."""

    ints: Tensor
    """IR intensities for GFN1-xTB."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([0.0, 0.0, 0.0]),
        "quadrupole": torch.tensor(
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        ),
        "quadrupole2": torch.tensor(
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        ),
        "freqs": torch.tensor([0.0, 0.0, 0.0]),
        "ints": torch.tensor([0.0, 0.0, 0.0]),
    },
    "H2": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([0.0000, 0.0000, 0.1015]),
        "quadrupole": torch.tensor(
            [-0.1591, 0.0000, -0.1591, 0.0000, 0.0000, 0.3183]
        ),
        "quadrupole2": torch.tensor(
            [-0.1584, 0.0000, -0.1584, 0.0000, 0.0000, 0.3168]
        ),
        "freqs": torch.tensor([5363.23]),
        "ints": torch.tensor([0.00000]),
    },
    "LiH": {
        "dipole": torch.tensor([0.0000, 0.0000, -2.4794]),
        "dipole2": torch.tensor([-1.2293, 0.3073, -1.1911]),
        "quadrupole": torch.tensor(
            [-0.6422, 0.0000, -0.6422, -0.0000, 0.0000, 1.2843]
        ),
        "quadrupole2": torch.tensor(
            [-2.0799, 0.4581, -0.3620, 3.7017, -0.9254, 2.4419]
        ),
        "freqs": torch.tensor([1024.4719]),
        "ints": torch.tensor([276.62115]),
    },
    "HHe": {
        "dipole": torch.tensor([0.0000, 0.0000, 0.2565]),
        "dipole2": torch.tensor([0.0000, 0.0000, 0.2759]),
        "quadrupole": torch.tensor(
            [-0.2259, 0.0000, -0.2259, 0.0000, 0.0000, 0.4517]
        ),
        "quadrupole2": torch.tensor(
            [-0.2465, 0.0000, -0.2465, 0.0000, 0.0000, 0.4929]
        ),
        "freqs": torch.tensor([0.50]),
        "ints": torch.tensor([0.21973]),
    },
    "H2O": {
        "dipole": torch.tensor([-0.0000, -0.0000, 1.1208]),
        "dipole2": torch.tensor([-0.1418, 0.0006, 1.1680]),
        "quadrupole": torch.tensor(
            [2.2898, 0.0000, -0.8549, 0.0000, 0.0000, -1.4349]
        ),
        "quadrupole2": torch.tensor(
            [2.3831, 0.0001, -0.8838, -0.2055, -0.0022, -1.4993]
        ),
        "freqs": torch.tensor([1480.06, 3643.97, 3747.36]),
        "ints": torch.tensor([232.80476, 45.49973, 45.50967]),
    },
    "CH4": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([-0.2230, 0.0480, 0.1661]),
        "quadrupole": torch.tensor(
            [0.0000, 0.0000, -0.0000, 0.0000, -0.0000, 0.0000]
        ),
        "quadrupole2": torch.tensor(
            [-0.0003, -0.6487, 0.0003, -0.1852, 0.8712, -0.0000]
        ),
        "freqs": torch.tensor(
            [
                1375.46,
                1375.46,
                1375.46,
                1538.18,
                1538.18,
                3005.65,
                3075.94,
                3075.94,
                3075.94,
            ]
        ),
        "ints": torch.tensor(
            [
                31.73685,
                31.73685,
                31.73685,
                0.00000,
                0.00000,
                0.00000,
                12.29465,
                12.29465,
                12.29465,
            ]
        ),
    },
    "SiH4": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([-1.2136, 0.2518, 0.8971]),
        "quadrupole": torch.tensor(
            [-0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000]
        ),
        "quadrupole2": torch.tensor(
            [0.1276, -0.5236, -0.1362, -0.1603, 0.6765, 0.0085]
        ),
        "freqs": torch.tensor(
            [
                835.82,
                835.82,
                835.82,
                889.87,
                889.87,
                2036.71,
                2051.72,
                2051.72,
                2051.72,
            ]
        ),
        "ints": torch.tensor(
            [
                631.1669,
                631.1669,
                631.1669,
                0.0000,
                0.0000,
                0.0000,
                354.8140,
                354.8140,
                354.8140,
            ]
        ),
    },
    "PbH4-BiH3": {
        "dipole": torch.tensor([-0.0000, -1.0555, 0.0000]),
        "dipole2": torch.tensor([-1.7609, -0.6177, 1.3402]),
        "quadrupole": torch.tensor(
            [5.3145, -0.0000, -10.6290, -0.0000, -0.0000, 5.3145]
        ),
        "quadrupole2": torch.tensor(
            [4.8044, -0.9509, -10.4691, -0.7061, 0.2940, 5.6647]
        ),
        "freqs": torch.tensor(
            [
                -96.79,
                -96.77,
                14.72,
                127.19,
                179.37,
                179.39,
                696.64,
                698.33,
                698.41,
                791.06,
                791.06,
                853.36,
                853.36,
                864.43,
                1731.85,
                1765.54,
                1798.91,
                1798.92,
                1886.47,
                1891.21,
                1891.32,
            ]
        ),
        "ints": torch.tensor(
            [
                47.12630,
                47.14283,
                0.01511,
                0.00010,
                0.74027,
                0.72463,
                345.07755,
                329.54513,
                329.52192,
                0.02299,
                0.02193,
                33.39170,
                33.40284,
                162.42625,
                632.03366,
                0.03999,
                782.65258,
                782.70185,
                654.80401,
                689.61196,
                688.24284,
            ]
        ),
    },
    "MB16_43_01": {
        "dipole": torch.tensor([0.2903, -1.0541, -2.0211]),
        "dipole2": torch.tensor([-5.3891, 1.2219, 2.6970]),
        "quadrupole": torch.tensor(
            [6.0210, 26.8833, -4.7743, 24.5314, -35.8644, -1.2467]
        ),
        "quadrupole2": torch.tensor(
            [7.8772, 22.4950, -0.3045, 26.1071, -34.4454, -7.5727],
            dtype=torch.float64,
        ),
        "freqs": torch.tensor(
            [
                -38.81,
                54.56,
                65.85,
                87.71,
                101.98,
                114.75,
                122.62,
                155.40,
                167.31,
                181.92,
                187.55,
                218.00,
                250.48,
                269.96,
                316.56,
                379.85,
                385.81,
                404.31,
                449.11,
                471.47,
                543.64,
                566.99,
                576.14,
                720.02,
                822.82,
                832.94,
                864.02,
                904.02,
                1013.63,
                1066.68,
                1098.51,
                1119.42,
                1170.94,
                1202.56,
                1415.72,
                1499.93,
                2128.93,
                2333.41,
                2501.58,
                3305.47,
                3378.08,
                3617.80,
            ]
        ),
        "ints": torch.tensor(
            [
                1.02724,
                11.27524,
                7.79673,
                6.01311,
                11.25863,
                26.11891,
                15.70509,
                13.25386,
                65.02493,
                30.52791,
                9.68097,
                12.13180,
                27.17798,
                48.10941,
                39.43056,
                73.84893,
                244.84878,
                138.47728,
                27.81167,
                113.58381,
                48.35116,
                344.23257,
                104.56833,
                203.11443,
                206.52690,
                16.37454,
                79.44936,
                106.28927,
                615.11899,
                194.62435,
                43.45037,
                230.11826,
                24.05317,
                44.24037,
                758.40020,
                113.43777,
                393.87364,
                313.47811,
                177.04048,
                7.41869,
                0.80154,
                42.87853,
            ]
        ),
    },
    "LYS_xao": {
        "dipole": torch.tensor([-1.0012, -1.6513, -0.7423]),
        "dipole2": torch.tensor([-14.0946, 0.3401, 2.1676]),
        "quadrupole": torch.tensor(
            [-7.9018, -15.7437, 11.1642, 8.3550, 21.6634, -3.2624]
        ),
        "quadrupole2": torch.tensor(
            [-68.4795, 7.1007, 49.4005, 19.7302, 23.4305, 19.0790],
        ),
        "freqs": torch.tensor(
            [
                -194.80,
                -145.75,
                -136.89,
                -38.34,
                29.69,
                50.47,
                64.99,
                76.08,
                100.32,
                104.11,
                135.22,
                162.97,
                167.77,
                175.90,
                225.53,
                235.05,
                277.23,
                312.82,
                354.21,
                366.26,
                403.08,
                478.78,
                498.85,
                524.21,
                580.85,
                600.49,
                659.18,
                706.67,
                726.24,
                766.10,
                852.58,
                910.44,
                947.25,
                951.70,
                988.33,
                990.63,
                1008.90,
                1033.49,
                1059.40,
                1070.98,
                1079.80,
                1088.02,
                1097.17,
                1099.91,
                1128.30,
                1139.59,
                1160.17,
                1193.88,
                1203.32,
                1231.06,
                1247.48,
                1269.09,
                1278.17,
                1285.21,
                1301.16,
                1324.48,
                1327.88,
                1350.56,
                1359.77,
                1379.81,
                1385.91,
                1406.24,
                1425.55,
                1437.81,
                1447.90,
                1449.87,
                1457.52,
                1466.21,
                1469.65,
                1475.83,
                1488.93,
                1585.35,
                1656.54,
                1689.06,
                3016.32,
                3023.82,
                3032.19,
                3035.51,
                3045.37,
                3047.53,
                3053.87,
                3057.94,
                3061.55,
                3066.42,
                3071.74,
                3098.47,
                3105.42,
                3112.46,
                3133.56,
                3444.09,
                3444.65,
                3508.19,
                3534.91,
            ]
        ),
        "ints": torch.tensor(
            [
                0.30064,
                5.57130,
                4.32087,
                5.48467,
                0.40296,
                1.79258,
                4.31504,
                1.96665,
                16.14644,
                22.33495,
                7.59844,
                6.66851,
                77.87400,
                7.95378,
                1.39116,
                0.99830,
                7.81997,
                18.39108,
                16.17152,
                158.80643,
                16.92263,
                32.62491,
                82.71786,
                83.41603,
                16.13353,
                19.00254,
                7.84907,
                26.31573,
                25.00048,
                10.50814,
                4.85859,
                3.39591,
                8.34989,
                257.06204,
                29.01031,
                32.79689,
                3.64991,
                51.04946,
                3.18792,
                15.21284,
                50.47534,
                2.65474,
                5.48444,
                12.11968,
                16.65278,
                7.43643,
                1.05867,
                16.17067,
                3.60628,
                1.68795,
                38.71487,
                20.38357,
                26.63257,
                13.85973,
                0.92896,
                1.13116,
                3.08628,
                57.20959,
                20.55878,
                3.21913,
                3.76419,
                381.42368,
                221.68257,
                7.86165,
                18.63458,
                12.59819,
                7.86034,
                47.88724,
                4.81830,
                2.13249,
                10.63995,
                81.91686,
                336.91557,
                675.62523,
                17.77610,
                12.86078,
                21.50570,
                19.17176,
                23.17081,
                33.48569,
                3.15788,
                1.36972,
                13.26456,
                23.00528,
                43.36212,
                10.87891,
                1.47907,
                7.90423,
                6.75291,
                84.39868,
                4.60758,
                3.01276,
                18.63315,
            ]
        ),
    },
    "C60": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([-20.5633, 5.3990, 15.6925]),
        "quadrupole": torch.tensor(
            [0.0000, 0.0000, -0.0000, 0.0000, 0.0000, -0.0000]
        ),
        "quadrupole2": torch.tensor(
            [-0.3883, -5.9370, -1.4709, -6.8867, 2.8313, 1.8591]
        ),
        "freqs": torch.tensor([]),
        "ints": torch.tensor([]),
    },
    # "vancoh2": {
    # "dipole": torch.tensor([2.4516, 8.1274, 0.3701]),
    # "dipole2": torch.tensor([-81.9131 ,   22.9779   , 42.5335]),
    # "quadrupole": torch.tensor([-24.1718,   -11.7343 ,   37.5885 ,  -28.6426  ,  20.7013   ,-13.4167]),
    # "quadrupole2": torch.tensor([-56.4013, -713.0469 ,  80.4179   , 58.0042  , 164.6469  , -24.0167]),
    # "freqs": torch.tensor([]),
    # "ints": torch.tensor([]),
    # },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
