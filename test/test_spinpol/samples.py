# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Reference energies and spinconstants (from tblite)
https://github.com/tblite/tblite
"""

from __future__ import annotations

import torch
from tad_mctc.data.molecules import merge_nested_dicts, mols

from dxtb._src.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """
    Format of reference records containing spGFN1-xTB and spGFN2-xTB reference values.
    """

    espgfn1: Tensor
    """Total energy for spGFN1-xTB"""

    espgfn2: Tensor
    """Total energy for spGFN2-xTB"""

    gspgfn1: Tensor
    """Gradient of spGFN1-xTB """

    gspgfn2: Tensor
    """Gradient of spGFN2-xTB"""

    spconst: Tensor
    """Spin Constants (same for spGFN1-xTB and spGFN2-xTB)"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "MB16_43_02": {
        # tblite run coord --spin-polarized --spin 1 --method gfn1
        "espgfn1": torch.tensor(-2.6860491628287e01),
        "espgfn2": torch.tensor(-2.4072043583538e01),
    },
    "LiH": {
        # spin-polarized GFN1 reference (spin=2)
        "espgfn1": torch.tensor(-7.2271964862480e-01),
        # tblite run lih.xyz --method gfn2 --spin 2 --spin-polarized
        # (LiH with 2 valence electrons, spin=2 → 2 alpha, 0 beta)
        "espgfn2": torch.tensor(-6.1184408843268e-01),
        "gspgfn1": torch.tensor(
            [
                [
                    9.8670107052753681e-18,
                    0.0000000000000000e00,
                    1.1789296433583266e-02,
                ],
                [
                    -9.8670107052753681e-18,
                    0.0000000000000000e00,
                    -1.1789296433583266e-02,
                ],
            ]
        ),
        "gspgfn2": torch.tensor(
            [
                [
                    -5.5584836545384057e-18,
                    2.6279636649476664e-39,
                    1.6154983674634370e-02,
                ],
                [
                    5.5584836545384057e-18,
                    -2.6279636649476664e-39,
                    -1.6154983674634370e-02,
                ],
            ]
        ),
        "spconst": torch.tensor(
            [
                [
                    -0.0178000,
                    -0.0139500,
                    -0.0180500,
                    0.0000000,
                    0.0000000,
                    0.0000000,
                ],
                [
                    -0.0716250,
                    0.0000000,
                    0.0000000,
                    0.0000000,
                    0.0000000,
                    0.0000000,
                ],
            ]
        ),
        "wllgfn1": torch.tensor(
            [
                [-0.0178000, -0.0139500, 0, 0],
                [-0.0139500, -0.0180500, 0, 0],
                [0, 0, -0.0716250, -0.0716250],
                [0, 0, -0.0716250, -0.0716250],
            ]
        ),
        "wllgfn2": torch.tensor(
            [
                [-0.0178000, -0.0139500, 0],
                [-0.0139500, -0.0180500, 0],
                [0, 0, -0.0716250],
            ]
        ),
        "eshellgfn2": torch.tensor(
            [
                -0.01104296,
                -0.02489042,
            ]
        ),  # after one scf iteration
        "potshellgfn2": torch.tensor([0.00778357, 0.00706252, 0.02388490]),
    },
    "SiH4": {
        # spin-polarized GFN1 reference (spin=2)
        "espgfn1": torch.tensor(-3.7411749857645e00),
        # tblite run coord --method gfn2 --spin 2 --spin-polarized
        "espgfn2": torch.tensor(-3.5279708942225e00),
        "gspgfn1": torch.tensor(
            [
                [
                    9.4627773498945356e-11,
                    -1.1906547107845902e-10,
                    4.3905345234029197e-11,
                ],
                [
                    -2.2289509122165969e-03,
                    -2.2289508421061300e-03,
                    2.2289508667640732e-03,
                ],
                [
                    2.2289508463076632e-03,
                    2.2289509164178239e-03,
                    2.2289508629479583e-03,
                ],
                [
                    -2.2289508867627448e-03,
                    2.2289508947800794e-03,
                    -2.2289508701216791e-03,
                ],
                [
                    2.2289508580438998e-03,
                    -2.2289508500262993e-03,
                    -2.2289509034956836e-03,
                ],
            ]
        ),
        "gspgfn2": torch.tensor(
            [
                [
                    9.4719476554684656e-10,
                    -1.1077132956553715e-09,
                    -8.2794972483882234e-10,
                ],
                [
                    -7.3064361903602626e-03,
                    -7.3064354959752002e-03,
                    7.3064361485410505e-03,
                ],
                [
                    7.3064355235252232e-03,
                    7.3064362179122338e-03,
                    7.3064361233837480e-03,
                ],
                [
                    -7.3064361411859123e-03,
                    7.3064361938951737e-03,
                    -7.3064355413285046e-03,
                ],
                [
                    7.3064358608261881e-03,
                    -7.3064358081189017e-03,
                    -7.3064359026465634e-03,
                ],
            ]
        ),
        "wllgfn1": torch.tensor(
            [
                [
                    -0.01952500,
                    -0.01500000,
                    -0.00845000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    -0.01500000,
                    -0.01437500,
                    -0.01161200,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    -0.00845000,
                    -0.01161200,
                    -0.01400000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    -0.07162500,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    -0.07162500,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    -0.07162500,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    -0.07162500,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    -0.07162500,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    -0.07162500,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    -0.07162500,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    -0.07162500,
                ],
            ]
        ),
        "wllgfn2": torch.tensor(
            [
                [
                    -0.01952500,
                    -0.01500000,
                    -0.00845000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    -0.01500000,
                    -0.01437500,
                    -0.01161200,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    -0.00845000,
                    -0.01161200,
                    -0.01400000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    0.00000000,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    -0.07162500,
                ],
            ]
        ),
    },
}

samples: dict[str, Record] = merge_nested_dicts(mols, refs)
