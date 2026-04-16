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
    """Spin Constants (same forsp GFN1-xTB and spGFN2-xTB)"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "LiH": {
        # "espgfn1": torch.tensor(),
        # tblite run lih.xyz --method gfn2 --spin 2 --spin-polarized
        # (LiH with 2 valence electrons, spin=2 → 2 alpha, 0 beta)
        "espgfn2": torch.tensor(-6.1184408843268e-01),
        # "gspgfn1": torch.tensor(),
        # "gspgfn2": torch.tensor(),
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
        # "espgfn1": torch.tensor(),
        # tblite run coord --method gfn2 --spin 4 --spin-polarized
        "espgfn2": torch.tensor(-3.2989725755175e00),
        # "gspgfn1": torch.tensor(),
        # "gspgfn2": torch.tensor(),
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
