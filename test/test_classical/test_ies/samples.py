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
Reference values for isotropic electrostatics tests.
"""

from __future__ import annotations

import torch
from tad_mctc.data.molecules import merge_nested_dicts, mols

from dxtb._src.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """Format of reference records containing GFN0-xTB IES values."""

    eies: Tensor
    """Reference isotropic electrostatics energy."""

    charge: Tensor
    """Total molecular charge."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "LiH": {
        "charge": torch.tensor(0.0),
        "eies": torch.tensor(-0.060894980661, dtype=torch.float64),
    },
    "SiH4": {
        "charge": torch.tensor(0.0),
        "eies": torch.tensor(-0.030492301198, dtype=torch.float64),
    },
    "PbH4-BiH3": {
        "charge": torch.tensor(0.0),
        "eies": torch.tensor(-0.000478532711, dtype=torch.float64),
    },
    "vancoh2": {
        "charge": torch.tensor(0.0),
        "eies": torch.tensor(-1.341577342595, dtype=torch.float64),
    },
    "MB16_43_01": {
        "charge": torch.tensor(0.0),
        "eies": torch.tensor(-0.328372556259, dtype=torch.float64),
    },
    "Ag2Cl22-": {
        "charge": torch.tensor(-2.0),
        "eies": torch.tensor(-2.204807969012, dtype=torch.float64),
    },
    "Rn": {
        "charge": torch.tensor(0.0),
        "eies": torch.tensor(0.000000000000, dtype=torch.float64),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
