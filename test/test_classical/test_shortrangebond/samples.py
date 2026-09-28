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
Reference values for short-range bond correction tests.
"""

from __future__ import annotations

import torch
from tad_mctc.data.molecules import merge_nested_dicts, mols

from dxtb._src.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """Format of reference records containing GFN0-xTB SRB values."""

    esrb: Tensor
    """Reference short range bond correction energy."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "LiH": {
        "esrb": torch.tensor(0.000000000000, dtype=torch.float64),
    },
    "SiH4": {
        "esrb": torch.tensor(0.000000000000, dtype=torch.float64),
    },
    "PbH4-BiH3": {
        "esrb": torch.tensor(0.000000000000, dtype=torch.float64),
    },
    "vancoh2": {
        "esrb": torch.tensor(-0.575042732267, dtype=torch.float64),
    },
    "MB16_43_01": {
        "esrb": torch.tensor(-0.026453365313, dtype=torch.float64),
    },
    "Ag2Cl22-": {
        "esrb": torch.tensor(0.000000000000, dtype=torch.float64),
    },
    "Rn": {
        "esrb": torch.tensor(0.000000000000, dtype=torch.float64),
    },
    "LYS_xao": {
        "esrb": torch.tensor(-0.088162441935, dtype=torch.float64),
    },
    "NO2": {
        "esrb": torch.tensor(-0.025539163389, dtype=torch.float64),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
