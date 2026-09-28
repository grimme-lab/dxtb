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
Reference single point energies (from tblite).
"""

from __future__ import annotations

import torch
from tad_mctc.data.molecules import merge_nested_dicts, mols

from dxtb._src.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """
    Format of reference records containing GFN0-xTB, GFN1-xTB and GFN2-xTB reference values.
    """

    egfn0: Tensor
    """Total energy for GFN0-xTB"""

    egfn1: Tensor
    """Total energy for GFN1-xTB"""

    egfn2: Tensor
    """Total energy for GFN2-xTB"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H": {
        "egfn0": torch.tensor(-4.38138811313e-01),
        "egfn1": torch.tensor(-4.0142947446183e-01),
        "egfn2": torch.tensor(-3.9348275927054e-01),
    },
    "H2": {
        "egfn0": torch.tensor(-1.150088142708e00),
        "egfn1": torch.tensor(-1.0362714373390e00),
        "egfn2": torch.tensor(-9.8211694450068e-01),
    },
    "LiH": {
        "egfn0": torch.tensor(-9.94315793446e-01),
        "egfn1": torch.tensor(-8.8142483637514e-01),
        "egfn2": torch.tensor(-7.7924380347126e-01),
    },
    "H2O": {
        "egfn0": torch.tensor(-4.366814051917e00),
        "egfn1": torch.tensor(-5.7686218257620e00),
        "egfn2": torch.tensor(-5.0703655057333e00),
    },
    "NO2": {
        "egfn0": torch.tensor(-9.361706469426e00),
        "egfn1": torch.tensor(-1.2409798675060e01),
        "egfn2": torch.tensor(-1.0906101651436e01),
    },
    "CH4": {
        "egfn0": torch.tensor(-4.359243832062e00),
        "egfn1": torch.tensor(-4.2741992424931e00),
        "egfn2": torch.tensor(-4.1750000873275e00),
    },
    "SiH4": {
        "egfn0": torch.tensor(-4.097876348630e00),
        "egfn1": torch.tensor(-4.0087585461086e00),
        "egfn2": torch.tensor(-3.7632337516532e00),
    },
    "LYS_xao": {
        "egfn0": torch.tensor(-4.5003398823421e01),
        "egfn1": torch.tensor(-4.8324739766346e01),
        "egfn2": torch.tensor(-4.5885049573120e01),
    },
    "C60": {
        "egfn0": torch.tensor(-1.25545821269601e02),
        "egfn1": torch.tensor(-1.2673081838911e02),
        "egfn2": torch.tensor(-1.2845329122498e02),
    },
    "vancoh2": {
        "egfn0": torch.tensor(-2.86917484536758e02),
        "egfn1": torch.tensor(-3.2295379428673e02),
        "egfn2": torch.tensor(-3.0473320623366e02),
    },
    "AD7en+": {
        "egfn0": torch.tensor(-4.1766556682529e01),
        "egfn1": torch.tensor(-4.2547841532513e01),
        "egfn2": torch.tensor(-4.2203000285882e01),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
