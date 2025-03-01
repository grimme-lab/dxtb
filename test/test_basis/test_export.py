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
Test export of the basis set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.typing import DD, Literal


@pytest.mark.parametrize("number", range(1, 87))
@pytest.mark.parametrize("qcformat", ["nwchem"])
@pytest.mark.parametrize("xtb_version", ["gfn1", "gfn2"])
def test_export(
    number: int,
    qcformat: Literal["gaussian94", "nwchem"],
    xtb_version: Literal["gfn1", "gfn2"],
    dtype: torch.dtype = torch.double,
):
    # always use CPU to have exact same results
    dd: DD = {"dtype": dtype, "device": torch.device("cpu")}

    numbers = torch.tensor([number], device=dd["device"])

    if xtb_version == "gfn1":
        par = GFN1_XTB
    elif xtb_version == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)

    txt = bas.to_bse(qcformat=qcformat)

    # check with saved basis files
    root = Path(__file__).parents[2]
    s = f"src/dxtb/_src/exlibs/pyscf/mol/basis/{xtb_version}/{number:02d}"
    p = root / f"{s}.{qcformat}"
    assert p.exists(), f"Basis set not found in '{p}'."

    with open(p, encoding="utf-8") as f:
        content = f.read()

    def round_numbers(data):
        rounded_data = []
        for item in data:
            try:
                rounded_item = round(float(item), 10)
                rounded_data.append(rounded_item)
            except ValueError:
                rounded_data.append(item)
        return rounded_data

    content = round_numbers(content.split())
    txt = round_numbers(txt.split())

    assert content == txt
