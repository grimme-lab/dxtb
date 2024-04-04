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

from dxtb.basis import Basis, IndexHelper
from dxtb.param import GFN1_XTB
from dxtb.typing import Literal


@pytest.mark.parametrize("number", range(1, 87))
@pytest.mark.parametrize("qcformat", ["nwchem"])
@pytest.mark.parametrize("xtb_version", ["gfn1"])
def test_export(
    number: int,
    qcformat: Literal["gaussian94", "nwchem"],
    xtb_version: Literal["gfn1", "gfn2"],
    dtype: torch.dtype = torch.double,
):
    numbers = torch.tensor([number])

    if xtb_version == "gfn1":
        par = GFN1_XTB
    else:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, dtype=dtype)

    txt = bas.to_bse(qcformat=qcformat)

    # check with saved basis files
    root = Path(__file__).parents[2]
    p = root / f"src/dxtb/mol/external/basis/{xtb_version}/{number:02d}.{qcformat}"
    assert p.exists()

    with open(p, encoding="utf8") as f:
        content = f.read()

    assert content == txt
