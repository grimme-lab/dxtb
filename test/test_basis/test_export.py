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
from dxtb._src.basis.bas import Basis, format_contraction
from dxtb._src.typing import DD, Literal


def _round_numbers(data: list[str]) -> list[str | float]:
    """Round numbers in a list to 10 decimal places."""
    rounded_data = []
    for item in data:
        try:
            rounded_item = round(float(item), 10)
            rounded_data.append(rounded_item)
        except ValueError:
            rounded_data.append(item)
    return rounded_data


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

    content = _round_numbers(content.split())
    txt = _round_numbers(txt.split())

    assert content == txt


def test_export_gaussian() -> None:
    # always use CPU to have exact same results
    dd: DD = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3], device=dd["device"])
    ihelp = IndexHelper.from_numbers(numbers, GFN2_XTB)
    bas = Basis(numbers, GFN2_XTB, ihelp, **dd)

    # fmt: off
    ref = [
        "Li", "s", 4.0, 1.0, 6.5346266392, -0.0349085439, 1.125316779,
        -0.042610482, 0.0904240047, 0.0682290063, 0.0344628264,
        0.0271924865, "p", 4.0, 1.0, 0.5596087345, 0.0394160282,
        0.1450981885, 0.0364751473, 0.0511516015, 0.019133122,
        0.020364334, 0.002886461, "****",
    ]
    # fmt: on

    txt = bas.to_bse(qcformat="gaussian94")

    txt = _round_numbers(txt.split())
    assert txt == ref


def test_fail_symbol() -> None:
    dd: DD = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3], device=dd["device"])
    ihelp = IndexHelper.from_numbers(numbers, GFN2_XTB)
    bas = Basis(numbers, GFN2_XTB, ihelp, **dd)

    # wipe shells info to trigger error
    bas.shells = {}

    with pytest.raises(ValueError) as e:
        bas.to_bse(qcformat="gaussian94")

    assert str(e.value) == "Element 'Li' not found in the basis set."


def test_format_contraction() -> None:
    """
    Test the format_contraction function.
    """
    shells = ["1s", "2p", "3d"]
    ngauss = torch.tensor([6, 6, 6], dtype=torch.uint8)

    result = format_contraction(shells, ngauss)
    assert result == "(6s,6p,6d) -> [1s,1p,1d]"


def test_format_contraction_fail() -> None:
    """
    Test the format_contraction function.
    """
    shells = ["1x"]
    ngauss = torch.tensor([6], dtype=torch.uint8)

    with pytest.raises(ValueError):
        format_contraction(shells, ngauss)
