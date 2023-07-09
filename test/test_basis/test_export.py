"""
Test export of the basis set.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dxtb._types import Literal
from dxtb.basis import Basis, IndexHelper
from dxtb.param import GFN1_XTB, get_elem_angular


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

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp.unique_angular, dtype=dtype)

    txt = bas.to_bse(ihelp, qcformat=qcformat)

    # check with saved basis files
    root = Path(__file__).parents[2]
    p = root / f"src/dxtb/mol/external/basis/{xtb_version}/{number:02d}.{qcformat}"
    assert p.exists()

    with open(p, encoding="utf8") as f:
        content = f.read()

    assert content == txt
