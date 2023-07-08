"""
Test export of the basis set.
"""
from __future__ import annotations

import torch

from dxtb.basis import Basis, IndexHelper
from dxtb.param import GFN1_XTB, get_elem_angular


def test_export(dtype: torch.dtype = torch.double):
    # arbitrary elements
    numbers = torch.arange(1, 87)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    bas = Basis(numbers, GFN1_XTB, ihelp.unique_angular, dtype=dtype)
    bas.to_bse_nwchem(ihelp, verbose=True, save=False, overwrite=True)
    bas.to_bse_gaussian94(ihelp, save=False, overwrite=True)
