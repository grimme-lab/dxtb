"""
Run tests for the gradient of the Hamiltonian matrix.
References calculated with tblite 0.3.0.
"""

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.integral import Overlap
from dxtb.ncoord import exp_count, get_coordination_number
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch
from dxtb.xtb import Hamiltonian, Calculator

from ..utils import load_from_npz
from .samples import samples

small = ["C", "Rn", "H2", "LiH", "HLi", "S2", "SiH4"]
large = ["PbH4-BiH3", "LYS_xao"]


opts = {"verbosity": 0}


@pytest.mark.parametrize("dtype", [torch.float])  # , torch.double])
@pytest.mark.parametrize("name", ["LiH"])
def test_single(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    chrg = torch.tensor(0.0, **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, chrg)

    doverlap = torch.tensor(
        [
            +0.0000000000000000e00,
            -2.4923262869826657e-02,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            -6.8918795086107362e-02,
            -2.4923262869826657e-02,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
            +0.0000000000000000e00,
        ],
        **dd
    )

    print("")

    cn = get_coordination_number(numbers, positions, exp_count)
    pmat = result.density
    wmat = torch.einsum(
        "...ik,...k,...jk->...ij",
        result.coefficients,
        result.emo * result.occupation.sum(-2),
        result.coefficients,  # transposed
    )
    pot = result.potential

    grad = calc.hamiltonian.get_gradient(
        positions,
        result.overlap,
        doverlap,
        pmat,
        wmat,
        pot,
        cn,
    )

    print(grad)
