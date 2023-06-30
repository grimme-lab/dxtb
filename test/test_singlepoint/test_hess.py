"""
Run tests for singlepoint gradient calculation with read from coord file.
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path

import pytest
import torch

from dxtb._types import Tensor
from dxtb.io import read_chrg, read_coord
from dxtb.param import GFN1_XTB as par
from dxtb.utils import hessian, reshape_fortran
from dxtb.xtb import Calculator

from ..test_dispersion.samples import samples as samples_disp
from ..test_halogen.samples import samples as samples_hal
from ..test_repulsion.samples import samples as samples_rep

opts = {
    "verbosity": 0,
    "maxiter": 50,
    "xitorch_fatol": 1.0e-10,
    "xitorch_xatol": 1.0e-10,
}


sample_list = ["LiH", "SiH4", "MB16_43_01"]


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    # convert to tensors
    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions, **dd)
    charge = torch.tensor(charge, **dd)

    # variable to be differentiated
    positions.requires_grad_(True)

    options = dict(opts, **{"exclude": ["scf"]})
    calc = Calculator(numbers, par, opts=options, **dd)

    def singlepoint(numbers, positions, charge) -> Tensor:
        result = calc.singlepoint(numbers, positions, charge)
        return result.total

    positions.requires_grad_(True)

    ref = reshape_fortran(
        (
            samples_disp[name]["hessian"]
            + samples_hal[name]["hessian"]
            + samples_rep[name]["gfn1_hess"]
        ).type(dtype),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    hess = hessian(singlepoint, (numbers, positions, charge), argnums=1)
    assert pytest.approx(ref, abs=tol, rel=tol) == hess.detach()

    positions.detach_()
