"""
Run tests for SCF Hessian.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.param import GFN1_XTB as par
from dxtb.utils import _hessian as hessian
from dxtb.xtb import Calculator

from ..utils import reshape_fortran
from .samples import samples

sample_list = ["LiH", "SiH4"]

opts = {
    "exclude": ["disp", "hal", "rep"],
    "maxiter": 50,
    "xitorch_fatol": 1.0e-8,
    "xitorch_xatol": 1.0e-8,
    "verbosity": 0,
    "int_driver": "dxtb",
}

device = None


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    atol, rtol = 1e-4, 1e-1  # should be lower!

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)
    ref = reshape_fortran(
        samples[name]["hessian"].to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    calc = Calculator(numbers, par, opts=opts, **dd)
    print()
    print()

    # numerical hessian
    numref = _numhess(calc, numbers, positions, charge)

    # variable to be differentiated
    positions.requires_grad_(True)

    def scf(numbers, positions, charge) -> Tensor:
        result = calc.singlepoint(numbers, positions, charge)
        return result.scf

    hess = hessian(scf, (numbers, positions, charge), argnums=1)
    print()
    print()
    print(hess)
    print(ref)
    print(numref)
    positions.detach_()
    hess = hess.detach().reshape_as(ref)
    numref = numref.reshape_as(ref)

    assert ref.shape == numref.shape == hess.shape
    assert pytest.approx(ref, abs=1e-6, rel=1e-6) == numref
    assert pytest.approx(ref, abs=atol, rel=rtol) == hess


def _numhess(
    calc: Calculator, numbers: Tensor, positions: Tensor, charge: Tensor
) -> Tensor:
    """Calculate numerical Hessian for reference."""

    hess = torch.zeros(
        *(*positions.shape, *positions.shape),
        **{"device": positions.device, "dtype": positions.dtype},
    )

    def _gradfcn(numbers: Tensor, positions: Tensor, charge: Tensor) -> Tensor:
        positions.requires_grad_(True)
        result = calc.singlepoint(numbers, positions, charge, grad=True)
        positions.detach_()
        return result.total_grad.detach()

    step = 1.0e-4
    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            gr = _gradfcn(numbers, positions, charge)
            print(gr)

            positions[i, j] -= 2 * step
            gl = _gradfcn(numbers, positions, charge)
            print(gl)

            positions[i, j] += step
            hess[:, :, i, j] = 0.5 * (gr - gl) / step

    return hess
