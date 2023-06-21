"""
Run tests for SCF Hessian.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import Tensor
from dxtb.param import GFN1_XTB as par
from dxtb.utils import hessian, reshape_fortran
from dxtb.xtb import Calculator

from .samples import samples

sample_list = ["LiH", "SiH4"]

opts = {
    "exclude": ["disp", "hal", "rep"],
    "maxiter": 50,
    "xitorch_fatol": 1.0e-8,
    "xitorch_xatol": 1.0e-8,
    "verbosity": 0,
}


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd = {"dtype": dtype}
    atol, rtol = 1e-4, 1e-1  # should be lower!

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype)
    charge = torch.tensor(0.0, **dd)
    ref = reshape_fortran(
        samples[name]["hessian"].type(dtype),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    calc = Calculator(numbers, par, opts=opts, **dd)

    # numerical hessian
    numref = _numhess(calc, numbers, positions, charge)

    # variable to be differentiated
    positions.requires_grad_(True)

    def scf(numbers, positions, charge) -> Tensor:
        result = calc.singlepoint(numbers, positions, charge)
        return result.scf

    hess = hessian(scf, (numbers, positions, charge), argnums=1)
    assert ref.shape == numref.shape == hess.shape
    assert pytest.approx(ref, abs=1e-6, rel=1e-6) == numref
    assert pytest.approx(ref, abs=atol, rel=rtol) == hess.detach()

    positions.detach_()


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

            positions[i, j] -= 2 * step
            gl = _gradfcn(numbers, positions, charge)

            positions[i, j] += step
            hess[:, :, i, j] = 0.5 * (gr - gl) / step

    return hess
