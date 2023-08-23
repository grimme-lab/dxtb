"""
Run tests for IR spectra.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.constants import units
from dxtb.interaction import new_efield
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from ..utils import dgradcheck
from .samples import samples

sample_list = ["LiH", "H2O", "SiH4"]

opts = {
    "maxiter": 100,
    "xitorch_fatol": 1.0e-10,
    "xitorch_xatol": 1.0e-10,
    "verbosity": 0,
    "scf_mode": "full",
    "mixer": "anderson",
}

device = None


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)

    # required for autodiff of energy w.r.t. efield and dipole
    # field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    def f(pos: Tensor) -> Tensor:
        return calc.dipole(numbers, pos, charge)

    assert dgradcheck(f, positions)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    ref = samples[name]["dipole"].to(**dd)

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)

    # required for autodiff of energy w.r.t. efield and dipole
    field_vector.requires_grad_(True)
    positions.requires_grad_(True)

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # ir = calc.ir_spectrum(numbers, positions, charge)
    # print(ir)
    print("")
    print("")
    print("")
    #############################
    from dxtb.constants import units

    dmu_dr = num_grad(calc, numbers, positions, charge).view(3, -1)
    # print("dmu_dr\n", dmu_dr)

    freqs, modes = calc.vibration(numbers, positions, charge, True)
    dmu_dq = torch.matmul(dmu_dr, modes)  # (ndim, nfreqs)
    ir_ints = torch.einsum("...df,...df->...f", dmu_dq, dmu_dq)  # (nfreqs,)

    # print(ir_ints)
    # print(ir_ints * units.AU2KMMOL)
    # print(ir_ints * 974.8801118351438)
    print(ir_ints * 1378999.7790799031)
    print(freqs * units.AU2RCM)

    print("")

    # assert pytest.approx(ref, abs=1e-3) == dipole


def num_grad(calc: Calculator, numbers, positions, charge) -> Tensor:
    # setup numerical gradient
    positions = positions.detach().clone()
    n_atoms = positions.shape[0]
    gradient = torch.zeros((3, n_atoms, 3), dtype=positions.dtype)

    step = 1.0e-7

    for i in range(n_atoms):
        for j in range(3):
            positions[i, j] += step
            er = calc.dipole(numbers, positions, charge)

            positions[i, j] -= 2 * step
            el = calc.dipole(numbers, positions, charge)

            positions[i, j] += step
            gradient[:, i, j] = 0.5 * (er - el) / step

    return gradient


def _project_freqs(
    freqs: Tensor, modes: Tensor, is_linear: bool = False
) -> tuple[Tensor, Tensor]:
    skip = 5 if is_linear is True else 6
    freqs = freqs[skip:]  # (nfreqs,)
    modes = modes[:, skip:]  # (natoms * ndim, nfreqs)
    return freqs, modes
