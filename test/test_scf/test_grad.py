from math import sqrt

import numpy as np
import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator

from ..utils import load_from_npz
from .samples import samples
from .samples_grad import refs_grad

opts = {"verbosity": 0, "maxiter": 50, "exclude": ["rep", "disp", "hal"]}

ref_grad = np.load("test/test_scf/grad.npz")


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH"])
def test_grad_backwards(name: str, dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    pos = samples[name]["positions"].type(dtype)
    positions = pos.detach().clone().requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    # Values obtained with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    options = dict(opts, **{"exclude": ["rep", "disp", "hal"]})
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    energy.backward()
    if positions.grad is None:
        assert False
    gradient = positions.grad.clone()
    assert pytest.approx(gradient, abs=tol) == ref


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "CH4", "SiH4"])
def test_grad(name: str, dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    # Values obtained with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    options = dict(
        opts,
        **{
            "exclude": ["rep", "disp", "hal"],
            "maxiter": 50,
            "xitorch_fatol": 1.0e-8,
            "xitorch_xatol": 1.0e-8,
        }
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    gradient = torch.autograd.grad(
        energy,
        positions,
    )[0]

    assert pytest.approx(gradient, abs=tol, rel=1e-5) == ref


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", ["LYS_xao", "LYS_xao_dist", "C60", "vancoh2"])
def test_grad_large(name: str, dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    options = dict(
        opts,
        **{
            "exclude": ["rep", "disp", "hal"],
            "maxiter": 50,
            "xitorch_fatol": 1.0e-10,
            "xitorch_xatol": 1.0e-10,
        }
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    gradient = torch.autograd.grad(
        energy,
        positions,
    )[0]

    assert pytest.approx(gradient, abs=tol, rel=1e-5) == ref


@pytest.mark.grad
@pytest.mark.parametrize(
    "testcase",
    [
        (
            "LiH",
            {
                "selfenergy": torch.tensor(
                    [+0.0002029369, +0.0017547115, +0.1379896402, -0.1265652627]
                ),
                "kcn": torch.tensor(
                    [-0.1432282478, -0.0013212233, -0.1811404824, +0.0755317509]
                ),
                "shpoly": torch.tensor(
                    [+0.0408593193, -0.0007219329, -0.0385218151, +0.0689999014]
                ),
            },
        ),
    ],
)
def test_param_gradgrad(testcase, dtype: torch.dtype = torch.float):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    name, ref = testcase
    numbers = samples[name]["numbers"]
    pos = samples[name]["positions"].type(dtype)
    positions = pos.detach().clone().requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)
    calc.hamiltonian.selfenergy.requires_grad_(True)
    calc.hamiltonian.kcn.requires_grad_(True)
    calc.hamiltonian.shpoly.requires_grad_(True)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    gradient = torch.autograd.grad(
        energy,
        positions,
        create_graph=True,
    )[0]

    pgrad = torch.autograd.grad(
        gradient[0, :].sum(),
        (calc.hamiltonian.selfenergy, calc.hamiltonian.kcn, calc.hamiltonian.shpoly),
    )

    assert pytest.approx(pgrad[0], abs=tol) == ref["selfenergy"]
    assert pytest.approx(pgrad[1], abs=tol) == ref["kcn"]
    assert pytest.approx(pgrad[2], abs=tol) == ref["shpoly"]


@pytest.mark.grad
@pytest.mark.parametrize("name", ["LiH", "H2O", "SiH4", "LYS_xao"])
def test_param_grad_energy(name: str, dtype: torch.dtype = torch.float):
    """
    Test autograd of SCF without gradient tracking vs. SCF with full gradient
    tracking. References obtained with full tracking and `torch.float`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)
    ref = refs_grad[name]["egrad"]
    charges = torch.tensor(0.0, **dd)

    options = dict(opts, **{"xitorch_fatol": 1.0e-6, "xitorch_xatol": 1.0e-6})
    calc = Calculator(numbers, par, opts=options, **dd)
    calc.hamiltonian.selfenergy.requires_grad_(True)
    calc.hamiltonian.kcn.requires_grad_(True)
    calc.hamiltonian.shpoly.requires_grad_(True)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    pgrad = torch.autograd.grad(
        energy,
        (calc.hamiltonian.selfenergy, calc.hamiltonian.kcn, calc.hamiltonian.shpoly),
    )

    assert pytest.approx(pgrad[0], abs=tol) == ref["selfenergy"]
    assert pytest.approx(pgrad[1], abs=tol) == ref["kcn"]
    assert pytest.approx(pgrad[2], abs=tol) == ref["shpoly"]


@pytest.mark.grad
@pytest.mark.parametrize("name", ["LiH", "H2O", "SiH4", "LYS_xao"])
def test_param_grad_force(name: str, dtype: torch.dtype = torch.float):
    """
    Test autograd of SCF without gradient tracking vs. SCF with full gradient
    tracking. References obtained with full tracking and `torch.float`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype).detach().clone()
    positions.requires_grad_(True)
    ref = refs_grad[name]["ggrad"]
    charges = torch.tensor(0.0, **dd)

    options = dict(opts, **{"xitorch_fatol": 1.0e-6, "xitorch_xatol": 1.0e-6})
    calc = Calculator(numbers, par, opts=options, **dd)
    calc.hamiltonian.selfenergy.requires_grad_(True)
    calc.hamiltonian.kcn.requires_grad_(True)
    calc.hamiltonian.shpoly.requires_grad_(True)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    (gradient,) = torch.autograd.grad(
        energy,
        positions,
        create_graph=True,
    )

    pgrad = torch.autograd.grad(
        gradient[0, :].sum(),
        (calc.hamiltonian.selfenergy, calc.hamiltonian.kcn, calc.hamiltonian.shpoly),
    )

    assert pytest.approx(pgrad[0], abs=tol) == ref["selfenergy"]
    assert pytest.approx(pgrad[1], abs=tol) == ref["kcn"]
    assert pytest.approx(pgrad[2], abs=tol) == ref["shpoly"]
