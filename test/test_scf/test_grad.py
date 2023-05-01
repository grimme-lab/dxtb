from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator

from ..utils import load_from_npz
from .samples import samples

opts = {"verbosity": 0, "maxiter": 50, "exclude": ["rep", "disp", "hal"]}

ref_grad = np.load("test/test_scf/grad.npz")
ref_grad_param = np.load("test/test_scf/grad_param.npz")


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH", "SiH4"])
def test_grad_backwards(name: str, dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype)
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    # Values obtained with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    options = dict(opts, **{"exclude": ["rep", "disp", "hal"]})
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    energy.backward()
    assert positions.grad is not None
    gradient = positions.grad.clone()
    positions.detach_()

    assert pytest.approx(ref, abs=tol) == gradient


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "CH4", "SiH4"])
def test_grad_autograd(name: str, dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype)
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    # Values obtained with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    # Values obtained with dxtb using full gradient tracking
    ref_full = load_from_npz(ref_grad, f"{name}_full", dtype)

    assert pytest.approx(ref_full, abs=tol) == ref

    options = dict(opts, **{"xitorch_fatol": tol**2, "xitorch_xatol": tol**2})
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    gradient = torch.autograd.grad(
        energy,
        positions,
    )[0]

    assert pytest.approx(gradient, abs=tol) == ref
    assert pytest.approx(gradient, abs=tol) == ref_full

    positions.detach_()


# FIXME: fails for LYS_xao_dist
@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", ["LYS_xao", "C60", "vancoh2"])
def test_grad_large(name: str, dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype)
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    # Values obtained with dxtb using full gradient tracking
    ref_full = load_from_npz(ref_grad, f"{name}_full", dtype)

    assert pytest.approx(ref_full, abs=tol, rel=1e-5) == ref

    options = dict(opts, **{"xitorch_fatol": tol**2, "xitorch_xatol": tol**2})
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    energy = result.scf.sum(-1)

    gradient = torch.autograd.grad(
        energy,
        positions,
    )[0]

    assert pytest.approx(gradient, abs=tol, rel=1e-5) == ref
    assert pytest.approx(gradient, abs=tol, rel=1e-5) == ref_full

    positions.detach_()


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
    positions = samples[name]["positions"].type(dtype)
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    options = dict(opts, **{"xitorch_fatol": tol**2, "xitorch_xatol": tol**2})
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

    ref_se = load_from_npz(ref_grad_param, f"{name}_egrad_selfenergy", dtype)
    assert pytest.approx(pgrad[0], abs=tol) == ref_se
    ref_kcn = load_from_npz(ref_grad_param, f"{name}_egrad_kcn", dtype)
    assert pytest.approx(pgrad[1], abs=tol) == ref_kcn
    ref_shpoly = load_from_npz(ref_grad_param, f"{name}_egrad_shpoly", dtype)
    assert pytest.approx(pgrad[2], abs=tol) == ref_shpoly

    positions.detach_()


# FIXME!
@pytest.mark.grad
@pytest.mark.parametrize("name", ["LiH", "H2O", "SiH4", "LYS_xao"])
def skip_test_param_grad_force(name: str, dtype: torch.dtype = torch.float):
    """
    Test autograd of SCF without gradient tracking vs. SCF with full gradient
    tracking. References obtained with full tracking and `torch.float`.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype)
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    options = dict(opts, **{"xitorch_fatol": tol**2, "xitorch_xatol": tol**2})
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

    ref_se = load_from_npz(ref_grad_param, f"{name}_ggrad_selfenergy", dtype)
    assert pytest.approx(pgrad[0], abs=tol) == ref_se
    ref_kcn = load_from_npz(ref_grad_param, f"{name}_ggrad_kcn", dtype)
    assert pytest.approx(pgrad[1], abs=tol) == ref_kcn
    ref_shpoly = load_from_npz(ref_grad_param, f"{name}_ggrad_shpoly", dtype)
    assert pytest.approx(pgrad[2], abs=tol) == ref_shpoly

    positions.detach_()
