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
Gradient tests for SCF.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.constants import labels
from dxtb._src.typing import DD

from ..conftest import DEVICE
from ..utils import load_from_npz
from .samples import samples

opts = {
    "verbosity": 0,
    "maxiter": 50,
    "exclude": ["rep", "disp", "hal"],
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
}

ref_grad = np.load("test/test_scf/grad.npz")
ref_grad_param = np.load("test/test_scf/grad_param.npz")


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH"])
@pytest.mark.parametrize("scp_mode", ["potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full", "single-shot"])
def test_grad_backwards(
    name: str, dtype: torch.dtype, scf_mode: str, scp_mode: str
) -> None:
    run_grad_backwards(name, dtype, scf_mode, scp_mode)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["SiH4"])
@pytest.mark.parametrize("scp_mode", ["potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full", "single-shot"])
def test_grad_backwards_large(
    name: str, dtype: torch.dtype, scf_mode: str, scp_mode: str
) -> None:
    run_grad_backwards(name, dtype, scf_mode, scp_mode)


def run_grad_backwards(
    name: str, dtype: torch.dtype, scf_mode: str, scp_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    # Values obtained with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "scp_mode": scp_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charges)
    energy = result.scf.sum(-1)

    energy.backward()
    assert positions.grad is not None

    gradient = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == gradient.cpu()


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH"])
def test_grad_autograd(name: str, dtype: torch.dtype):
    run_grad_autograd(name, dtype)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "H2O", "CH4", "SiH4"])
def run_grad_autograd(name: str, dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    # Values obtained with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    # Values obtained with dxtb using full gradient tracking
    ref_full = load_from_npz(ref_grad, f"{name}_full", dtype)

    assert pytest.approx(ref_full, abs=tol, rel=1e-5) == ref

    options = dict(opts, **{"f_atol": tol, "x_atol": tol})
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charges)
    energy = result.scf.sum(-1)

    (gradient,) = torch.autograd.grad(energy, positions)

    assert pytest.approx(gradient.cpu(), abs=tol, rel=1e-5) == ref.cpu()
    assert pytest.approx(gradient.cpu(), abs=tol, rel=1e-5) == ref_full.cpu()

    positions.detach_()


# FIXME: fails for LYS_xao_dist
@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", ["LYS_xao", "C60", "vancoh2"])
def test_grad_large(name: str, dtype: torch.dtype):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    # Values obtained with dxtb using full gradient tracking
    ref_full = load_from_npz(ref_grad, f"{name}_full", dtype)

    assert pytest.approx(ref_full.cpu(), abs=tol, rel=1e-5) == ref.cpu()

    # variable to be differentiated
    positions.requires_grad_(True)

    options = dict(opts, **{"f_atol": tol**2, "x_atol": tol**2})
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charges)
    energy = result.scf.sum(-1)

    (gradient,) = torch.autograd.grad(energy, positions)

    assert pytest.approx(gradient.cpu(), abs=tol, rel=1e-5) == ref.cpu()
    assert pytest.approx(gradient.cpu(), abs=tol, rel=1e-5) == ref_full.cpu()

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("name", ["LiH"])
def test_param_grad_energy(name: str, dtype: torch.dtype = torch.float):
    """
    Test autograd of SCF without gradient tracking vs. SCF with full gradient
    tracking. References obtained with full tracking and `torch.float`.
    """
    run_param_grad_energy(name, dtype)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("name", ["H2O", "SiH4", "LYS_xao"])
def test_param_grad_energy_large(name: str, dtype: torch.dtype = torch.float):
    """
    Test autograd of SCF without gradient tracking vs. SCF with full gradient
    tracking. References obtained with full tracking and `torch.float`.
    """
    run_param_grad_energy(name, dtype)


def run_param_grad_energy(name: str, dtype: torch.dtype = torch.float):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    options = dict(opts, **{"f_atol": tol**2, "x_atol": tol**2})
    calc = Calculator(numbers, par, opts=options, **dd)

    assert calc.integrals.hcore is not None
    h = calc.integrals.hcore.integral
    h.selfenergy.requires_grad_(True)
    h.kcn.requires_grad_(True)
    h.shpoly.requires_grad_(True)

    result = calc.singlepoint(positions, charges)
    energy = result.scf.sum(-1)

    pgrad = torch.autograd.grad(
        energy,
        (h.selfenergy, h.kcn, h.shpoly),
    )

    ref_se = load_from_npz(ref_grad_param, f"{name}_egrad_selfenergy", dtype)
    assert pytest.approx(pgrad[0].cpu(), abs=tol) == ref_se.cpu()
    ref_kcn = load_from_npz(ref_grad_param, f"{name}_egrad_kcn", dtype)
    assert pytest.approx(pgrad[1].cpu(), abs=tol) == ref_kcn.cpu()
    ref_shpoly = load_from_npz(ref_grad_param, f"{name}_egrad_shpoly", dtype)
    assert pytest.approx(pgrad[2].cpu(), abs=tol) == ref_shpoly.cpu()

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
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    positions.requires_grad_(True)
    charges = torch.tensor(0.0, **dd)

    options = dict(opts, **{"f_atol": tol**2, "x_atol": tol**2})
    calc = Calculator(numbers, par, opts=options, **dd)

    assert calc.integrals.hcore is not None
    h = calc.integrals.hcore.integral

    h.selfenergy.requires_grad_(True)
    h.kcn.requires_grad_(True)
    h.shpoly.requires_grad_(True)

    result = calc.singlepoint(positions, charges)
    energy = result.scf.sum(-1)

    (gradient,) = torch.autograd.grad(
        energy,
        positions,
        create_graph=True,
    )

    pgrad = torch.autograd.grad(
        gradient[0, :].sum(),
        (h.selfenergy, h.kcn, h.shpoly),
    )

    ref_se = load_from_npz(ref_grad_param, f"{name}_ggrad_selfenergy", dtype)
    assert pytest.approx(pgrad[0].cpu(), abs=tol) == ref_se.cpu()
    ref_kcn = load_from_npz(ref_grad_param, f"{name}_ggrad_kcn", dtype)
    assert pytest.approx(pgrad[1].cpu(), abs=tol) == ref_kcn.cpu()
    ref_shpoly = load_from_npz(ref_grad_param, f"{name}_ggrad_shpoly", dtype)
    assert pytest.approx(pgrad[2].cpu(), abs=tol) == ref_shpoly.cpu()

    positions.detach_()
