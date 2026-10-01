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
Test the gradients of spin-polarized calculations.

Reference gradients obtained with tblite 0.7.0 (see `samples.py`).
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, IndexHelper
from dxtb._src.components.interactions.spin import new_spin_polarization
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor
from dxtb.calculators import AnalyticalCalculator

from ..conftest import DEVICE, NONDET_TOL
from ..utils import load_from_npz
from .samples import samples

ref_grad = np.load(Path(__file__).parent / "grad.npz")

opts = {"verbosity": 0}


@pytest.mark.grad
@pytest.mark.parametrize("name", ["H2+", "OH"])
def test_grad_charges(name: str) -> None:
    """Gradient with respect to the magnetization charges."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    qsh = sample["qshgfn1"].to(**dd)[1].detach()

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    spin = new_spin_polarization(numbers, **dd)
    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)

    # variable to be differentiated
    q = qsh.clone().requires_grad_(True)

    def func(q: Tensor) -> Tensor:
        return spin.get_monopole_shell_energy(cache, q)

    assert dgradcheck(func, q, nondet_tol=NONDET_TOL)

    # variables are detached after the check
    q.requires_grad_(True)
    assert dgradgradcheck(func, q, nondet_tol=NONDET_TOL)


@pytest.mark.grad
@pytest.mark.parametrize("name", ["H2+", "OH"])
def test_grad_param(name: str) -> None:
    """Gradient with respect to the scaling of the spin constants."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    qsh = sample["qshgfn1"].to(**dd)[1]

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    # variable to be differentiated
    wscale = torch.tensor(1.0, **dd, requires_grad=True)

    def func(wscale: Tensor) -> Tensor:
        spin = new_spin_polarization(numbers, wscale=wscale, **dd)
        cache = spin.get_cache(numbers=numbers, ihelp=ihelp)
        return spin.get_monopole_shell_energy(cache, qsh)

    assert dgradcheck(func, wscale, nondet_tol=NONDET_TOL)


@pytest.mark.grad
def test_grad_param_cache() -> None:
    """Differentiable cache is rebuilt after its graph was freed."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    numbers = samples["OH"]["numbers"].to(DEVICE)
    qsh = samples["OH"]["qshgfn1"].to(**dd)[1]
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    wscale = torch.tensor(1.0, **dd, requires_grad=True)
    spin = new_spin_polarization(numbers, wscale=wscale, **dd)

    for _ in range(2):
        cache = spin.get_cache(numbers=numbers, ihelp=ihelp)
        e = spin.get_monopole_shell_energy(cache, qsh).sum()
        (g,) = torch.autograd.grad(e, wscale)
        assert pytest.approx(e.item()) == g.item()


@pytest.mark.grad
@pytest.mark.parametrize("scf_mode", ["implicit", "full"])
def test_grad_param_scf(scf_mode: str) -> None:
    """Gradient of the SCF energy with respect to the scaling."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    tol = sqrt(torch.finfo(torch.double).eps) * 10

    sample = samples["OH"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    spin = sample["spin"].to(**dd)

    # variable to be differentiated
    wscale = torch.tensor(1.0, **dd, requires_grad=True)

    def func(wscale: Tensor) -> Tensor:
        spinpol = new_spin_polarization(numbers, wscale=wscale, **dd)
        _opts = {**opts, "scf_mode": scf_mode, "x_atol": tol, "f_atol": tol}
        calc = Calculator(
            numbers, GFN1_XTB, interaction=[spinpol], opts=_opts, **dd
        )
        return calc.energy(positions, spin=spin)

    assert dgradcheck(func, wscale, atol=tol, nondet_tol=NONDET_TOL)


def gradient(dtype: torch.dtype, name: str, gfn: str, mode: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    atol, rtol = 1e-5, 1e-4

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).detach()
    chrg = sample["charge"].to(**dd)
    spin = sample["spin"].to(**dd)
    ref = load_from_npz(ref_grad, f"{name}_{gfn}{mode}", dtype)

    par = GFN1_XTB if gfn == "gfn1" else GFN2_XTB
    if mode == "_uhf":
        calc = Calculator(numbers, par, opts={**opts, "uhf_mode": True}, **dd)
    else:
        spinpol = new_spin_polarization(numbers, **dd)
        calc = Calculator(numbers, par, interaction=[spinpol], opts=opts, **dd)

    pos = positions.clone().requires_grad_(True)
    energy = calc.energy(pos, chrg, spin)
    (grad,) = torch.autograd.grad(energy, pos)

    assert pytest.approx(ref.cpu(), abs=atol, rel=rtol) == grad.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["H2+", "OH"])
@pytest.mark.parametrize("mode", ["", "_uhf"])
def test_gradient_gfn1(dtype: torch.dtype, name: str, mode: str) -> None:
    gradient(dtype, name, "gfn1", mode)


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["LiH", "OH", "SiH4", "C2H5O"])
@pytest.mark.parametrize("mode", ["", "_uhf"])
def test_gradient_gfn2(dtype: torch.dtype, name: str, mode: str) -> None:
    gradient(dtype, name, "gfn2", mode)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2+", "OH"])
@pytest.mark.parametrize("mode", ["", "_uhf"])
def test_analytical(dtype: torch.dtype, name: str, mode: str) -> None:
    """Analytical GFN1-xTB forces agree with autograd."""
    dd: DD = {"dtype": dtype, "device": DEVICE}
    atol = rtol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).detach()
    chrg = sample["charge"].to(**dd)
    spin = sample["spin"].to(**dd)

    kwargs = {}
    _opts = {**opts, "scf_mode": "full", "uhf_mode": mode == "_uhf"}
    if mode == "":
        kwargs["interaction"] = [new_spin_polarization(numbers, **dd)]

    calc = Calculator(numbers, GFN1_XTB, opts=_opts, **kwargs, **dd)
    pos = positions.clone().requires_grad_(True)
    ref = calc.forces(pos, chrg, spin)

    calc = AnalyticalCalculator(numbers, GFN1_XTB, opts=_opts, **kwargs, **dd)
    pos = positions.clone().requires_grad_(True)
    forces = calc.forces_analytical(pos, chrg, spin)

    assert (
        pytest.approx(ref.detach().cpu(), abs=atol, rel=rtol)
        == forces.detach().cpu()
    )


@pytest.mark.grad
def test_hessian_closed_shell() -> None:
    """The UHF Hessian of a closed-shell molecule equals the restricted one."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    tol = sqrt(torch.finfo(torch.double).eps) * 10

    numbers = samples["OH"]["numbers"].to(DEVICE)
    positions = samples["OH"]["positions"].to(**dd).detach()
    chrg = torch.tensor(-1.0, **dd)  # hydroxide anion

    _opts = {**opts, "scf_mode": "full"}
    calc = Calculator(numbers, GFN1_XTB, opts=_opts, **dd)
    pos = positions.clone().requires_grad_(True)
    ref = calc.hessian(pos, chrg)

    spinpol = new_spin_polarization(numbers, **dd)
    calc = Calculator(
        numbers, GFN1_XTB, interaction=[spinpol], opts=_opts, **dd
    )
    pos = positions.clone().requires_grad_(True)
    hessian = calc.hessian(pos, chrg)

    assert pytest.approx(ref.detach().cpu(), abs=tol) == hessian.detach().cpu()
