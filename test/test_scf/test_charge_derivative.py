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
"""
Fractional charges and the derivative of the energy w.r.t. the total charge.

At finite electronic temperature, the number of electrons used to be rounded
to integers in every SCF step. Fractional charges then gave the energy of the
nearest integer charge, and dE/dQ was zero (``full``) or raised an error
(``implicit``). The electron count is now fixed at the setup and not rounded,
hence, the energy varies smoothly with Q and dE/dQ equals the negative
chemical potential (GFN1-xTB only: the charge-dependent D4 term of GFN2 breaks
this identity). Charges at half-integer electron totals (H2O: Q = 0.5, 1.5)
are avoided, as the alpha/beta split changes discontinuously there.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.data.molecules import mols
from tad_mctc.units.energy import KELVIN2AU

from dxtb import GFN1_XTB, Calculator
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE

MODES = ["full", "implicit", "experimental"]
GUESSES = ["eeq", "sad"]

C2 = ([6, 6], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.9]])

# step of the central differences
STEP = 1e-3

# Measured errors: central differences 8.3e-9 (C2, all modes and guesses;
# H2O 1.8e-10; 0 K 1.5e-9), -mu 1.5e-12. The limits are 10x the maximum
# (-mu: rounded up to stay clear of rounding noise on other hardware).
FD_ATOL = 1e-7
MU_ATOL = 1e-10


def _system(name: str, dd: DD) -> tuple[Tensor, Tensor]:
    if name == "C2":
        numbers = torch.tensor(C2[0], device=dd["device"])
        pos = torch.tensor(C2[1], **dd)
    else:
        numbers = mols[name]["numbers"].to(dd["device"])
        pos = mols[name]["positions"].to(**dd)
    return numbers, pos


def _calc(
    name: str, etemp: float, mode: str, guess: str = "eeq"
) -> tuple[Calculator, Tensor]:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, pos = _system(name, dd)
    options = {
        "fermi_etemp": etemp,
        "fermi_maxiter": 500,
        "fermi_thresh": 1e-12,
        "f_atol": 1e-10,
        "x_atol": 1e-10,
        "scf_mode": mode,
        "guess": guess,
        "verbosity": 0,
    }
    return Calculator(numbers, GFN1_XTB, opts=options, **dd), pos


def _energy(name: str, etemp: float, mode: str, guess: str, q: Tensor):
    # a new calculator for every energy: results are cached per calculator
    calc, pos = _calc(name, etemp, mode, guess)
    return calc.singlepoint(pos, chrg=q).total.sum(-1)


def _grad_and_fd(
    name: str, etemp: float, mode: str, guess: str, q0: float
) -> tuple[Tensor, Tensor]:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    q = torch.tensor(q0, **dd, requires_grad=True)
    (grad,) = torch.autograd.grad(_energy(name, etemp, mode, guess, q), q)

    ep = _energy(name, etemp, mode, guess, torch.tensor(q0 + STEP, **dd))
    em = _energy(name, etemp, mode, guess, torch.tensor(q0 - STEP, **dd))
    return grad, (ep - em) / (2 * STEP)


def test_no_plateau() -> None:
    """Energy increases strictly with fractional charge (no rounding)."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    qs = [0.0, 0.1, 0.2, 0.3, 0.4]
    e = [
        _energy("H2O", 300.0, "full", "eeq", torch.tensor(q, **dd)).item()
        for q in qs
    ]
    assert all(b > a for a, b in zip(e[:-1], e[1:])), e


@pytest.mark.parametrize("guess", GUESSES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "name,etemp,q0",
    [("H2O", 300.0, 0.3), ("C2", 5000.0, 0.0), ("C2", 5000.0, 0.2)],
)
def test_gradient_fd(
    name: str, etemp: float, q0: float, mode: str, guess: str
) -> None:
    """dE/dQ at finite temperature matches central differences."""
    grad, fd = _grad_and_fd(name, etemp, mode, guess, q0)
    assert pytest.approx(fd.cpu(), abs=FD_ATOL) == grad.cpu()


def test_chemical_potential() -> None:
    """dE/dQ equals the negative chemical potential of the Fermi level."""
    etemp = 300.0
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    calc, pos = _calc("H2O", etemp, "full")
    q = torch.tensor(0.3, **dd, requires_grad=True)
    res = calc.singlepoint(pos, chrg=q)
    (grad,) = torch.autograd.grad(res.total.sum(-1), q)

    # closed shell: alpha channel, partially filled orbital
    f = res.occupation[0].detach()
    i = torch.argmin((f - 0.5).abs())
    assert 1e-8 < f[i] < 1 - 1e-8

    kt = etemp * KELVIN2AU
    mu = res.emo[i].detach() - kt * torch.log(1 / f[i] - 1)
    assert pytest.approx(-mu.cpu(), abs=MU_ATOL) == grad.cpu()


@pytest.mark.parametrize("mode", ["full", "implicit"])
def test_zero_kelvin(mode: str) -> None:
    """The derivative w.r.t. the charge at 0 K is unchanged."""
    grad, fd = _grad_and_fd("H2O", 0.0, mode, "eeq", 0.3)
    assert pytest.approx(fd.cpu(), abs=FD_ATOL) == grad.cpu()


def test_batch_culling() -> None:
    """Batched fractional charges (with culling) equal the single runs."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    names = ["H2", "H2O", "LYS_xao"]
    chrg = [0.0, 0.3, 0.0]

    # systems converge at different iterations, so that `cull` actually runs
    numbers = torch.nn.utils.rnn.pad_sequence(
        [mols[n]["numbers"] for n in names], batch_first=True
    ).to(DEVICE)
    pos = torch.nn.utils.rnn.pad_sequence(
        [mols[n]["positions"] for n in names], batch_first=True
    ).to(**dd)

    opts = {
        "fermi_etemp": 300.0,
        "fermi_maxiter": 500,
        "fermi_thresh": 1e-12,
        "f_atol": 1e-10,
        "x_atol": 1e-10,
        "scf_mode": "full",
        "batch_mode": 1,
        "verbosity": 0,
    }
    calc = Calculator(numbers, GFN1_XTB, opts=opts, **dd)
    batch = calc.singlepoint(pos, chrg=torch.tensor(chrg, **dd)).total.sum(-1)

    for i, (name, q) in enumerate(zip(names, chrg)):
        e = _energy(name, 300.0, "full", "eeq", torch.tensor(q, **dd))
        assert pytest.approx(e.cpu(), abs=1e-8) == batch[i].cpu()
