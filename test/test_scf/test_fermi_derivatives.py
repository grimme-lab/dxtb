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
Forces and Hessians with Fermi smearing against finite differences.

The differentiable Newton steps of the Fermi energy are required for exact
derivatives: without them, even the forces are wrong for small gaps and high
temperatures, and one step is not enough for the Hessian. The other SCF
derivative tests do not detect a missing step, hence, these systems are
chosen such that the occupations are strongly fractional. Each test also runs
with too few steps to show that it detects the problem.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, Calculator
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE

# (numbers, positions, unpaired electrons); stretched bonds for small gaps
SYSTEMS = {
    "C2": ([6, 6], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.9]], None),
    "O2": ([8, 8], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]], 2),
    "BeH2": (
        [4, 1, 1],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.5], [0.3, 0.0, -2.5]],
        None,
    ),
}

# step of the central differences (bohr)
STEP = 1e-4

# Measured errors of two steps: forces 4e-9 in all modes (dominated by the
# finite differences). Hessian of C2: 3e-9 (full, non-pure implicit and
# single-shot mode) and 4e-7 (implicit mode, its second derivative is less
# accurate). The limits contain a safety factor of 10 and more.
FORCES_ATOL = 5e-8
HESSIAN_ATOL = {"full": 5e-8, "implicit_nonpure": 5e-8, "implicit": 5e-6}

# Error with too few steps (zero for forces, one for Hessians), the test
# demands at least this much (measured: forces at least 7e-3, Hessian of C2
# 0.3 and of BeH2 2.8e-5).
TOO_FEW_FORCES = 1e-4
TOO_FEW_HESSIAN = {"C2": 1e-2, "BeH2": 1e-5}

# All SCF differentiation modes. The forces of the single-shot mode are exact
# since the energy is variational.
MODES = ["full", "implicit", "implicit_nonpure", "experimental"]


def _setup(name: str, ktemp: float, mode: str, steps: int | None = None):
    """
    Positions and energy function of a system. ``steps`` differentiable
    Newton steps of the Fermi energy are exact through order ``2**steps - 1``
    (``None`` keeps the default derivative order).
    """
    numbers, positions, spin = SYSTEMS[name]
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    opts = {
        "fermi_etemp": ktemp,
        "fermi_maxiter": 500,
        "fermi_thresh": 1e-12,
        "f_atol": 1e-10,
        "x_atol": 1e-10,
        "scf_mode": mode,
        "verbosity": 0,
    }
    if steps is not None:
        opts["fermi_diff_order"] = 2**steps - 1
    numbers_ = torch.tensor(numbers, device=DEVICE)

    def energy(pos: Tensor) -> Tensor:
        calc = Calculator(numbers_, GFN1_XTB, opts=opts, **dd)
        return calc.singlepoint(pos, spin=spin).total.sum(-1)

    return torch.tensor(positions, **dd), energy


def _gradient(energy, pos: Tensor, create_graph: bool = False) -> Tensor:
    (grad,) = torch.autograd.grad(energy(pos), pos, create_graph=create_graph)
    return grad


_REFERENCES: dict = {}


def _forces_reference(name: str, ktemp: float) -> Tensor:
    """Central differences of the energy."""
    key = ("forces", name, ktemp)
    if key not in _REFERENCES:
        pos, energy = _setup(name, ktemp, "implicit")
        ref = torch.zeros_like(pos)
        for i in range(pos.shape[0]):
            for j in range(3):
                d = torch.zeros_like(pos)
                d[i, j] = STEP
                ref[i, j] = (energy(pos + d) - energy(pos - d)) / (2 * STEP)
        _REFERENCES[key] = ref
    return _REFERENCES[key]


def _hessian_reference(name: str, ktemp: float) -> Tensor:
    """
    Central differences of the gradient (with the production number of
    steps, i.e., independent of the override in the tests).
    """
    key = ("hessian", name, ktemp)
    if key not in _REFERENCES:
        pos, energy = _setup(name, ktemp, "implicit")
        dd: DD = {"device": pos.device, "dtype": pos.dtype}
        ref = torch.zeros(pos.numel(), pos.numel(), **dd)
        for i in range(pos.numel()):
            d = torch.zeros(pos.numel(), **dd)
            d[i] = STEP
            d = d.view_as(pos)
            gp = _gradient(energy, (pos + d).requires_grad_())
            gm = _gradient(energy, (pos - d).requires_grad_())
            ref[i] = ((gp - gm) / (2 * STEP)).flatten()
        _REFERENCES[key] = ref
    return _REFERENCES[key]


def _forces_error(
    name: str, ktemp: float, mode: str, steps: int | None
) -> float:
    ref = _forces_reference(name, ktemp)
    pos, energy = _setup(name, ktemp, mode, steps)

    grad = _gradient(energy, pos.clone().requires_grad_())
    return (grad - ref).abs().max().item()


def _hessian_error(
    name: str, ktemp: float, mode: str, steps: int | None
) -> float:
    ref = _hessian_reference(name, ktemp)
    pos, energy = _setup(name, ktemp, mode, steps)

    p = pos.clone().requires_grad_()
    grad = _gradient(energy, p, create_graph=True)
    hess = torch.stack(
        [
            torch.autograd.grad(g, p, retain_graph=True)[0].flatten()
            for g in grad.flatten()
        ]
    )
    return (hess - ref).abs().max().item()


FORCES = [("C2", 300.0), ("C2", 5000.0), ("O2", 5000.0)]


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name, ktemp", FORCES)
def test_forces(name: str, ktemp: float, mode: str) -> None:
    """Forces agree with finite differences and need the Newton steps."""
    # production number of steps
    assert _forces_error(name, ktemp, mode, None) < FORCES_ATOL
    assert _forces_error(name, ktemp, mode, 1) < FORCES_ATOL

    # too few steps: the change of the Fermi energy is missing
    assert _forces_error(name, ktemp, mode, 0) > TOO_FEW_FORCES


# C2 in all modes, BeH2 (25000 K) only in the unrolled mode, see below
HESSIANS = [
    ("C2", 5000.0, "full"),
    ("C2", 5000.0, "implicit"),
    ("C2", 5000.0, "implicit_nonpure"),
    ("BeH2", 25000.0, "full"),
]


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name, ktemp, mode", HESSIANS)
def test_hessian(name: str, ktemp: float, mode: str) -> None:
    """The Hessian agrees with finite differences and needs two steps."""
    # production number of steps
    assert _hessian_error(name, ktemp, mode, None) < HESSIAN_ATOL[mode]

    # one step is exact for forces only
    assert _hessian_error(name, ktemp, mode, 1) > TOO_FEW_HESSIAN[name]


# The second derivative of these modes is not exact for BeH2 for reasons
# unrelated to the Fermi smearing: the error does not change with the number
# of steps (three or five steps give the same), and for the implicit mode it is
# 3.6e-3 without smearing (and at 300 K) as well. It is the linearly
# convergent case (single-shot: Bolte, Pauwels and Vaiter, Corollary 1) or the
# accuracy of the implicit second derivative. The measured errors must not
# grow by more than a factor of 10.
NOT_EXACT = {
    "implicit": 3.4e-4,
    "implicit_nonpure": 1.5e-5,
    "experimental": 5.4e-4,
}


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("mode", NOT_EXACT)
def test_hessian_not_exact(mode: str) -> None:
    """The inexact Hessians of BeH2 stay at their recorded errors."""
    assert _hessian_error("BeH2", 25000.0, mode, None) < 10 * NOT_EXACT[mode]
