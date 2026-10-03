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
GFN2-xTB (dipole and quadrupole integrals) with the PyTorch integral drivers
against the libcint driver: energies, forces, a padded batch, the Hessian and
the integral wrappers.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN2_XTB, Calculator, IndexHelper
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.pytorch.impls.kernels import ALGORITHMS
from dxtb._src.typing import DD, Tensor
from dxtb.integrals.wrappers import dipint, quadint
from dxtb.labels import INTDRIVER_LIBCINT, INTDRIVER_PYTORCH

from ..conftest import DEVICE
from ..utils import get_param_module
from .samples import samples

pytestmark = pytest.mark.skipif(
    not has_libcint, reason="libcint is the reference"
)

# libcint order -> pytorch order of the components within a shell
PERM_BY_L = {0: [0], 1: [1, 2, 0], 2: [0, 1, 2, 3, 4]}


def calculator(numbers: Tensor, opts: dict, dd: DD) -> Calculator:
    par = get_param_module("gfn2", **dd)
    return Calculator(numbers, par, opts={**opts, "verbosity": 0}, **dd)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("name", ["LiH", "SiH4"])
def test_energy_and_forces(name: str, algorithm: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    def energy_and_forces(opts: dict) -> tuple[Tensor, Tensor]:
        pos = positions.clone().requires_grad_(True)
        calc = calculator(numbers, opts, dd)
        return calc.get_energy(pos), calc.get_forces(pos)

    e_ref, f_ref = energy_and_forces({"int_driver": "libcint"})
    e, f = energy_and_forces(
        {"int_driver": "pytorch", "int_algorithm": algorithm}
    )

    assert torch.allclose(e, e_ref, atol=1e-11, rtol=0.0)
    assert torch.allclose(f, f_ref, atol=1e-11, rtol=0.0)


def test_batch() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    names = ["SiH4", "H2O"]
    numbers = pack([samples[n]["numbers"].to(DEVICE) for n in names])
    positions = pack([samples[n]["positions"].to(**dd) for n in names])

    def energy(opts: dict) -> Tensor:
        return calculator(numbers, opts, dd).get_energy(positions)

    e_ref = energy({"int_driver": "libcint"})
    e = energy({"int_driver": "autograd"})
    assert torch.allclose(e, e_ref, atol=1e-11, rtol=0.0)


@pytest.mark.grad
def test_hessian() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["LiH"]["numbers"].to(DEVICE)
    positions = samples["LiH"]["positions"].to(**dd)

    def hessian(opts: dict) -> Tensor:
        pos = positions.clone().requires_grad_(True)
        return calculator(numbers, opts, dd).get_hessian(pos)

    h_ref = hessian({"int_driver": "libcint"})
    h = hessian({"int_driver": "autograd"})
    assert torch.allclose(h, h_ref, atol=1e-11, rtol=0.0)


def test_integral_wrappers() -> None:
    """``dipint`` and ``quadint`` with the PyTorch driver, including the
    shared post-processing of the quadrupole."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["SiH4"]["numbers"].to(DEVICE)
    positions = samples["SiH4"]["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN2_XTB)
    orbs = ihelp.orbitals_per_shell.tolist()
    p = torch.zeros((sum(orbs), sum(orbs)), **dd)
    off = 0
    for l, nao in zip(ihelp.angular.tolist(), orbs):
        for k, pk in enumerate(PERM_BY_L[l]):
            p[off + k, off + pk] = 1.0
        off += nao

    for fn in (dipint, quadint):
        ref = fn(numbers, positions, GFN2_XTB, driver=INTDRIVER_LIBCINT)
        out = fn(numbers, positions, GFN2_XTB, driver=INTDRIVER_PYTORCH)
        assert torch.allclose(out, p @ ref.to(DEVICE) @ p.mT, atol=1e-11)
