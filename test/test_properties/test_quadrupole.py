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
Quadrupole moment against the (GFN1-xTB) reference values, without and with an
electric field.

The analytical quadrupole moment reproduces all six reference components. The
derivative of the energy with respect to the electric field gradient (autograd
and finite differences) agrees for the diagonal elements only; its off-diagonal
elements differ by design, see ``test_quadrupole_fieldgrad.py``.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.units import VAA2AU

from dxtb import GFN1_XTB, Calculator
from dxtb._src.components.interactions import new_efield, new_efield_grad
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples

sample_list = [
    "H",
    "H2",
    "LiH",
    "HHe",
    "H2O",
    "CH4",
    "SiH4",
    "PbH4-BiH3",
    "MB16_43_01",
]

opts = {
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "f_atol": 1.0e-9,
    "x_atol": 1.0e-9,
    # an electric field alone only raises the integral level to the dipole
    "int_level": 4,
}

DIAG = [0, 2, 5]
"""Diagonal elements of the packed ``(xx, yx, yy, zx, zy, zz)`` quadrupole."""

# (reference key, electric field in Angstrom-based atomic units)
FIELDS = {
    "quadrupole": (0.0, 0.0, 0.0),
    "quadrupole2": (-2.0, 0.5, 1.5),
}


def _field(key: str, dd: DD) -> Tensor:
    return torch.tensor(FIELDS[key], **dd) * VAA2AU


def single(
    name: str,
    key: str,
    dd: DD,
    atol: float,
    rtol: float,
) -> None:
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    ref = samples[name][key].to(**dd)

    efield = new_efield(_field(key, dd))
    calc = Calculator(numbers, GFN1_XTB, interaction=[efield], opts=opts, **dd)

    quad = calc.get_quadrupole(positions).detach()

    assert pytest.approx(ref.cpu(), abs=atol, rel=rtol) == quad.cpu()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("key", FIELDS)
def test_single(dtype: torch.dtype, name: str, key: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    single(name, key, dd, atol=1e-3, rtol=1e-4)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["LYS_xao", "C60"])
@pytest.mark.parametrize("key", FIELDS)
def test_single_medium(dtype: torch.dtype, name: str, key: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    single(name, key, dd, atol=1e-2, rtol=1e-2)


def autograd_or_numerical(
    name: str, key: str, kind: str, dd: DD, atol: float
) -> None:
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    ref = samples[name][key].to(**dd)

    field_grad = torch.zeros((3, 3), **dd)
    efield = new_efield(_field(key, dd))

    if kind == "autograd":
        field_grad.requires_grad_(True)

    efg = new_efield_grad(field_grad)
    calc = Calculator(
        numbers, GFN1_XTB, interaction=[efield, efg], opts=opts, **dd
    )

    if kind == "autograd":
        quad = calc.quadrupole(positions).detach()
    else:
        quad = calc.quadrupole_numerical(positions, step_size=1e-4)

    assert (
        pytest.approx(ref[DIAG].cpu(), abs=atol, rel=1e-4) == quad[DIAG].cpu()
    )


# the field gradient derivative is covered in `test_quadrupole_fieldgrad.py`
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["LiH", "H2O"])
@pytest.mark.parametrize("key", FIELDS)
@pytest.mark.parametrize("kind", ["autograd", "numerical"])
def test_single_diagonal(
    dtype: torch.dtype, name: str, key: str, kind: str
) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    autograd_or_numerical(name, key, kind, dd, atol=1e-3)


def batched(
    name1: str,
    name2: str,
    key: str,
    dd: DD,
    atol: float,
    rtol: float,
    options: dict | None = None,
) -> None:
    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        [sample1["numbers"].to(DEVICE), sample2["numbers"].to(DEVICE)]
    )
    positions = pack(
        [sample1["positions"].to(**dd), sample2["positions"].to(**dd)]
    )
    ref = pack([sample1[key].to(**dd), sample2[key].to(**dd)])

    efield = new_efield(_field(key, dd))
    calc = Calculator(
        numbers,
        GFN1_XTB,
        interaction=[efield],
        opts=opts if options is None else options,
        **dd,
    )

    quad = calc.get_quadrupole(positions).detach()

    assert pytest.approx(ref.cpu(), abs=atol, rel=rtol) == quad.cpu()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["H", "H2O", "PbH4-BiH3"])
@pytest.mark.parametrize("key", FIELDS)
def test_batch(dtype: torch.dtype, name1: str, name2: str, key: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    batched(name1, name2, key, dd, atol=1e-3, rtol=1e-3)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["C60"])
@pytest.mark.parametrize("key", FIELDS)
def test_batch_medium(
    dtype: torch.dtype, name1: str, name2: str, key: str
) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    batched(name1, name2, key, dd, atol=1e-2, rtol=1e-2)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "H2O"])
@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_batch_settings(
    dtype: torch.dtype, name1: str, name2: str, scp_mode: str, mixer: str
) -> None:
    if scp_mode == "charge" and name2 == "H2O":
        # Pre-existing (also on `main`): multipole-level integrals, charge
        # mixing and a padded batch fail in `Potential.from_tensor`
        # (shape mismatch in `deflate`), independent of the quadrupole.
        pytest.xfail("charge mixing with multipoles in padded batches")

    dd: DD = {"dtype": dtype, "device": DEVICE}
    options = dict(opts, **{"scp_mode": scp_mode, "mixer": mixer})
    batched(
        name1, name2, "quadrupole", dd, atol=1e-4, rtol=1e-4, options=options
    )


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "LiH", "H2O"])
def test_batch_unconverged(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    # with 5 iterations, both do not converge, but pass the test
    options = dict(opts, **{"maxiter": 5, "mixer": "simple"})
    batched(
        name1, name2, "quadrupole", dd, atol=1e-4, rtol=1e-4, options=options
    )
