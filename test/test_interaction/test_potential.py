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
Test InteractionList.
"""

from __future__ import annotations

import pytest
import torch

from dxtb.components.interactions import Potential
from dxtb.constants import defaults
from dxtb.typing import ContainerData
from dxtb.utils import batch

nbatch = 10

# monopolar potential is orbital-resolved
vmono = torch.randn(6)
vmonob = torch.randn((nbatch, 6))

# multipolar potentials are atom-resolved
vdipole = torch.randn(2)
vdipoleb = torch.randn((nbatch, 2))
vquad = torch.randn(2)
vquadb = torch.randn((nbatch, 2))

data: ContainerData = {
    "mono": vmono.shape,
    "dipole": vdipole.shape,
    "quad": vquad.shape,
    "label": None,
}

datab: ContainerData = {
    "mono": vmonob.shape,
    "dipole": vdipoleb.shape,
    "quad": vquadb.shape,
    "label": None,
}

PAD = -9999
AXIS = 0
AXISB = 1


def test_astensor_empty() -> None:
    pot = Potential()

    with pytest.raises(RuntimeError):
        pot.as_tensor()


def test_astensor_mono() -> None:
    pot = Potential(mono=vmono)
    tensor = pot.as_tensor()

    # multipole dimension is always present after `as_tensor`
    _vmono = vmono.unsqueeze(-2)

    assert _vmono.shape == tensor.shape
    assert (_vmono == tensor).all()


def test_astensor_mono_dipole() -> None:
    pot = Potential(mono=vmono, dipole=vdipole)
    tensor = pot.as_tensor()

    ref = batch.pack([vmono, vdipole], value=defaults.PADNZ, axis=AXIS)
    assert ref.shape == tensor.shape
    assert (ref == tensor).all()


def test_astensor_all() -> None:
    pot = Potential(mono=vmono, dipole=vdipole, quad=vquad)
    tensor = pot.as_tensor()

    ref = batch.pack([vmono, vdipole, vquad], value=defaults.PADNZ, axis=AXIS)
    assert ref.shape == tensor.shape
    assert (ref == tensor).all()


# from tensor: single


def test_fromtensor_mono() -> None:
    pot = Potential.from_tensor(vmono, data)

    assert (pot.mono == vmono).all()
    assert pot.dipole is None
    assert pot.quad is None


def test_fromtensor_mono_withpack() -> None:
    tensor = batch.pack([vmono], value=defaults.PADNZ, axis=AXIS)
    pot = Potential.from_tensor(tensor, data, batched=True)

    assert (pot.mono == tensor).all()
    assert pot.dipole is None
    assert pot.quad is None


def test_fromtensor_mono_dipole() -> None:
    tensor = batch.pack([vmono, vdipole], value=defaults.PADNZ, axis=AXIS)
    pot = Potential.from_tensor(tensor, data)

    assert (pot.mono == vmono).all()
    assert (pot.dipole == vdipole).all()
    assert pot.quad is None


def test_fromtensor_all() -> None:
    tensor = batch.pack([vmono, vdipole, vquad], value=defaults.PADNZ, axis=AXIS)
    pot = Potential.from_tensor(tensor, data, pad=defaults.PADNZ)

    assert (pot.mono == vmono).all()
    assert (pot.dipole == vdipole).all()
    assert (pot.quad == vquad).all()


# from tensor: batched


def test_fromtensor_mono_batch() -> None:
    pot = Potential.from_tensor(vmonob, datab, batched=True)

    assert (pot.mono == vmonob).all()
    assert pot.dipole is None
    assert pot.quad is None


def test_fromtensor_mono_batch_withpack() -> None:
    tensor = batch.pack([vmonob], value=defaults.PADNZ, axis=AXISB)

    # packing adds a dimension, so we must update the shape info
    localdata = datab.copy()
    localdata["mono"] = tensor.shape

    pot = Potential.from_tensor(tensor, localdata, batched=True)

    assert (pot.mono == tensor).all()
    assert pot.dipole is None
    assert pot.quad is None


def test_fromtensor_mono_dipole_batch() -> None:
    tensor = batch.pack([vmonob, vdipoleb], value=PAD, axis=AXISB)
    pot = Potential.from_tensor(tensor, datab, batched=True, pad=PAD)

    assert (pot.mono == vmonob).all()
    assert (pot.dipole == vdipoleb).all()
    assert pot.quad is None


def test_fromtensor_all_batch() -> None:
    tensor = batch.pack([vmonob, vdipoleb, vquadb], value=PAD, axis=AXISB)
    pot = Potential.from_tensor(tensor, datab, batched=True, pad=PAD)

    assert (pot.mono == vmonob).all()
    assert (pot.dipole == vdipoleb).all()
    assert (pot.quad == vquadb).all()
