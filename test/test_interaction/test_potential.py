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
from tad_mctc.batch import pack

from dxtb._src.components.interactions import (
    Charges,
    ContainerLayout,
    Potential,
)
from dxtb._src.constants import defaults
from dxtb._src.typing import ContainerData

from ..conftest import DEVICE

nbatch = 10

# monopolar potential is orbital-resolved
vmono = torch.randn(6, device=DEVICE)
vmonob = torch.randn((nbatch, 6), device=DEVICE)

# multipolar potentials are atom-resolved
vdipole = torch.randn(2, device=DEVICE)
vdipoleb = torch.randn((nbatch, 2), device=DEVICE)
vquad = torch.randn(2, device=DEVICE)
vquadb = torch.randn((nbatch, 2), device=DEVICE)

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

    ref = pack([vmono, vdipole], value=defaults.PADNZ, axis=AXIS)
    assert ref.shape == tensor.shape
    assert (ref == tensor).all()


def test_astensor_all() -> None:
    pot = Potential(mono=vmono, dipole=vdipole, quad=vquad)
    tensor = pot.as_tensor()

    ref = pack([vmono, vdipole, vquad], value=defaults.PADNZ, axis=AXIS)
    assert ref.shape == tensor.shape
    assert (ref == tensor).all()


# from tensor: single


def test_fromtensor_mono() -> None:
    pot = Potential.from_tensor(vmono, data)

    assert (pot.mono == vmono).all()
    assert pot.dipole is None
    assert pot.quad is None


def test_fromtensor_mono_withpack() -> None:
    tensor = pack([vmono], value=defaults.PADNZ, axis=AXIS)
    pot = Potential.from_tensor(tensor, data, batch_mode=2)

    assert (pot.mono == tensor).all()
    assert pot.dipole is None
    assert pot.quad is None


def test_fromtensor_mono_dipole() -> None:
    tensor = pack([vmono, vdipole], value=defaults.PADNZ, axis=AXIS)
    pot = Potential.from_tensor(tensor, data)

    assert (pot.mono == vmono).all()
    assert (pot.dipole == vdipole).all()
    assert pot.quad is None


def test_fromtensor_all() -> None:
    tensor = pack([vmono, vdipole, vquad], value=defaults.PADNZ, axis=AXIS)
    pot = Potential.from_tensor(tensor, data, pad=defaults.PADNZ)

    assert (pot.mono == vmono).all()
    assert (pot.dipole == vdipole).all()
    assert (pot.quad == vquad).all()


# from tensor: batched


def test_fromtensor_mono_batch() -> None:
    pot = Potential.from_tensor(vmonob, datab, batch_mode=1)

    assert (pot.mono == vmonob).all()
    assert pot.dipole is None
    assert pot.quad is None


def test_fromtensor_mono_batch_withpack() -> None:
    tensor = pack([vmonob], value=defaults.PADNZ, axis=AXISB)

    # packing adds a dimension, so we must update the shape info
    localdata = datab.copy()
    localdata["mono"] = tensor.shape

    pot = Potential.from_tensor(tensor, localdata, batch_mode=1)

    assert (pot.mono == tensor).all()
    assert pot.dipole is None
    assert pot.quad is None


def test_fromtensor_mono_dipole_batch() -> None:
    tensor = pack([vmonob, vdipoleb], value=PAD, axis=AXISB)
    pot = Potential.from_tensor(tensor, datab, batch_mode=1, pad=PAD)

    assert (pot.mono == vmonob).all()
    assert (pot.dipole == vdipoleb).all()
    assert pot.quad is None


def test_fromtensor_all_batch() -> None:
    tensor = pack([vmonob, vdipoleb, vquadb], value=PAD, axis=AXISB)
    pot = Potential.from_tensor(tensor, datab, batch_mode=1, pad=PAD)

    assert (pot.mono == vmonob).all()
    assert (pot.dipole == vdipoleb).all()
    assert (pot.quad == vquadb).all()


@pytest.mark.parametrize("batch_mode", [0, 1, 2])
def test_layout_roundtrip_uhf(batch_mode: int) -> None:
    batch = () if batch_mode == 0 else (3,)
    mono = torch.randn((*batch, 2, 5), device=DEVICE)
    dipole = torch.randn((*batch, 2, 2, 3), device=DEVICE)
    quad = torch.randn((*batch, 2, 2, 6), device=DEVICE)
    charges = Charges(mono, dipole, quad, batch_mode=batch_mode, nspin=2)

    layout = charges.layout
    assert isinstance(layout, ContainerLayout)
    assert layout.nspin == 2

    restored = Charges.from_tensor(charges.as_tensor(), layout)
    assert restored.nspin == 2
    assert (restored.mono == mono).all()
    assert restored.dipole is not None
    assert (restored.dipole == dipole).all()
    assert restored.quad is not None
    assert (restored.quad == quad).all()


def test_layout_truncation_uhf() -> None:
    """
    Culling converged systems in a batched SCF truncates the serialized
    properties. For two spin channels, this must not mix up the channels.
    """
    mono = torch.randn((2, 2, 5), device=DEVICE)
    dipole = torch.randn((2, 2, 4, 3), device=DEVICE)
    charges = Charges(mono, dipole, batch_mode=1, nspin=2)

    # keep 3 of 5 orbitals and 2 of 4 atoms
    culled = Charges(
        mono[..., :3], dipole[..., :2, :], batch_mode=1, nspin=2
    ).layout

    tensor = charges.as_tensor()
    restored = Charges.from_tensor(tensor[..., : 2 * 2 * 3], culled)
    assert (restored.mono == mono[..., :3]).all()
    assert restored.dipole is not None
    assert (restored.dipole == dipole[..., :2, :]).all()


def test_select_and_embed_spin_channel() -> None:
    mono = torch.randn((2, 5), device=DEVICE)
    dipole = torch.randn((2, 3, 3), device=DEVICE)
    charges = Charges(mono, dipole, nspin=2)

    magnetization = charges.select_channel(1)
    assert magnetization.nspin == 1
    assert (magnetization.mono == mono[1]).all()
    assert magnetization.dipole is not None
    assert (magnetization.dipole == dipole[1]).all()

    embedded = magnetization.to_spin_channels(1)
    assert embedded.nspin == 2
    assert (embedded.mono[0] == 0.0).all()
    assert (embedded.mono[1] == mono[1]).all()

    # restricted containers have no magnetization channel
    with pytest.raises(RuntimeError):
        Charges(mono[0]).select_channel(1)


def test_add_uhf() -> None:
    left = Potential(torch.ones((2, 4), device=DEVICE), label="a", nspin=2)
    right = Potential(torch.ones((2, 4), device=DEVICE), label="b", nspin=2)

    result = left + right
    assert isinstance(result, Potential)
    assert result.nspin == 2
    assert result.label == ["a", "b"]

    with pytest.raises(ValueError):
        left += Potential(mono=torch.ones(4, device=DEVICE))
