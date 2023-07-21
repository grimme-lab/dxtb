"""
Test InteractionList.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import PotentialData
from dxtb.constants import defaults
from dxtb.interaction import Potential
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

data: PotentialData = {
    "mono": vmono.shape,
    "dipole": vdipole.shape,
    "quad": vquad.shape,
    "label": None,
}

datab: PotentialData = {
    "mono": vmonob.shape,
    "dipole": vdipoleb.shape,
    "quad": vquadb.shape,
    "label": None,
}

PAD = -9999


def test_astensor_empty() -> None:
    pot = Potential()

    with pytest.raises(RuntimeError):
        pot.as_tensor()


def test_astensor_mono() -> None:
    pot = Potential(vmono=vmono)
    tensor = pot.as_tensor()

    ref = batch.pack([vmono], value=defaults.PADNZ)
    assert (ref == tensor).all()


def test_astensor_mono_dipole() -> None:
    pot = Potential(vmono=vmono, vdipole=vdipole)
    tensor = pot.as_tensor()

    ref = batch.pack([vmono, vdipole], value=defaults.PADNZ)
    assert (ref == tensor).all()


def test_astensor_all() -> None:
    pot = Potential(vmono=vmono, vdipole=vdipole, vquad=vquad)
    tensor = pot.as_tensor()

    ref = batch.pack([vmono, vdipole, vquad], value=defaults.PADNZ)
    assert (ref == tensor).all()


# from tensor: single


def test_fromtensor_mono() -> None:
    pot = Potential.from_tensor(vmono, data)

    assert (pot.vmono == vmono).all()
    assert pot.vdipole is None
    assert pot.vquad is None


def test_fromtensor_mono_withpack() -> None:
    tensor = batch.pack([vmono], value=defaults.PADNZ)
    pot = Potential.from_tensor(tensor, data, batched=True)

    assert (pot.vmono == tensor).all()
    assert pot.vdipole is None
    assert pot.vquad is None


def test_fromtensor_mono_dipole() -> None:
    tensor = batch.pack([vmono, vdipole], value=defaults.PADNZ)
    pot = Potential.from_tensor(tensor, data)

    assert (pot.vmono == vmono).all()
    assert (pot.vdipole == vdipole).all()
    assert pot.vquad is None


def test_fromtensor_all() -> None:
    tensor = batch.pack([vmono, vdipole, vquad], value=defaults.PADNZ)
    pot = Potential.from_tensor(tensor, data, pad=defaults.PADNZ)

    assert (pot.vmono == vmono).all()
    assert (pot.vdipole == vdipole).all()
    assert (pot.vquad == vquad).all()


# from tensor: batched


def test_fromtensor_mono_batch() -> None:
    pot = Potential.from_tensor(vmonob, datab, batched=True)

    assert (pot.vmono == vmonob).all()
    assert pot.vdipole is None
    assert pot.vquad is None


def test_fromtensor_mono_batch_withpack() -> None:
    tensor = batch.pack([vmonob], value=defaults.PADNZ)
    pot = Potential.from_tensor(tensor, datab, batched=True)

    assert (pot.vmono == tensor).all()
    assert pot.vdipole is None
    assert pot.vquad is None


def test_fromtensor_mono_dipole_batch() -> None:
    tensor = batch.pack([vmonob, vdipoleb], value=PAD)
    pot = Potential.from_tensor(tensor, datab, batched=True, pad=PAD)

    assert (pot.vmono == vmonob).all()
    assert (pot.vdipole == vdipoleb).all()
    assert pot.vquad is None


def test_fromtensor_all_batch() -> None:
    tensor = batch.pack([vmonob, vdipoleb, vquadb], value=PAD)
    pot = Potential.from_tensor(tensor, datab, batched=True, pad=PAD)

    assert (pot.vmono == vmonob).all()
    assert (pot.vdipole == vdipoleb).all()
    assert (pot.vquad == vquadb).all()
