"""
Test external molecular representations.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.mol import Mol

try:
    from dxtb.mol.external._pyscf import M, PyscfMol

    pyscf = True
except ImportError:
    pyscf = False

from .samples import samples

sample_list = ["H2", "LiH", "S", "SiH4", "MB16_43_01", "C60"]

device = None


@pytest.mark.skipif(pyscf is False, reason="PySCF not installed")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_construction(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    # pyscf molecule
    mol1 = M(numbers, positions, parse_arg=False)

    # pyscf molecule from dxtb's molecule
    mol_dxtb = Mol(numbers, positions)
    mol2 = PyscfMol.from_mol(mol_dxtb)

    for a1, a2 in zip(mol1.atom, mol2.atom):
        num1, pos1 = a1
        num2, pos2 = a2

        assert num1 == num2
        assert pos2.dtype == pos2.dtype  # type: ignore
        assert pytest.approx(pos1) == pos2


@pytest.mark.skipif(pyscf is False, reason="PySCF not installed")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_error(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    numbers = torch.tensor([-1, 1], device=device)
    positions = torch.randn((2, 3), **dd)

    with pytest.raises(ValueError):
        M(numbers, positions, parse_arg=False)
