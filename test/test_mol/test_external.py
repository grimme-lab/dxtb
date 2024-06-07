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
Test external molecular representations.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.molecule.container import Mol

from dxtb._src.typing import DD

try:
    from dxtb._src.exlibs import pyscf as _pyscf

    pyscf = True
except ImportError:
    pyscf = False

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "LiH", "S", "SiH4", "MB16_43_01", "C60"]


@pytest.mark.skipif(pyscf is False, reason="PySCF not installed")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_construction(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # pyscf molecule
    mol1 = _pyscf.mol.M(numbers, positions, parse_arg=False)

    # pyscf molecule from dxtb's molecule
    mol_dxtb = Mol(numbers, positions)
    mol2 = _pyscf.mol.PyscfMol.from_mol(mol_dxtb)

    for a1, a2 in zip(mol1.atom, mol2.atom):
        num1, pos1 = a1
        num2, pos2 = a2

        assert num1 == num2
        assert pos2.dtype == pos2.dtype  # type: ignore
        assert pytest.approx(pos1) == pos2


@pytest.mark.skipif(pyscf is False, reason="PySCF not installed")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_error(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([-1, 1], device=DEVICE)
    positions = torch.randn((2, 3), **dd)

    with pytest.raises(ValueError):
        _pyscf.mol.M(numbers, positions, parse_arg=False)
