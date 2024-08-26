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
Test spread and reduce capabilities of IndexHelper with extra dimension.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import IndexHelper

from ..conftest import DEVICE


def test_spread() -> None:
    numbers = torch.tensor([14, 1, 1], device=DEVICE)
    angular = {
        1: [0, 0],  # H
        14: [0, 1, 2],  # Si
    }
    ihelp = IndexHelper.from_numbers_angular(numbers, angular)

    # 13 orbitals = Si3s + 3*Si3p + 5*Si3d + 2*(H1s + H2s)
    nao = 13
    nsh = 7
    nat = 3

    # atom to shell
    x = torch.randn((nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nsh, 3))

    x = torch.randn((nat, nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_shell(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nsh, nat, 3))

    x = torch.randn((nat, nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nat, nsh, 3))

    # orbital to atom
    x = torch.randn((nat, nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_orbital(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nao, nat, 3))

    x = torch.randn((nat, nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_orbital(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nat, nao, 3))


def test_spread_batch() -> None:
    angular = {
        1: [0, 0],  # H
        14: [0, 1, 2],  # Si
    }
    numbers = torch.tensor([[14, 1, 1], [14, 1, 0]], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, angular)

    # 13 orbitals = Si3s + 3*Si3p + 5*Si3d + 2*(H1s + H2s)
    nao = 13
    nsh = 7
    nat = 3
    nbatch = 2

    # nat to shell
    x = torch.randn((nbatch, nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nsh, 3))

    x = torch.randn((nbatch, nat, nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_shell(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nbatch, nsh, nat, 3))

    x = torch.randn((nbatch, nat, nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nat, nsh, 3))

    # atom to orbital
    x = torch.randn((nbatch, nat, nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_orbital(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nbatch, nao, nat, 3))

    x = torch.randn((nbatch, nat, nat, 3), device=DEVICE)
    out = ihelp.spread_atom_to_orbital(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nat, nao, 3))


def test_spread_unique() -> None:
    numbers = torch.tensor([14, 1, 1], device=DEVICE)
    angular = {
        1: [0, 0],  # H
        14: [0, 1, 2],  # Si
    }
    ihelp = IndexHelper.from_numbers_angular(numbers, angular)

    # 13 orbitals = Si3s + 3*Si3p + 5*Si3d + 2*(H1s + H2s)
    nao = 13
    nsh = 7
    nat = 3
    nat_u = 2  # unique atoms
    nsh_u = 5  # unique shells

    # unique species to atom
    x = torch.randn((nat_u, 3), device=DEVICE)
    out = ihelp.spread_uspecies_to_atom(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nat, 3))
    out = ihelp.spread_uspecies_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nsh, 3))
    out = ihelp.spread_uspecies_to_orbital(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nao, 3))

    # unique shell to shell
    x = torch.randn((nsh_u, 3), device=DEVICE)
    out = ihelp.spread_ushell_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nsh, 3))

    x = torch.randn((nsh_u, nat, 3), device=DEVICE)
    out = ihelp.spread_ushell_to_shell(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nsh, nat, 3))

    # unique shell to orbital
    x = torch.randn((nsh_u, 3), device=DEVICE)
    out = ihelp.spread_ushell_to_orbital(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nao, 3))

    x = torch.randn((nsh_u, nat, 3), device=DEVICE)
    out = ihelp.spread_ushell_to_orbital(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nao, nat, 3))


@pytest.mark.xfail
def test_spread_unique_batch() -> None:
    """
    Spreading batched unique stuff does not work properly!
    """
    angular = {
        1: [0, 0],  # H
        14: [0, 1, 2],  # Si
    }
    numbers = torch.tensor([[14, 1, 1], [14, 1, 0]], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, angular)

    # 13 orbitals = Si3s + 3*Si3p + 5*Si3d + 2*(H1s + H2s)
    nao = 13
    nsh = 7
    nat = 3
    nat_u = 2  # unique atoms
    nsh_u = 5  # unique shells
    nbatch = 2

    # unique species to atom
    x = torch.randn((nbatch, nat_u, 3), device=DEVICE)

    # pollutes CUDA memory
    if DEVICE is not None:
        assert False

    out = ihelp.spread_uspecies_to_atom(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nat, 3))

    out = ihelp.spread_uspecies_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nsh, 3))

    out = ihelp.spread_uspecies_to_orbital(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nao, 3))

    # unique shell to shell
    x = torch.randn((nbatch, nsh_u, 3), device=DEVICE)
    out = ihelp.spread_ushell_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nsh, 3))

    x = torch.randn((nbatch, nsh_u, nat, 3), device=DEVICE)
    out = ihelp.spread_ushell_to_shell(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nbatch, nsh, nat, 3))

    # unique shell to orbital
    x = torch.randn((nbatch, nsh_u, 3), device=DEVICE)
    out = ihelp.spread_ushell_to_orbital(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nao, 3))

    x = torch.randn((nbatch, nsh_u, nat, 3), device=DEVICE)
    out = ihelp.spread_ushell_to_orbital(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nbatch, nao, nat, 3))


##########################################################


def test_reduce() -> None:
    numbers = torch.tensor([14, 1, 1], device=DEVICE)
    angular = {
        1: [0, 0],  # H
        14: [0, 1, 2],  # Si
    }
    ihelp = IndexHelper.from_numbers_angular(numbers, angular)

    # 13 orbitals = Si3s + 3*Si3p + 5*Si3d + 2*(H1s + H2s)
    nao = 13
    nsh = 7
    nat = 3

    # orbital to shell
    x = torch.randn((nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nsh, 3))

    x = torch.randn((nao, nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_shell(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nsh, nao, 3))

    x = torch.randn((nao, nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nao, nsh, 3))

    # orbital to atom
    x = torch.randn((nao, nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_atom(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nat, nao, 3))

    x = torch.randn((nao, nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_atom(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nao, nat, 3))


def test_reduce_batch() -> None:
    angular = {
        1: [0, 0],  # H
        14: [0, 1, 2],  # Si
    }
    numbers = torch.tensor([[14, 1, 1], [14, 1, 0]], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, angular)

    # 13 orbitals = Si3s + 3*Si3p + 5*Si3d + 2*(H1s + H2s)
    nao = 13
    nsh = 7
    nat = 3
    nbatch = 2

    # orbital to shell
    x = torch.randn((nbatch, nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nsh, 3))

    x = torch.randn((nbatch, nao, nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_shell(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nbatch, nsh, nao, 3))

    x = torch.randn((nbatch, nao, nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_shell(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nao, nsh, 3))

    # orbital to atom
    x = torch.randn((nbatch, nao, nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_atom(x, dim=-3, extra=True)
    assert out.shape == torch.Size((nbatch, nat, nao, 3))

    x = torch.randn((nbatch, nao, nao, 3), device=DEVICE)
    out = ihelp.reduce_orbital_to_atom(x, dim=-2, extra=True)
    assert out.shape == torch.Size((nbatch, nao, nat, 3))


##########################################################


def test_fail() -> None:
    angular = {
        1: [0, 0],  # H
        14: [0, 1, 2],  # Si
    }
    numbers = torch.tensor([[14, 1, 1], [14, 1, 0]], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, angular)

    # last dimension is reserved
    with pytest.raises(ValueError):
        x = torch.randn((2, 13, 3), device=DEVICE)
        ihelp.reduce_orbital_to_shell(x, dim=-1, extra=True)

    # no tuples allowed
    with pytest.raises(TypeError):
        x = torch.randn((2, 13, 3), device=DEVICE)
        ihelp.reduce_orbital_to_shell(x, dim=(-1, -2), extra=True)

    # source tensor has too few dimensions
    with pytest.raises(RuntimeError):
        x = torch.randn(3, device=DEVICE)
        ihelp.reduce_orbital_to_shell(x, dim=-2, extra=True)

    # source tensor has too many dimensions
    with pytest.raises(NotImplementedError):
        x = torch.randn((2, 2, 2, 13, 3), device=DEVICE)
        ihelp.reduce_orbital_to_shell(x, dim=-2, extra=True)
