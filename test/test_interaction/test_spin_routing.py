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
Test the routing of the charge and magnetization channels to the
interactions in unrestricted (UHF) calculations.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import IndexHelper
from dxtb._src.components.interactions import (
    ChargeChannel,
    Charges,
    Interaction,
    InteractionCache,
    InteractionList,
)
from dxtb._src.typing import Tensor, TensorOrTensors

from ..conftest import DEVICE


class TotalProbe(Interaction):
    """Quadratic interaction that exposes every routed total-charge input."""

    label = "TotalProbe"

    def get_monopole_atom_energy(
        self, cache: InteractionCache, qat: Tensor, **_: object
    ) -> Tensor:
        return 0.5 * qat.square()

    def get_monopole_atom_potential(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        return qat

    def get_dipole_atom_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        assert qdp is not None
        return 0.5 * qdp.square().sum(-1)

    def get_dipole_atom_potential(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor | None:
        return qdp

    def get_quadrupole_atom_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        assert qqp is not None
        return 0.5 * qqp.square().sum(-1)

    def get_quadrupole_atom_potential(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor | None:
        return qqp

    def get_atom_gradient(
        self,
        charges: Tensor,
        positions: Tensor,
        cache: InteractionCache,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        return charges.unsqueeze(-1).expand_as(positions)


class MagnetizationProbe(Interaction):
    """Quadratic monopole interaction consuming only magnetization."""

    label = "MagnetizationProbe"
    charge_channel = ChargeChannel.MAGNETIZATION
    requires_uhf = True

    def get_monopole_atom_energy(
        self, cache: InteractionCache, qat: Tensor, **_: object
    ) -> Tensor:
        return 0.5 * qat.square()

    def get_monopole_atom_potential(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        return qat

    def get_atom_gradient(
        self,
        charges: Tensor,
        positions: Tensor,
        cache: InteractionCache,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        return charges.unsqueeze(-1).expand_as(positions)


def setup_system() -> tuple[Tensor, Tensor, IndexHelper, Charges]:
    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.zeros((2, 3), device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    mono = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
    dipole = torch.arange(12.0, device=DEVICE).reshape(2, 2, 3)
    quad = torch.arange(24.0, device=DEVICE).reshape(2, 2, 6)
    charges = Charges(mono, dipole, quad, nspin=2)
    return numbers, positions, ihelp, charges


def test_two_channel_energy_and_decomposition() -> None:
    numbers, positions, ihelp, charges = setup_system()
    interactions = InteractionList(TotalProbe(), MagnetizationProbe())
    cache = interactions.get_cache(numbers, positions, ihelp)

    total = charges.select_channel(0)
    magnetization = charges.select_channel(1)
    expected_total = 0.5 * total.mono.square()
    assert total.dipole is not None
    expected_total += 0.5 * total.dipole.square().sum(-1)
    assert total.quad is not None
    expected_total += 0.5 * total.quad.square().sum(-1)
    expected_magnetization = 0.5 * magnetization.mono.square()

    energy = interactions.get_energy(charges, cache, ihelp)
    decomposition = interactions.get_energy_as_dict(charges, cache, ihelp)

    assert torch.equal(energy, expected_total + expected_magnetization)
    assert torch.equal(decomposition["TotalProbe"], expected_total)
    assert torch.equal(
        decomposition["MagnetizationProbe"], expected_magnetization
    )


def test_two_channel_potential_isolation_including_multipoles() -> None:
    numbers, positions, ihelp, charges = setup_system()
    interactions = InteractionList(TotalProbe(), MagnetizationProbe())
    cache = interactions.get_cache(numbers, positions, ihelp)

    potential = interactions.get_potential(cache, charges, ihelp)

    assert potential.nspin == 2
    assert potential.mono is not None
    assert torch.equal(potential.mono, charges.mono)
    assert potential.dipole is not None
    assert charges.dipole is not None
    assert torch.equal(potential.dipole[0], charges.dipole[0])
    assert torch.equal(potential.dipole[1], torch.zeros_like(charges.dipole[1]))
    assert potential.quad is not None
    assert charges.quad is not None
    assert torch.equal(potential.quad[0], charges.quad[0])
    assert torch.equal(potential.quad[1], torch.zeros_like(charges.quad[1]))
    assert set(potential.label) == {"TotalProbe", "MagnetizationProbe"}


def test_two_channel_gradient_routing() -> None:
    numbers, positions, ihelp, charges = setup_system()
    interactions = InteractionList(TotalProbe(), MagnetizationProbe())
    cache = interactions.get_cache(numbers, positions, ihelp)

    gradient = interactions.get_gradient(charges, positions, cache, ihelp)
    expected = (charges.mono[0] + charges.mono[1]).unsqueeze(-1)
    assert torch.equal(gradient, expected.expand_as(positions))


def test_magnetization_interaction_rejects_restricted_list_input() -> None:
    numbers, positions, ihelp, charges = setup_system()
    interactions = InteractionList(MagnetizationProbe())
    cache = interactions.get_cache(numbers, positions, ihelp)
    restricted = charges.select_channel(0)

    with pytest.raises(RuntimeError, match="magnetization channel"):
        interactions.get_energy(restricted, cache, ihelp)
    with pytest.raises(RuntimeError, match="magnetization channel"):
        interactions.get_potential(cache, restricted, ihelp)
    with pytest.raises(RuntimeError, match="magnetization channel"):
        interactions.get_gradient(restricted, positions, cache, ihelp)


def test_empty_list_uses_total_energy_and_zeroes_both_potentials() -> None:
    numbers, positions, ihelp, charges = setup_system()
    interactions = InteractionList()
    cache = interactions.get_cache(numbers, positions, ihelp)

    energy = interactions.get_energy(charges, cache, ihelp)
    decomposition = interactions.get_energy_as_dict(charges, cache, ihelp)
    potential = interactions.get_potential(cache, charges, ihelp)

    assert energy.shape == numbers.shape
    assert torch.count_nonzero(energy) == 0
    assert decomposition["none"].shape == charges.mono[0].shape
    assert potential.nspin == 2
    assert potential.mono is not None
    assert torch.equal(potential.mono, torch.zeros_like(charges.mono))


@pytest.mark.parametrize("batch_mode", [1, 2])
def test_two_channel_routing_preserves_batch_mode(batch_mode: int) -> None:
    numbers, positions, _, charges = setup_system()
    numbers = numbers.unsqueeze(0).expand(3, -1)
    positions = positions.unsqueeze(0).expand(3, -1, -1)
    ihelp = IndexHelper.from_numbers_angular(
        numbers, {1: [0]}, batch_mode=batch_mode
    )
    batched = Charges(
        mono=charges.mono.unsqueeze(0).expand(3, -1, -1),
        dipole=(
            None
            if charges.dipole is None
            else charges.dipole.unsqueeze(0).expand(3, -1, -1, -1)
        ),
        quad=(
            None
            if charges.quad is None
            else charges.quad.unsqueeze(0).expand(3, -1, -1, -1)
        ),
        batch_mode=batch_mode,
        nspin=2,
    )
    interactions = InteractionList(TotalProbe(), MagnetizationProbe())
    cache = interactions.get_cache(numbers, positions, ihelp)

    energy = interactions.get_energy(batched, cache, ihelp)
    potential = interactions.get_potential(cache, batched, ihelp)

    assert energy.shape == numbers.shape
    assert potential.batch_mode == batch_mode
    assert potential.nspin == 2
    assert potential.mono is not None
    assert torch.equal(potential.mono, batched.mono)
