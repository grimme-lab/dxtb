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
Test fone-shot electronic solver.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import IndexHelper, labels
from dxtb._src.components.interactions import Interaction, InteractionList
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.integral.factory import new_hcore
from dxtb._src.integral.wrappers import overlap
from dxtb._src.param import GFN0_XTB
from dxtb._src.scf.iterator import solve
from dxtb._src.scf.pure.conversions import diagonalize
from dxtb._src.scf.result import SCFResult
from dxtb._src.typing import Tensor
from dxtb._src.xtb.gfn0 import GFN0Hamiltonian
from dxtb.config import ConfigSCF

from .samples import Record, samples


def build_problem(
    name: str,
) -> tuple[
    Record,
    Tensor,
    Tensor,
    IndexHelper,
    GFN0Hamiltonian,
    IntegralMatrices,
]:
    """Build the neutral GFN0 H0/S problem from the shared SCF samples."""
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    charge = torch.tensor(0.0, dtype=torch.float64)

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    hcore = new_hcore(numbers, GFN0_XTB, ihelp, dtype=torch.float64)
    assert isinstance(hcore, GFN0Hamiltonian)
    ovlp = overlap(
        numbers,
        positions,
        GFN0_XTB,
        driver=labels.INTDRIVER_ANALYTICAL,
    )
    h0 = hcore.build(positions, ovlp, charge=charge)
    integrals = IntegralMatrices(
        _hcore=h0,
        _overlap=ovlp,
        device=positions.device,
        dtype=positions.dtype,
    )
    return sample, numbers, positions, ihelp, hcore, integrals


def run_problem(
    name: str,
    *,
    config: ConfigSCF | None = None,
    interactions: InteractionList | None = None,
) -> tuple[Record, SCFResult]:
    """Run the non-self-consistent solver for one neutral molecule."""
    sample, numbers, positions, ihelp, hcore, integrals = build_problem(name)
    if config is None:
        config = ConfigSCF(method=labels.GFN0_XTB, dtype=torch.float64)
    if interactions is None:
        interactions = InteractionList()

    result = solve(
        numbers,
        positions,
        torch.tensor(0.0, dtype=torch.float64),
        None,
        interactions,
        interactions.get_cache(numbers, positions, ihelp),
        ihelp,
        config,
        integrals,
        hcore.refocc,
    )
    return sample, result


def test_requires_iterations_is_derived() -> None:
    """Select one-shot execution only from the configured physical method."""
    gfn0 = ConfigSCF(method=labels.GFN0_XTB)
    assert gfn0.requires_iterations is False
    assert ConfigSCF(method=labels.GFN1_XTB).requires_iterations is True
    assert ConfigSCF(method=labels.GFN2_XTB).requires_iterations is True

    with pytest.raises(AttributeError):
        gfn0.requires_iterations = True  # type: ignore[misc]


@pytest.mark.parametrize("name", ["H2", "LiH", "H2O"])
def test_complete_result_and_reference_energy(name: str) -> None:
    """Populate the full result and reproduce the shared GFN0 H0 energy."""
    sample, result = run_problem(name)

    assert set(result) == {
        "charges",
        "coefficients",
        "density",
        "emo",
        "energy",
        "fenergy",
        "hamiltonian",
        "occupation",
        "potential",
        "iterations",
    }
    assert result["iterations"] == 0
    assert result["energy"].sum().item() == pytest.approx(
        sample["egfn0"].item(), abs=5.0e-7
    )
    assert result["charges"].mono.sum().item() == pytest.approx(
        0.0, abs=1.0e-10
    )
    potential = result["potential"].mono
    assert potential is not None
    assert torch.count_nonzero(potential).item() == 0

    for key in (
        "coefficients",
        "density",
        "emo",
        "energy",
        "fenergy",
        "hamiltonian",
        "occupation",
    ):
        assert torch.isfinite(result[key]).all()


def test_exactly_one_diagonalization_and_no_initial_guess() -> None:
    """Dispatch directly to one diagonalization without an SCF guess."""
    config = ConfigSCF(
        method=labels.GFN0_XTB,
        maxiter=999,
        mixer=labels.MIXER_BROYDEN,
        dtype=torch.float64,
    )

    with patch(
        "dxtb._src.scf.pure.conversions.diagonalize", wraps=diagonalize
    ) as diagonalize_mock:
        with patch(
            "dxtb._src.scf.iterator.get_guess",
            side_effect=AssertionError("SCF guess must not be called"),
        ):
            _, result = run_problem("H2O", config=config)

    assert result["iterations"] == 0
    assert diagonalize_mock.call_count == 1


@pytest.mark.parametrize(
    ("maxiter", "mixer"),
    [
        (0, labels.MIXER_LINEAR),
        (1, labels.MIXER_ANDERSON),
        (999, labels.MIXER_BROYDEN),
    ],
)
def test_mixer_and_maxiter_are_inert(maxiter: int, mixer: int) -> None:
    """Ignore mixer and iteration-limit settings for the GFN0 solver."""
    _, reference = run_problem("H2O")
    _, result = run_problem(
        "H2O",
        config=ConfigSCF(
            method=labels.GFN0_XTB,
            maxiter=maxiter,
            mixer=mixer,
            dtype=torch.float64,
        ),
    )

    assert result["iterations"] == 0
    for key in (
        "density",
        "emo",
        "energy",
        "fenergy",
        "hamiltonian",
        "occupation",
    ):
        torch.testing.assert_close(
            result[key], reference[key], rtol=0.0, atol=0.0
        )
    torch.testing.assert_close(
        result["charges"].mono,
        reference["charges"].mono,
        rtol=0.0,
        atol=0.0,
    )


def test_charge_dependent_interaction_is_rejected() -> None:
    """Reject density-dependent interactions instead of applying one pass."""
    interactions = InteractionList(Interaction(dtype=torch.float64))

    with pytest.raises(
        ValueError,
        match="does not support charge-dependent Interaction components",
    ):
        run_problem("H2", interactions=interactions)


def test_heterogeneous_batch_matches_separate_solves() -> None:
    """Match a packed H2/H2O solve to the separate one-shot calculations."""
    names = ("H2", "H2O")
    number_list = [samples[name]["numbers"] for name in names]
    position_list = [
        samples[name]["positions"].to(dtype=torch.float64) for name in names
    ]
    numbers = pack(number_list)
    positions = pack(position_list)
    charges = torch.zeros((len(names), 1), dtype=torch.float64)

    ihelp = IndexHelper.from_numbers(numbers, GFN0_XTB)
    hcore = new_hcore(numbers, GFN0_XTB, ihelp, dtype=torch.float64)
    assert isinstance(hcore, GFN0Hamiltonian)
    ovlp = overlap(
        numbers,
        positions,
        GFN0_XTB,
        driver=labels.INTDRIVER_ANALYTICAL,
    )
    h0 = hcore.build(positions, ovlp, charge=charges)
    integrals = IntegralMatrices(
        _hcore=h0,
        _overlap=ovlp,
        device=positions.device,
        dtype=positions.dtype,
    )
    interactions = InteractionList()
    result = solve(
        numbers,
        positions,
        charges,
        None,
        interactions,
        interactions.get_cache(numbers, positions, ihelp),
        ihelp,
        ConfigSCF(
            method=labels.GFN0_XTB,
            batch_mode=1,
            dtype=torch.float64,
        ),
        integrals,
        hcore.refocc,
    )

    separate = [run_problem(name)[1] for name in names]
    assert result["iterations"] == 0
    torch.testing.assert_close(
        result["energy"].sum(-1),
        torch.stack([item["energy"].sum() for item in separate]),
        rtol=0.0,
        atol=1.0e-10,
    )
    potential = result["potential"].mono
    assert potential is not None
    assert torch.count_nonzero(potential).item() == 0
