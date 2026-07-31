# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, IndexHelper
from dxtb._src.components.interactions import Charges
from dxtb._src.components.interactions.spin import factory, new_spinpolarisation
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

SINGLE_CASES = [
    pytest.param("LiH", 2, id="LiH-spin2"),
    pytest.param("SiH4", 2, id="SiH4-spin2"),
    pytest.param("MB16_43_02", 1, id="MB16_43_02-spin1"),
]


@pytest.mark.parametrize("name, spin", SINGLE_CASES)
@pytest.mark.parametrize(
    "model_cls, ref_key",
    [
        pytest.param(GFN1_XTB, "espgfn1", id="gfn1"),
        pytest.param(
            GFN2_XTB,
            "espgfn2",
            id="gfn2",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(
    dtype: torch.dtype,
    name: str,
    spin: int,
    model_cls,
    ref_key,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(DEVICE)
    ref = sample[ref_key].to(**dd)

    spinpol = new_spinpolarisation(numbers=numbers, **dd)
    calc = Calculator(numbers, par=model_cls, interaction=[spinpol], **dd)

    result = calc.singlepoint(
        positions, chrg=torch.tensor(0.0, **dd), spin=spin
    )
    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


def test_zero_spin_autoselects_odd_electron_state() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    sample = samples["MB16_43_02"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    energies = []
    for spin in (0, 1):
        calc = Calculator(
            numbers,
            par=GFN1_XTB,
            interaction=[new_spinpolarisation(numbers, **dd)],
            opts={"verbosity": 0},
            **dd,
        )
        result = calc.singlepoint(
            positions, chrg=torch.tensor(0.0, **dd), spin=spin
        )
        energies.append(result.total.sum(-1))

    torch.testing.assert_close(energies[0], energies[1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "model_cls, ref_key",
    [
        pytest.param(GFN1_XTB, "espgfn1", id="gfn1"),
        pytest.param(
            GFN2_XTB,
            "espgfn2",
            id="gfn2",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_batch(dtype: torch.dtype, model_cls, ref_key, scf_mode: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    names = ("LiH", "SiH4")
    numbers = pack(tuple(samples[name]["numbers"].to(DEVICE) for name in names))
    positions = pack(
        tuple(samples[name]["positions"].to(DEVICE) for name in names)
    )
    ref = pack(tuple(samples[name][ref_key].to(**dd) for name in names))
    spins = torch.tensor([2, 2], device=DEVICE)
    options = {
        "verbosity": 0,
        "scf_mode": scf_mode,
        "mixer": "anderson" if scf_mode == "full" else "broyden",
    }

    spinpol = new_spinpolarisation(numbers=numbers, **dd)
    calc = Calculator(
        numbers, par=model_cls, interaction=[spinpol], opts=options, **dd
    )

    result = calc.singlepoint(
        positions, chrg=torch.zeros(len(names), **dd), spin=spins
    )
    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
def test_batch_full_all_scp_modes(scp_mode: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    names = ("LiH", "SiH4")
    numbers = pack(tuple(samples[name]["numbers"].to(DEVICE) for name in names))
    positions = pack(
        tuple(samples[name]["positions"].to(DEVICE) for name in names)
    )
    reference = pack(tuple(samples[name]["espgfn1"].to(**dd) for name in names))

    calc = Calculator(
        numbers,
        par=GFN1_XTB,
        interaction=[new_spinpolarisation(numbers=numbers, **dd)],
        opts={
            "verbosity": 0,
            "scf_mode": "full",
            "scp_mode": scp_mode,
            "mixer": "anderson",
        },
        **dd,
    )
    result = calc.singlepoint(
        positions,
        chrg=torch.zeros(len(names), **dd),
        spin=torch.tensor([2, 2], device=DEVICE),
    )

    assert result.charges.nspin == 2
    assert result.charges.mono.shape == (2, 2, 17)
    assert (
        pytest.approx(reference.cpu(), abs=2e-7, rel=2e-7)
        == result.total.sum(-1).cpu()
    )


@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
def test_pure_implicit_matches_nonpure(scp_mode: str) -> None:
    """Pure implicit SCF must retain both UHF channels for every target."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    sample = samples["LiH"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(DEVICE)

    results = []
    for scf_mode in ("implicit", "nonpure"):
        spinpol = new_spinpolarisation(numbers=numbers, **dd)
        calc = Calculator(
            numbers,
            par=GFN1_XTB,
            interaction=[spinpol],
            opts={
                "verbosity": 0,
                "scf_mode": scf_mode,
                "scp_mode": scp_mode,
            },
            **dd,
        )
        results.append(
            calc.singlepoint(positions, chrg=torch.tensor(0.0, **dd), spin=2)
        )

    pure, nonpure = results
    assert pure.charges.nspin == nonpure.charges.nspin == 2
    assert pure.charges.mono.shape == nonpure.charges.mono.shape == (2, 6)
    torch.testing.assert_close(
        pure.total.sum(-1), nonpure.total.sum(-1), rtol=1e-7, atol=1e-7
    )
    torch.testing.assert_close(
        pure.charges.mono, nonpure.charges.mono, rtol=1e-7, atol=1e-7
    )
    torch.testing.assert_close(pure.emo, nonpure.emo, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("name", ["LiH"])
@pytest.mark.parametrize(
    "model_cls, ref_key",
    [
        (GFN2_XTB, "eshellgfn2"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_get_monopol_shell_energy(
    dtype: torch.dtype, name: str, model_cls, ref_key
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    qsh = torch.tensor(
        [
            [0.17175602, -0.82824398],
            [-0.33807717, -0.33807717],
            [0.16632115, -0.83367885],
        ],
        **dd,
    )

    # SpinPolarisation now expects single-channel magnetization
    qsh_mag = qsh[:, -1]

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = sample[ref_key].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, model_cls)
    spin = factory.new_spinpolarisation(numbers, **dd)

    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)

    eshell = spin.get_monopole_shell_energy(cache=cache, qsh=qsh_mag)

    at_shell = ihelp.reduce_shell_to_atom(eshell)

    assert pytest.approx(ref.cpu(), abs=tol) == at_shell.cpu()


def test_spinpolarisation_is_rhf_noop() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = samples["LiH"]["numbers"].to(DEVICE)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    spinpol = new_spinpolarisation(numbers, **dd)
    cache = spinpol.get_cache(numbers=numbers, ihelp=ihelp)
    charges = Charges(mono=torch.ones(ihelp.nao, **dd), nspin=1)

    energy = spinpol.get_energy(cache, charges, ihelp)
    potential = spinpol.get_potential(cache, charges, ihelp)

    assert torch.count_nonzero(energy) == 0
    assert potential.mono is not None
    assert torch.count_nonzero(potential.mono) == 0


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_spin_resolved_atomic_multipoles() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    sample = samples["LiH"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    options = {"verbosity": 0, "scf_mode": "nonpure", "scp_mode": "potential"}

    rhf = Calculator(numbers, GFN2_XTB, opts=options, **dd).singlepoint(
        positions, chrg=torch.tensor(0.0, **dd)
    )
    uhf = Calculator(
        numbers,
        GFN2_XTB,
        opts={**options, "uhf_mode": True},
        **dd,
    ).singlepoint(positions, chrg=torch.tensor(0.0, **dd), spin=0)

    assert rhf.charges.dipole is not None
    assert rhf.charges.quad is not None
    assert uhf.charges.dipole is not None
    assert uhf.charges.quad is not None
    assert uhf.charges.dipole.shape == (2, 2, 3)
    assert uhf.charges.quad.shape == (2, 2, 6)
    torch.testing.assert_close(uhf.charges.dipole[0], rhf.charges.dipole)
    torch.testing.assert_close(uhf.charges.quad[0], rhf.charges.quad)
    torch.testing.assert_close(
        uhf.charges.dipole[1],
        torch.zeros_like(uhf.charges.dipole[1]),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        uhf.charges.quad[1],
        torch.zeros_like(uhf.charges.quad[1]),
        atol=1e-7,
        rtol=0.0,
    )

    spinpol = new_spinpolarisation(numbers, **dd)
    open_shell = Calculator(
        numbers,
        GFN2_XTB,
        interaction=[spinpol],
        opts=options,
        **dd,
    ).singlepoint(positions, chrg=torch.tensor(0.0, **dd), spin=2)
    assert open_shell.charges.dipole is not None
    assert open_shell.charges.quad is not None
    assert torch.linalg.vector_norm(open_shell.charges.dipole[1]) > 0
    assert torch.linalg.vector_norm(open_shell.charges.quad[1]) > 0


@pytest.mark.parametrize("name", ["LiH"])
@pytest.mark.parametrize(
    "model_cls, ref_key",
    [
        (GFN2_XTB, "potshellgfn2"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_get_monopol_shell_potential(
    dtype: torch.dtype, name: str, model_cls, ref_key
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # tblite run lih.coord --method gfn2 --spin 2 --spin-polarized --iterations 2
    # then print out the potential in spin.f90
    qsh = torch.tensor(
        [
            [0.06870241, -0.33129759],
            [-0.13523087, -0.13523087],
            [0.06652846, -0.33347154],
        ],
        **dd,
    )

    # SpinPolarisation now expects single-channel magnetization
    qsh_mag = qsh[:, -1]

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = sample[ref_key].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, model_cls)
    spin = factory.new_spinpolarisation(numbers, **dd)

    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)

    potshell = spin.get_monopole_shell_potential(cache=cache, qsh=qsh_mag)

    assert pytest.approx(ref.cpu(), abs=tol) == potshell.cpu()
