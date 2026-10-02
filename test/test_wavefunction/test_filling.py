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
Test for fractional occupation (Fermi smearing).
Reference values obtained with tbmalt.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.units import KELVIN2AU
from torch.func import jacrev

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.scf.implicit import SelfConsistentFieldImplicit as SCF
from dxtb._src.typing import DD
from dxtb._src.wavefunction import filling
from dxtb.config import ConfigSCF

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H", "H2", "LiH", "SiH4", "S2"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail(dtype: torch.dtype):
    dd: DD = {"device": DEVICE, "dtype": dtype}

    evals = torch.arange(1, 6, **dd)
    nel = torch.tensor([4.0, 4.0], **dd)

    #
    with pytest.raises(RuntimeError):
        filling.get_alpha_beta_occupation(nel, uhf=torch.tensor([0.0], **dd))

    # wrong type
    with pytest.raises(TypeError):
        kt = 300.0
        filling.get_fermi_occupation(nel, evals, kt)  # type: ignore

    # negative etemp
    with pytest.raises(ValueError):
        kt = torch.tensor(-1.0, **dd)
        filling.get_fermi_occupation(nel, evals.expand(2, -1), kt)

    # convergence fails
    with pytest.raises(RuntimeError):
        sample = samples["SiH4"]
        emo = sample["emo"].to(**dd)
        nel = sample["n_electrons"].to(**dd)
        nab = filling.get_alpha_beta_occupation(nel, torch.zeros_like(nel))
        kt = torch.tensor(10000 * KELVIN2AU, dtype=dtype)
        filling.get_fermi_occupation(nab, emo, kt, maxiter=0)


@pytest.mark.parametrize("uhf", [[0, 0, 0], [1, 1, 0], [3, 1, 0]])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail_uhf(dtype: torch.dtype, uhf: list):
    dd: DD = {"device": DEVICE, "dtype": dtype}

    with pytest.raises(ValueError):
        nel = torch.tensor([2, 1, 2], **dd)
        filling.get_alpha_beta_occupation(nel, nel.new_tensor(uhf))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_no_electrons(dtype: torch.dtype):
    dd: DD = {"device": DEVICE, "dtype": dtype}

    evals = torch.arange(1, 6, **dd)
    nel = torch.tensor(0.0, **dd)
    occ = filling.get_fermi_occupation(nel, evals, torch.tensor(300, **dd))

    assert pytest.approx(torch.zeros_like(occ).cpu()) == occ.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]

    nel = sample["n_electrons"].to(**dd)
    uhf = sample["spin"].tolist()
    nab = filling.get_alpha_beta_occupation(nel, uhf)

    emo = sample["emo"].to(**dd)

    ref_focc = sample["focc"].to(**dd)
    ref_efermi = sample["e_fermi"].to(**dd)

    kt = emo.new_tensor(300 * KELVIN2AU)

    efermi, _ = filling.get_fermi_energy(nab, emo)
    assert pytest.approx(ref_efermi.cpu(), abs=tol) == efermi.cpu()

    focc = filling.get_fermi_occupation(nab, emo, kt)
    assert pytest.approx(ref_focc.cpu(), abs=tol) == focc.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

    nel = torch.stack(
        [
            torch.atleast_1d(sample1["n_electrons"].to(**dd)),
            torch.atleast_1d(sample2["n_electrons"].to(**dd)),
            torch.atleast_1d(sample2["n_electrons"].to(**dd)),
        ],
        dim=0,
    )
    uhf = torch.stack(
        [
            torch.atleast_1d(sample1["spin"].to(**dd)),
            torch.atleast_1d(sample2["spin"].to(**dd)),
            torch.atleast_1d(sample2["spin"].to(**dd)),
        ],
        dim=0,
    )
    nab = filling.get_alpha_beta_occupation(nel, uhf)

    emo = pack(
        [
            sample1["emo"].to(**dd),
            sample2["emo"].to(**dd),
            sample2["emo"].to(**dd),
        ]
    )
    # emo = emo.unsqueeze(-2).expand([*nab.shape, -1]) # if only one ref channel

    ref_efermi = pack(
        [
            sample1["e_fermi"].to(**dd),
            sample2["e_fermi"].to(**dd),
            sample2["e_fermi"].to(**dd),
        ]
    )
    # ref_efermi = ref_efermi.expand([*nab.shape]) # if only one ref channel

    ref_focc = pack(
        [
            sample1["focc"].to(**dd),
            sample2["focc"].to(**dd),
            sample2["focc"].to(**dd),
        ]
    )

    kt = emo.new_tensor(300 * KELVIN2AU)

    efermi, _ = filling.get_fermi_energy(nab, emo)
    assert pytest.approx(ref_efermi.cpu(), abs=tol) == efermi.cpu()

    focc = filling.get_fermi_occupation(nab, emo, kt)
    assert pytest.approx(ref_focc.cpu(), abs=tol) == focc.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("kt", [0.0, 5000.0])
def test_kt(dtype: torch.dtype, kt: float):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples["SiH4"]
    numbers = sample["numbers"].to(DEVICE)

    nel = sample["n_electrons"].to(**dd)
    nab = filling.get_alpha_beta_occupation(nel, torch.zeros_like(nel))

    emo = sample["emo"].to(**dd)

    ref_fenergy = {
        0.0: emo.new_tensor(0.0),
        5000.0: emo.new_tensor(-1.6758385176418445e-004),
    }
    ref_focc = {
        0.0: emo.new_tensor(
            [
                2.0,
                2.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        5000.0: emo.new_tensor(
            [
                1.9999999835050197,
                1.9998302497800045,
                1.9998302497800045,
                1.9998302497800045,
                1.6888179157220813e-004,
                1.6888179157220572e-004,
                1.6888179157220273e-004,
                9.3970980586158547e-007,
                9.3970980586158028e-007,
                2.2294515200028562e-007,
                2.2294515200028403e-007,
                2.2294515200027612e-007,
                7.3530914097230426e-008,
                4.7590701921523143e-020,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
            ]
        ),
    }

    # occupation
    focc = filling.get_fermi_occupation(
        nab, emo, emo.new_tensor(kt * KELVIN2AU)
    )
    assert pytest.approx(focc.sum(-2).cpu(), abs=tol) == ref_focc[kt].cpu()

    # electronic free energy
    d = torch.zeros_like(focc)  # dummy
    scf = SCF(
        d,  # type: ignore
        focc,
        d,
        numbers=numbers,
        ihelp=d,
        cache=d,
        integrals=IntegralMatrices(_hcore=d, _overlap=d, **dd),
        config=ConfigSCF(fermi_etemp=kt),
    )

    fenergy = scf.get_electronic_free_energy().sum(-1)
    ref = ref_fenergy[kt]
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == fenergy.cpu()

    scf.config.fermi.partition = -3
    with pytest.raises(ValueError):
        scf.get_electronic_free_energy()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_lumo_not_existing(dtype: torch.dtype) -> None:
    """Helium has no LUMO due to the minimal basis."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples["He"]

    nel = torch.stack(
        [
            torch.atleast_1d(sample["n_electrons"].to(**dd)),
            torch.atleast_1d(sample["n_electrons"].to(**dd)),
            torch.atleast_1d(sample["n_electrons"].to(**dd)),
        ],
        dim=0,
    )
    nab = filling.get_alpha_beta_occupation(nel, torch.zeros_like(nel))

    emo = pack(
        [
            sample["emo"].to(**dd),
            sample["emo"].to(**dd),
            sample["emo"].to(**dd),
        ]
    )

    ref_efermi = pack(
        [
            sample["e_fermi"].to(**dd),
            sample["e_fermi"].to(**dd),
            sample["e_fermi"].to(**dd),
        ]
    )
    ref_focc = pack(
        [
            sample["focc"].to(**dd),
            sample["focc"].to(**dd),
            sample["focc"].to(**dd),
        ]
    )

    kt = emo.new_tensor(300 * KELVIN2AU)

    efermi, _ = filling.get_fermi_energy(nab, emo)
    assert pytest.approx(ref_efermi.cpu(), abs=tol) == efermi.cpu()

    focc = filling.get_fermi_occupation(nab, emo, kt)
    assert pytest.approx(ref_focc.cpu(), abs=tol) == focc.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_lumo_obscured_by_padding(dtype: torch.dtype) -> None:
    """A missing LUMO can be obscured by padding."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples["H2"], samples["He"]

    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    nel = torch.stack(
        [
            torch.atleast_1d(sample1["n_electrons"].to(**dd)),
            torch.atleast_1d(sample2["n_electrons"].to(**dd)),
            torch.atleast_1d(sample2["n_electrons"].to(**dd)),
        ],
        dim=0,
    )
    nab = filling.get_alpha_beta_occupation(nel, torch.zeros_like(nel))

    emo = pack(
        [
            sample1["emo"].to(**dd),
            sample2["emo"].to(**dd),
            sample2["emo"].to(**dd),
        ]
    )

    ref_efermi = pack(
        [
            sample1["e_fermi"].to(**dd),
            sample2["e_fermi"].to(**dd),
            sample2["e_fermi"].to(**dd),
        ]
    )
    ref_focc = pack(
        [
            sample1["focc"].to(**dd),
            sample2["focc"].to(**dd),
            sample2["focc"].to(**dd),
        ]
    )

    kt = emo.new_tensor(300 * KELVIN2AU)

    mask = ihelp.orbitals_per_shell
    mask = mask.unsqueeze(-2).expand([*nab.shape, -1])

    efermi, _ = filling.get_fermi_energy(nab, emo, mask=mask)
    assert pytest.approx(ref_efermi.cpu(), abs=tol) == efermi.cpu()

    focc = filling.get_fermi_occupation(nab, emo, kt, mask=mask)
    assert pytest.approx(ref_focc.cpu(), abs=tol) == focc.cpu()


def _default_thr(dtype: torch.dtype) -> float:
    eps = torch.finfo(dtype).eps
    return min(sqrt(eps), 1e5 * eps, 1e-4)


def _channels(name: str, dd: DD, ktemp: float = 300.0):
    """Alpha/beta electrons, orbital energies (per channel) and kT."""
    sample = samples[name]
    nel = sample["n_electrons"].to(**dd)
    nab = filling.get_alpha_beta_occupation(nel, sample["spin"].tolist())
    emo = sample["emo"].to(**dd)
    kt = emo.new_tensor(ktemp * KELVIN2AU)
    return nab, emo, kt


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("ktemp", [300.0, 1000.0, 5000.0])
@pytest.mark.parametrize("nel", [0.5, 1.5, 2.5, 3.25, 4.0])
def test_fractional_electrons(dtype: torch.dtype, ktemp: float, nel: float):
    """The Fermi energy must be optimized for `nel`, not for its ceiling."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    thr = _default_thr(dtype)

    _, emo, kt = _channels("SiH4", dd, ktemp)
    nab = torch.tensor([nel, nel / 2], **dd)

    focc = filling.get_fermi_occupation(nab, emo, kt)
    assert pytest.approx(nab.cpu(), abs=10 * thr) == focc.sum(-1).detach().cpu()

    # batched
    nabs = torch.stack([nab, nab.flip(-1)])
    emos = torch.stack([emo, emo])
    focc = filling.get_fermi_occupation(nabs, emos, kt)
    assert (
        pytest.approx(nabs.cpu(), abs=10 * thr) == focc.sum(-1).detach().cpu()
    )


@pytest.mark.parametrize("name", ["H2", "LiH", "SiH4"])
@pytest.mark.parametrize("ktemp", [300.0, 5000.0])
@pytest.mark.parametrize("nel", [1.0, 1.5, 2.0, 2.75])
def test_reference_root_finding(name: str, ktemp: float, nel: float):
    """Compare to an independent bracketing root finder (double only)."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    _, emo, kt = _channels(name, dd, ktemp)
    nab = torch.tensor([nel, 0.0], **dd)

    e = emo[0].cpu().numpy()
    kt_ = kt.item()

    def fermi(mu: float) -> np.ndarray:
        return 1.0 / (np.exp(np.clip((e - mu) / kt_, -700, 700)) + 1.0)

    mu = brentq(
        lambda x: fermi(x).sum() - nel,
        e.min() - 100 * kt_,
        e.max() + 100 * kt_,
        xtol=1e-15,
        rtol=1e-15,
    )

    focc = filling.get_fermi_occupation(nab, emo, kt)
    assert pytest.approx(fermi(mu), abs=1e-8) == focc[0].cpu().numpy()
    assert (focc[1] == 0.0).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("ktemp", [300.0, 20000.0])
def test_batch_padding(dtype: torch.dtype, ktemp: float):
    """
    Batch results must equal the single results. Includes an empty beta
    channel (H atom) and padded orbitals (energy 0.0), which must never be
    occupied.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-12 if dtype == torch.double else 1e-6

    names = ["H", "SiH4", "H2"]
    nabs, emos, singles = [], [], []
    for name in names:
        nab, emo, kt = _channels(name, dd, ktemp)
        nabs.append(nab)
        emos.append(emo)
        singles.append(filling.get_fermi_occupation(nab, emo, kt))

    norb = torch.tensor([e.shape[-1] for e in emos], device=DEVICE)
    mask = torch.arange(int(norb.max()), device=DEVICE) < norb.unsqueeze(-1)
    mask = mask.unsqueeze(-2).expand(len(names), 2, -1)

    focc = filling.get_fermi_occupation(
        torch.stack(nabs), pack(emos), kt, mask=mask
    )

    for i, single in enumerate(singles):
        n = single.shape[-1]
        assert pytest.approx(single.cpu(), abs=tol) == focc[i, :, :n].cpu()
        assert (focc[i, :, n:] == 0.0).all()

    # empty beta channel of the H atom
    assert (focc[0, 1] == 0.0).all()
    assert pytest.approx(1.0, abs=10 * _default_thr(dtype)) == focc[0, 0].sum()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_zero_electrons_in_batch(dtype: torch.dtype):
    """A system without electrons must not influence the other systems."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    thr = _default_thr(dtype)

    nab, emo, kt = _channels("SiH4", dd)
    ref = filling.get_fermi_occupation(nab, emo, kt)

    focc = filling.get_fermi_occupation(
        torch.stack([torch.zeros_like(nab), nab]), torch.stack([emo, emo]), kt
    )
    assert (focc[0] == 0.0).all()
    assert pytest.approx(ref.cpu(), abs=thr) == focc[1].cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("nel", [2.0, 3.5])
def test_zero_temperature(dtype: torch.dtype, nel: float):
    """Without smearing, the aufbau filling is returned (no NaN)."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    _, emo, _ = _channels("SiH4", dd)
    emo = emo.clone().requires_grad_()
    nab = torch.tensor([nel, 1.0], **dd)
    kt = torch.tensor(0.0, **dd)

    focc = filling.get_fermi_occupation(nab, emo, kt)
    ref = filling.get_aufbau_occupation(torch.tensor(17), nab)
    assert torch.equal(focc.cpu(), ref.cpu())

    # aufbau does not depend on orbital energies
    (grad,) = torch.autograd.grad(focc.sum(), emo)
    assert (grad == 0.0).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("nel", [1.0, 1.5, 2.0])
def test_extreme_arguments(dtype: torch.dtype, nel: float):
    """No overflow for large `(emo - e_fermi) / kT`, forward and backward."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    emo = torch.tensor([-50.0, -1.0, 0.0, 1e3, 1e4], **dd)
    emo = emo.expand(2, -1).clone().requires_grad_()
    nab = torch.tensor([nel, 1.0], **dd)
    kt = torch.tensor(300.0 * KELVIN2AU, **dd)

    focc = filling.get_fermi_occupation(nab, emo, kt)
    assert torch.isfinite(focc).all()
    assert (
        pytest.approx(nab.cpu(), abs=_default_thr(dtype))
        == focc.sum(-1).detach().cpu()
    )

    weights = torch.arange(5, **dd)
    (grad,) = torch.autograd.grad((focc * weights).sum(), emo)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize(
    "nel, energies, ktemp",
    [
        ([2.0, 1.0], [-0.6, -0.3, -0.25, 0.1, 0.4, 0.8], 5000.0),
        ([1.5, 0.5], [-0.6, -0.3, -0.25, 0.1, 0.4, 0.8], 5000.0),
        ([2.0, 2.0], [-0.6, -0.3, -0.25, 0.1, 0.4, 0.8], 300.0),
        ([2.0, 3.0], [-0.6, -0.3, -0.3, -0.3, 0.4, 0.8], 1000.0),
        ([1.0, 1.0], [-0.9], 300.0),  # completely filled (no LUMO)
    ],
)
def test_gradcheck(nel: list[float], energies: list[float], ktemp: float):
    """Derivatives of the occupation w.r.t. the orbital energies."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    nab = torch.tensor(nel, **dd)
    kt = torch.tensor(ktemp * KELVIN2AU, **dd)

    def fcn(x: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation(nab, x, kt, thr=1e-14)

    # the checks detach their input, i.e., we need a new one for each
    def emo() -> torch.Tensor:
        return (
            torch.tensor(energies, **dd).expand(2, -1).clone().requires_grad_()
        )

    assert dgradcheck(fcn, (emo(),), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert dgradgradcheck(fcn, (emo(),), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_functorch():
    """Jacobian and Hessian with function transforms."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    nab = torch.tensor([2.0, 1.0], **dd)
    emo = torch.tensor([-0.6, -0.3, -0.25, 0.1, 0.4, 0.8], **dd)
    emo = emo.expand(2, -1).clone()
    kt = torch.tensor(5000.0 * KELVIN2AU, **dd)

    def fcn(x: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation(nab, x, kt)

    jac = jacrev(fcn)(emo)

    x = emo.clone().requires_grad_()
    out = fcn(x)
    ref = torch.stack(
        [torch.autograd.grad(o, x, retain_graph=True)[0] for o in out.flatten()]
    ).reshape(jac.shape)
    assert pytest.approx(ref.cpu(), abs=1e-12) == jac.cpu()

    def energy(x: torch.Tensor) -> torch.Tensor:
        return (fcn(x) * x).sum()

    hess = jacrev(jacrev(energy))(emo)

    x = emo.clone().requires_grad_()
    (grad,) = torch.autograd.grad(energy(x), x, create_graph=True)
    ref = torch.stack(
        [
            torch.autograd.grad(g, x, retain_graph=True)[0]
            for g in grad.flatten()
        ]
    ).reshape(hess.shape)
    assert pytest.approx(ref.cpu(), abs=1e-10) == hess.cpu()


def test_compile():
    """Smoke test for tracing with ``torch.compile``."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    nab, emo, kt = _channels("SiH4", dd, 5000.0)
    ref = filling.get_fermi_occupation(nab, emo, kt)

    compiled = torch.compile(filling.get_fermi_occupation, backend="aot_eager")
    focc = compiled(nab, emo, kt)
    assert pytest.approx(ref.cpu(), abs=1e-12) == focc.cpu()


def test_kt_must_be_scalar():
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    nab, emo, _ = _channels("SiH4", dd)
    kt = torch.tensor([300.0, 300.0], **dd) * KELVIN2AU

    with pytest.raises(ValueError, match="scalar"):
        filling.get_fermi_occupation(nab, emo, kt)

    # one element is fine
    kt = kt[:1]
    filling.get_fermi_occupation(nab, emo, kt)


def test_tiny_temperature_single_precision() -> None:
    """
    In single precision, a single ulp of the (absolute) Fermi energy changes
    the number of electrons by more than the threshold at tiny temperatures.
    The search is relative to the initial guess, whose resolution suffices:
    the degenerate pair shares the fractional electrons exactly.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.float}

    emo = torch.tensor([-1.0, -0.5, -0.5, 0.4, 0.5], **dd).expand(2, -1)
    nab = torch.tensor([2.5, 1.5], **dd)
    kt = torch.tensor(3.1e-7, **dd)

    # one ulp of the Fermi energy (at -0.5) moves 2 f (1 - f) / kT electrons
    ulp = torch.finfo(torch.float).eps / 2
    assert ulp * 2 * 0.25 * 0.75 / kt.item() > 2 * 1e-4

    occ = filling.get_fermi_occupation(nab, emo, kt)
    ref = torch.tensor(
        [[1.0, 0.75, 0.75, 0.0, 0.0], [1.0, 0.25, 0.25, 0.0, 0.0]], **dd
    )
    assert (occ - ref).abs().max() <= 1e-6


def test_degenerate_single_precision() -> None:
    """
    Almost degenerate orbitals at an absolute energy of a few Hartree (the 3d
    shell of Mn in the SCF) in single precision: one ulp of the absolute
    Fermi energy moves more electrons than the threshold, but the search
    (relative to the initial guess) converges.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.float}

    emo = torch.tensor(
        [-4.4926987, -4.4926891, -4.4926815, -4.4926419, -4.4926395, -2.30302],
        **dd,
    ).expand(2, -1)
    nab = torch.tensor([5.0, 3.0], **dd)
    kt = torch.tensor(300.0 * KELVIN2AU, **dd)
    thr = 1e-4

    # one ulp at -4.49 is 4.8e-7, about 5 * 0.24 / kT = 1260 electrons/Eh
    ulp = 4.0 * torch.finfo(torch.float).eps / 2
    assert ulp * 5 * 0.24 / kt.item() > 2 * thr

    occ = filling.get_fermi_occupation(nab, emo, kt, thr=thr)
    assert (occ.sum(-1) - nab).abs().max() <= thr


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("kt", [0.0, 300.0 * KELVIN2AU])
def test_padding_not_filled_at_zero_temperature(dtype: torch.dtype, kt: float):
    """
    Padding is never occupied, also for the aufbau filling. It is skipped
    even if it sits between existing orbitals (energy 0.0).
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}

    emo = torch.tensor([[-1.0, -0.5, 0.0, 0.0, 0.3, 0.8]], **dd).expand(2, -1)
    mask = torch.tensor([[1, 1, 0, 0, 1, 1]], device=DEVICE).expand(2, -1)
    nab = torch.tensor([3.0, 2.5], **dd)

    focc = filling.get_fermi_occupation(
        nab, emo, torch.tensor(kt, **dd), mask=mask
    )
    assert (focc[:, 2:4] == 0.0).all()
    assert (
        pytest.approx(nab.cpu(), abs=_default_thr(dtype)) == focc.sum(-1).cpu()
    )

    if kt == 0.0:
        ref = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.5, 0.0]],
            **dd,
        )
        assert torch.equal(focc.cpu(), ref.cpu())


@pytest.fixture
def host_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record all reads of tensor values into Python objects."""
    calls: list[str] = []

    for name in ("item", "tolist", "__bool__", "__float__", "__int__"):
        orig = getattr(torch.Tensor, name)

        def wrapper(self, *args, _orig=orig, _name=name, **kwargs):
            calls.append(_name)
            return _orig(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, wrapper)

    return calls


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_host_reads(dtype: torch.dtype, host_reads: list[str]):
    """
    Each read of a tensor value synchronizes the device. There must be a
    single one for converged initial guesses and only one per few iterations
    otherwise, not one per iteration. The validation of the input and the
    choice between aufbau and Fermi filling are part of the first one.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}

    def syncs() -> int:
        assert [c for c in host_reads if c != "item"] == []
        n = host_reads.count("item")
        host_reads.clear()
        return n

    # gapped integer case: initial guess is converged
    nab, emo, kt = _channels("SiH4", dd)
    host_reads.clear()
    filling.get_fermi_occupation(nab, emo.clone().requires_grad_(), kt)
    assert syncs() == 1

    # almost filled channel: slow convergence in the tail of the Fermi
    # function, i.e., many iterations (a completely filled channel needs no
    # search)
    emo = torch.tensor([[-1.0, -0.5]], **dd).expand(2, -1)
    nab = torch.tensor([1.999, 1.999], **dd)
    kt = torch.tensor(5000.0 * KELVIN2AU, **dd)
    filling.get_fermi_occupation(nab, emo, kt)
    assert 1 < syncs() < 6

    # zero temperature and empty channels need only the first check
    filling.get_fermi_occupation(nab, emo, torch.tensor(0.0, **dd))
    assert syncs() == 1
    filling.get_fermi_occupation(nab * 0, emo, kt)
    assert syncs() == 1


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_default_threshold_is_reachable(dtype: torch.dtype):
    """
    The default threshold is above the floating point noise of the sum of
    occupations for realistic numbers of orbitals and temperatures (no
    convergence failures), and below the tolerance of the electron number
    check in the SCF (5e-4 because of rounding to three decimals).
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}
    thr = _default_thr(dtype)
    assert thr < 5e-4

    gen = torch.Generator().manual_seed(0)
    for norb in (10, 60, 200):
        for ktemp in (100.0, 300.0, 5000.0):
            emo = torch.randn(norb, generator=gen, dtype=torch.double)
            emo = emo.sort().values.to(**dd).expand(2, -1)
            kt = torch.tensor(ktemp * KELVIN2AU, **dd)

            for nel in (norb // 4, norb / 4 + 0.37):
                nab = torch.tensor([nel, nel / 2], **dd)
                focc = filling.get_fermi_occupation(nab, emo, kt)
                assert (focc.sum(-1) - nab).abs().max() <= thr


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_too_many_electrons(dtype: torch.dtype):
    """More electrons than (existing) orbitals is invalid input."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    emo = torch.tensor([[-1.0, -0.5, 0.0, 0.0]], **dd).expand(2, -1)
    kt = torch.tensor(300.0 * KELVIN2AU, **dd)

    # 5 electrons in 4 orbitals
    with pytest.raises(ValueError, match="exceeds"):
        filling.get_fermi_occupation(torch.tensor([5.0, 1.0], **dd), emo, kt)

    # 3 electrons, but only two existing orbitals
    mask = torch.tensor([[1, 1, 0, 0]], device=DEVICE).expand(2, -1)
    nab = torch.tensor([3.0, 1.0], **dd)
    with pytest.raises(ValueError, match="exceeds"):
        filling.get_fermi_occupation(nab, emo, kt, mask=mask)

    # exactly as many electrons as orbitals is fine
    nab = torch.tensor([2.0, 1.0], **dd)
    focc = filling.get_fermi_occupation(nab, emo, kt, mask=mask)
    assert (
        pytest.approx(nab.cpu(), abs=_default_thr(dtype)) == focc.sum(-1).cpu()
    )

    # inactive channels are not checked (zero temperature: aufbau)
    filling.get_fermi_occupation(nab * 0, emo, kt)


def test_aufbau_batch_padding():
    """The public aufbau filling never occupies orbitals beyond `norb`."""
    nel = torch.tensor([2.0, 3.5, 1.5])
    norb = torch.tensor([4, 4, 2])

    occ = filling.get_aufbau_occupation(norb, nel)
    ref = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.5],
            [1.0, 0.5, 0.0, 0.0],
        ]
    )
    assert torch.equal(occ, ref)

    # more electrons than orbitals are cut off
    occ = filling.get_aufbau_occupation(
        torch.tensor([3, 2]), torch.tensor([4.0, 1.0])
    )
    assert torch.equal(occ, torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]))


def test_aufbau_scalar_electrons_batch() -> None:
    """A scalar number of electrons fills every system of a batch."""
    occ = filling.get_aufbau_occupation(torch.tensor([3, 2]), torch.tensor(1.5))
    assert torch.equal(occ, torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]))

    # the second system has only one orbital
    occ = filling.get_aufbau_occupation(torch.tensor([3, 1]), torch.tensor(1.5))
    assert torch.equal(occ, torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.0, 0.0]]))


@pytest.mark.cuda
@pytest.mark.parametrize("ktemp", [0.0, 5000.0])
def test_temperature_on_cpu(ktemp: float) -> None:
    """A temperature on the CPU works with orbital energies on the GPU."""
    dd: DD = {"device": torch.device("cuda"), "dtype": torch.double}

    nab, emo, _ = _channels("SiH4", dd)
    kt = torch.tensor(ktemp * KELVIN2AU, dtype=torch.double)
    ref = filling.get_fermi_occupation(nab, emo, kt.to(emo.device))
    occ = filling.get_fermi_occupation(nab, emo, kt)
    assert occ.device == emo.device
    assert torch.equal(occ, ref)

    with pytest.raises(ValueError):
        filling.get_fermi_occupation(nab, emo, -kt - 1.0)


@pytest.mark.parametrize("nel", [[2.0, 1.0], [1.5, 0.5]])
def test_third_derivative(nel: list[float]):
    """
    Two differentiable Newton steps give derivatives up to third order
    (quadratic convergence), which are required for, e.g., the
    hyperpolarizability. One step would be wrong for the Hessian already if
    the occupations are fractional.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    nab = torch.tensor(nel, **dd)
    emo = torch.tensor([-0.6, -0.3, -0.25, 0.1, 0.4, 0.8], **dd)
    emo = emo.expand(2, -1).clone()
    kt = torch.tensor(5000.0 * KELVIN2AU, **dd)

    def fcn(x: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation(nab, x, kt, thr=1e-14)

    def hess(x: torch.Tensor) -> torch.Tensor:
        return jacrev(jacrev(fcn))(x)

    third = jacrev(hess)(emo)

    # central differences of the Hessian along one orbital energy
    h = 1e-5
    step = torch.zeros_like(emo)
    step[0, 2] = h
    fd = (hess(emo + step) - hess(emo - step)) / (2 * h)

    assert pytest.approx(fd.cpu(), abs=1e-4, rel=1e-3) == third[..., 0, 2].cpu()


@pytest.mark.parametrize("delta", [4e-15, -4e-15])
def test_differentiable_steps_do_not_overshoot(delta: float):
    """
    The number of electrons is an integer within tolerance and the start of
    the search is accepted, but the far tail of the Fermi function has a
    negligible derivative, so the residual divided by it is huge. Unguarded
    Newton steps would throw the Fermi energy out of the gap (an error of one
    to three electrons in the occupations).
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    nab = torch.tensor([2.0 + delta, 1.0], **dd)
    emo = torch.tensor([-0.6, -0.3, -0.274, 0.9, 1.2], **dd)
    emo = emo.expand(2, -1).clone().requires_grad_()
    kt = torch.tensor(3e-4, **dd)

    occ = filling.get_fermi_occupation(nab, emo, kt)
    assert pytest.approx(nab.cpu(), abs=1e-12) == occ.sum(-1).detach().cpu()

    (grad,) = torch.autograd.grad(occ.pow(2).sum(), emo)
    assert torch.isfinite(grad).all()


def test_aufbau_batch_padding_channels():
    """Per-system orbitals with alpha/beta electrons (one per channel)."""
    norb = torch.tensor([4, 6, 5])
    nel = torch.tensor([[2.0, 1.0], [3.0, 2.5], [3.0, 0.0]])
    occ = filling.get_aufbau_occupation(norb, nel)
    assert occ.shape == (3, 2, 6)

    # padding is never occupied, electrons fill the existing orbitals
    for b in range(3):
        for c in range(2):
            ref = filling.get_aufbau_occupation(norb[b], nel[b, c])
            assert torch.equal(occ[b, c, : norb[b]], ref[: norb[b]])
            assert (occ[b, c, norb[b] :] == 0).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("padded", [False, True])
def test_full_channel(dtype: torch.dtype, padded: bool):
    """
    A completely filled channel (He, H-) has exactly constant occupations of
    one, i.e., vanishing derivatives of all orders (also in single precision).
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}

    norb = 3 if padded else 1
    emo = torch.full((1, 2, norb), -0.5, **dd)
    mask = torch.zeros_like(emo, dtype=torch.int)
    mask[..., 0] = 1
    emo[..., 1:] = 0.3  # padding, would be occupied without the mask
    emo.requires_grad_()

    nab = torch.tensor([[1.0, 1.0]], **dd)
    kt = torch.tensor(300.0 * KELVIN2AU, **dd)

    occ = filling.get_fermi_occupation(
        nab, emo, kt, mask=mask if padded else None
    )
    assert (occ[..., 0] == 1.0).all()
    assert (occ[..., 1:] == 0.0).all()

    (g1,) = torch.autograd.grad(
        (occ**2).sum(), emo, create_graph=True, retain_graph=True
    )
    (g2,) = torch.autograd.grad(g1.sum(), emo, allow_unused=True)
    assert (g1 == 0).all()
    assert g2 is None or (g2 == 0).all()

    # the other channel is still smeared
    nab = torch.tensor([[1.0, 0.5]], **dd)
    occ = filling.get_fermi_occupation(
        nab, emo, kt, mask=mask if padded else None
    )
    assert (occ[..., 0, 0] == 1.0).all()
    assert pytest.approx(0.5, abs=_default_thr(dtype)) == occ[0, 1].sum().item()


def test_maxiter_counts_updates():
    """
    `maxiter` is the number of updates: the result of the last update is
    checked, too. A search that needs `n` updates converges with `maxiter=n`.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    emo = torch.tensor([-0.5, -0.1, 0.2, 0.4, 0.45, 0.9], **dd)
    emo = emo.expand(1, 2, -1)
    nab = torch.tensor([[1.5, 0.3]], **dd)
    kt = torch.tensor(3e-3, **dd)

    needed = None
    for maxiter in range(1, 50):
        try:
            filling.get_fermi_occupation(nab, emo, kt, maxiter=maxiter)
        except RuntimeError:
            continue
        needed = maxiter
        break

    assert needed is not None and needed > 1
    with pytest.raises(RuntimeError, match="failed to converge"):
        filling.get_fermi_occupation(nab, emo, kt, maxiter=needed - 1)

    # the search converged after `needed` updates: a further iteration does
    # not change the outcome (no update is applied after the last check)
    ref = filling.get_fermi_occupation(nab, emo, kt, maxiter=200)
    focc = filling.get_fermi_occupation(nab, emo, kt, maxiter=needed)
    assert (focc.sum(-1) - nab).abs().max() <= 1e-9
    assert (focc - ref).abs().max() <= 1e-9


@pytest.mark.parametrize("pad", [-1e5, 0.3, 1e5])
def test_padding_energy_is_ignored(pad: float):
    """The value of padded orbital energies has no influence on the result."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    emo = torch.tensor([-0.5, -0.1, 0.2, 0.4, 0.45, 0.9, 0.0], **dd)
    emo = emo.expand(1, 2, -1).clone()
    mask = torch.tensor([1, 1, 1, 1, 1, 1, 0], device=DEVICE)
    mask = mask.expand(1, 2, -1)
    nab = torch.tensor([[2.5, 1.25]], **dd)
    kt = torch.tensor(1e-4, **dd)

    ref = filling.get_fermi_occupation(nab, emo, kt, mask=mask)
    emo[..., -1] = pad
    focc = filling.get_fermi_occupation(nab, emo, kt, mask=mask)
    assert torch.equal(focc, ref)
    assert (focc[..., -1] == 0.0).all()
