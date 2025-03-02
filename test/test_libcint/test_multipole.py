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
Test overlap from libcint.

We cannot compare with our internal integrals or the reference integrals from
tblite, because the sorting of the p-orbitals are different.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.convert import numpy_to_tensor
from tad_mctc.math import einsum

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, labels
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint, has_pyscf
from dxtb._src.integral.driver.manager import DriverManager
from dxtb._src.integral.types.quadrupole import _reduce_9_to_6
from dxtb._src.typing import DD, Tensor
from dxtb._src.utils import is_basis_list
from dxtb.integrals import Integrals
from dxtb.integrals.factories import new_dipint, new_overlap, new_quadint

if has_pyscf is True:
    from dxtb._src.exlibs.pyscf.mol import M
if has_libcint is True:
    from dxtb._src.exlibs import libcint

from ..conftest import DEVICE
from .samples import samples

slist = ["H2", "LiH", "Li2", "H2O", "S"]
slist_large = ["SiH4", "MB16_43_01", "C60"]


def snorm(overlap: Tensor) -> Tensor:
    return torch.pow(overlap.diagonal(dim1=-1, dim2=-2), -0.5)


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_single(dtype: torch.dtype, name: str, gfn) -> None:
    run_single(dtype, name, gfn)


@pytest.mark.large
@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist_large)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_large(dtype: torch.dtype, name: str, gfn: str) -> None:
    run_single(dtype, name, gfn)


def run_single(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_libcint(positions)
    assert is_basis_list(atombases)

    def _run(i: int, atol: float) -> None:
        # dxtb's libcint integral
        wrapper = libcint.LibcintWrapper(atombases, ihelp)
        dxtb_mpint = libcint.int1e(i * "j", wrapper)

        # pyscf reference integral
        assert M is not False
        mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
        pyscf_mpint = numpy_to_tensor(mol.intor(f"int1e_{i*'r'}_origj"), **dd)

        assert dxtb_mpint.shape == pyscf_mpint.shape
        assert pytest.approx(pyscf_mpint.cpu(), abs=atol) == dxtb_mpint.cpu()

        # normalize
        norm = snorm(libcint.overlap(wrapper))
        dxtb_mpint = einsum("xij,i,j->xij", dxtb_mpint, norm, norm)
        pyscf_mpint = einsum("xij,i,j->xij", pyscf_mpint, norm, norm)

        assert dxtb_mpint.shape == pyscf_mpint.shape
        assert pytest.approx(pyscf_mpint.cpu(), abs=tol) == dxtb_mpint.cpu()

    _run(1, atol=tol * 1e-2)  # dipole
    _run(2, atol=tol)  # quadrupole


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_shift_r0_rj(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    ##########################################################################

    # Setup the driver
    mgr = DriverManager(labels.INTDRIVER_LIBCINT, **dd)
    mgr.create_driver(numbers, par, ihelp)
    mgr.setup_driver(positions)

    # Setup, build and normalize OVERLAP integral
    ovlpint = new_overlap(**dd)
    ovlpint.build(mgr.driver)
    ovlpint.normalize()

    # Setup, build and normalize DIPOLE integral
    dipint = new_dipint(**dd)
    dipint.build(mgr.driver)
    dipint.normalize(ovlpint.norm)

    ##########################################################################

    # PySCF reference integral, centered at r0
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_r0 = einsum(
        "xij,i,j->xij",
        numpy_to_tensor(mol.intor("int1e_r"), **dd),
        ovlpint.norm,
        ovlpint.norm,
    )

    # Compare with PySCF reference
    assert pyscf_r0.shape == dipint.matrix.shape
    assert pytest.approx(pyscf_r0.cpu(), abs=tol) == dipint.matrix.cpu()

    ##########################################################################

    # Shift r0->rj
    pos = mgr.driver.ihelp.spread_atom_to_orbital(
        positions,
        dim=-2,
        extra=True,
    )
    dipint.shift_r0_rj(ovlpint.matrix, pos)

    # PySCF reference integral, centered at rj
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_rj = einsum(
        "xij,i,j->xij",
        numpy_to_tensor(mol.intor("int1e_r_origj"), **dd),
        ovlpint.norm,
        ovlpint.norm,
    )

    # Compare with PySCF reference
    assert pyscf_rj.shape == dipint.matrix.shape
    assert pytest.approx(pyscf_rj.cpu(), abs=tol) == dipint.matrix.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_shift_r0r0_rjrj(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    ##########################################################################

    # Setup the driver
    mgr = DriverManager(labels.INTDRIVER_LIBCINT, **dd)
    mgr.create_driver(numbers, par, ihelp)
    mgr.setup_driver(positions)

    # Setup, build and normalize OVERLAP integral
    ovlpint = new_overlap(**dd)
    ovlpint.build(mgr.driver)
    ovlpint.normalize()

    # Setup, build and normalize QUADRUPOLE integral
    quadint = new_quadint(**dd)
    quadint.build(mgr.driver)
    quadint.normalize(ovlpint.norm)

    ##########################################################################

    # PySCF reference integral, centered at r0
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_r0r0 = einsum(
        "xij,i,j->xij",
        numpy_to_tensor(mol.intor("int1e_rr"), **dd),
        ovlpint.norm,
        ovlpint.norm,
    )

    # Compare with PySCF reference
    assert pyscf_r0r0.shape == quadint.matrix.shape
    assert pytest.approx(pyscf_r0r0.cpu(), abs=tol) == quadint.matrix.cpu()

    ##########################################################################

    # Setup, build and normalize DIPOLE integral
    dipint = new_dipint(**dd)
    dipint.build(mgr.driver)
    dipint.normalize(ovlpint.norm)

    # Shift r0->rj, always 6 cartesian components
    pos = mgr.driver.ihelp.spread_atom_to_orbital(
        positions,
        dim=-2,
        extra=True,
    )
    quadint.shift_r0r0_rjrj(dipint.matrix, ovlpint.matrix, pos)

    # PySCF reference integral, centered at rj
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_qpint = einsum(
        "xij,i,j->xij",
        numpy_to_tensor(mol.intor("int1e_rr_origj"), **dd),
        ovlpint.norm,
        ovlpint.norm,
    )
    pyscf_qpint = _reduce_9_to_6(pyscf_qpint)

    # Compare with PySCF reference
    assert pyscf_qpint.shape == quadint.matrix.shape
    assert pytest.approx(pyscf_qpint.cpu(), abs=tol) == quadint.matrix.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_traceless(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    ##########################################################################

    # Setup the driver
    mgr = DriverManager(
        labels.INTDRIVER_LIBCINT, force_cpu_for_libcint=True, **dd
    )
    mgr.create_driver(numbers, par, ihelp)
    mgr.setup_driver(positions)

    i = Integrals(mgr, intlevel=labels.INTLEVEL_QUADRUPOLE, **dd)
    i.build_overlap(positions)
    i.build_dipole(positions)
    i.build_quadrupole(positions)

    assert i.overlap is not None
    assert i.dipole is not None
    assert i.quadrupole is not None

    # PySCF reference integral, centered at rj
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_qp = einsum(
        "xij,i,j->xij",
        numpy_to_tensor(mol.intor("int1e_rr_origj"), **dd),
        i.overlap.norm,
        i.overlap.norm,
    )

    # Remove symmetry-equivalent components
    pyscf_qp = _reduce_9_to_6(pyscf_qp)

    # Make traceless
    tr = 0.5 * (pyscf_qp[0] + pyscf_qp[2] + pyscf_qp[5])
    pyscf_qp = torch.stack(
        [
            1.5 * pyscf_qp[0] - tr,
            1.5 * pyscf_qp[1],
            1.5 * pyscf_qp[2] - tr,
            1.5 * pyscf_qp[3],
            1.5 * pyscf_qp[4],
            1.5 * pyscf_qp[5] - tr,
        ]
    )

    # Compare with PySCF reference
    assert pyscf_qp.shape == i.quadrupole.matrix.shape
    assert pytest.approx(pyscf_qp.cpu(), abs=tol) == i.quadrupole.matrix.cpu()
