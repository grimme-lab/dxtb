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
Test free energy calculation.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, Calculator
from dxtb._src.constants import labels
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .element_sets import REPRESENTATIVE_ELEMENTS, reps_both_dtypes
from .uhf_table import uhf_anion, uhf_cation

opts = {
    "fermi_etemp": 300,
    "fermi_maxiter": 500,
    "scf_mode": labels.SCF_MODE_IMPLICIT,
    "scp_mode": "potential",  # important for atoms (better convergence)
    "verbosity": 0,
}


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype, number", reps_both_dtypes())
@pytest.mark.parametrize("partition", ["equal", "atomic"])
def test_element(dtype: torch.dtype, partition: str, number: int) -> None:
    """
    Comparison of implicit vs. full (unrolled) SCF.

    Different solvers (Broyden vs. Anderson) stop anywhere within the SCF
    tolerance, so the modes can only agree to that tolerance. In double
    precision, both are converged tightly and compared with 1e-8; in single
    precision the tolerance is ten times the SCF tolerance.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}
    scf_tol = 1e-5 if dtype == torch.float32 else 1e-10
    tol = 10 * scf_tol if dtype == torch.float32 else 1e-8

    numbers = torch.tensor([number], device=DEVICE)
    positions = torch.zeros((1, 3), **dd)
    charges = torch.tensor(0.0, **dd)

    options = dict(
        opts,
        **{
            "f_atol": scf_tol,
            "x_atol": scf_tol,
            "fermi_partition": partition,
            "fermi_thresh": 1e-4 if dtype == torch.float32 else 1e-10,
            "maxiter": 100,
        },
    )

    o = dict(options, **{"scf_mode": "implicit"})
    calc1 = Calculator(numbers, GFN1_XTB, opts=o, **dd)
    result1 = calc1.singlepoint(positions, charges)

    o = dict(options, **{"scf_mode": "implicit_nonpure"})
    calc2 = Calculator(numbers, GFN1_XTB, opts=o, **dd)
    result2 = calc2.singlepoint(positions, charges)

    f1 = result1.fenergy.cpu()
    f2 = result2.fenergy.cpu()
    assert pytest.approx(f1, abs=tol) == f2

    e1 = result1.total.sum(-1).cpu()
    e2 = result2.total.sum(-1).cpu()
    assert pytest.approx(e1, abs=tol) == e2


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_unique(dtype: torch.dtype) -> None:
    """Different free energies for different atoms."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    def fcn(number):
        numbers = torch.tensor([number], device=DEVICE)
        positions = torch.zeros((1, 3), **dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(
            opts,
            **{
                "f_atol": 1e-5 if dtype == torch.float32 else 1e-6,
                "x_atol": 1e-5 if dtype == torch.float32 else 1e-6,
                "fermi_thresh": 1e-4 if dtype == torch.float32 else 1e-10,
                "damp": 0.95,
            },
        )
        calc = Calculator(numbers, GFN1_XTB, opts=options, **dd)
        result = calc.singlepoint(positions, charges)
        return result.fenergy

    fenergies = [fcn(n).item() for n in REPRESENTATIVE_ELEMENTS]
    unique = set(fenergies)
    assert len(unique) > 5


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_cation(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    def fcn(number):
        numbers = torch.tensor([number], device=DEVICE)
        positions = torch.zeros((1, 3), **dd)
        charges = torch.tensor(1.0, **dd)
        spin = uhf_cation[number - 1]

        options = dict(
            opts,
            **{
                "f_atol": 1e-5,  # avoids Jacobian inversion error
                "x_atol": 1e-5,  # avoids Jacobian inversion error
                "fermi_thresh": 1e-4 if dtype == torch.float32 else 1e-10,
                "damp": 0.9,
            },
        )
        calc = Calculator(numbers, GFN1_XTB, opts=options, **dd)
        result = calc.singlepoint(positions, charges, spin)
        return result.fenergy

    # no (valence) electrons OR gold
    _exclude = [1, 3, 11, 19, 37, 55, 79]
    numbers = [i for i in REPRESENTATIVE_ELEMENTS if i not in _exclude]

    fenergies = [fcn(n).item() for n in numbers]
    unique = set(fenergies)
    assert len(unique) > 5


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_anion(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    def fcn(number):
        numbers = torch.tensor([number], device=DEVICE)
        positions = torch.zeros((1, 3), **dd)
        charges = torch.tensor(-1.0, **dd)
        spin = uhf_anion[number - 1]

        options = dict(
            opts,
            **{
                "f_atol": 1e-5,  # avoid Jacobian inversion error
                "x_atol": 1e-5,  # avoid Jacobian inversion error
                "fermi_thresh": 1e-4 if dtype == torch.float32 else 1e-10,
            },
        )
        calc = Calculator(numbers, GFN1_XTB, opts=options, **dd)
        result = calc.singlepoint(positions, charges, spin)
        return result.fenergy

    # Helium doesn't have enough orbitals for negative charge,
    # SCF does not converge (in tblite too)
    _exclude = [2, 21, 22, 23, 25, 43, 57, 58, 59]
    numbers = [i for i in REPRESENTATIVE_ELEMENTS if i not in _exclude]

    fenergies = [fcn(n).item() for n in numbers]
    unique = set(fenergies)
    assert len(unique) > 5
