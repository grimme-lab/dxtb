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
Test overlap build from integral container.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb import integrals as ints
from dxtb._src.exlibs import libcint
from dxtb._src.integral.driver.libcint import IntDriverLibcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("name", ["H2"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_single(dtype: torch.dtype, name: str, force_cpu_for_libcint: bool):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    i = ints.Integrals(
        numbers,
        par,
        ihelp,
        force_cpu_for_libcint=force_cpu_for_libcint,
        intlevel=ints.levels.INTLEVEL_QUADRUPOLE,
        **dd,
    )

    i.setup_driver(positions)
    assert isinstance(i.driver, IntDriverLibcint)
    assert isinstance(i.driver.drv, libcint.LibcintWrapper)

    ################################################

    i.overlap = ints.types.Overlap(**dd)
    i.build_overlap(positions)

    o = i.overlap
    assert o is not None
    assert o.matrix is not None

    ################################################

    i.dipole = ints.types.Dipole(**dd)
    i.build_dipole(positions)

    d = i.dipole
    assert d is not None
    assert d.matrix is not None

    ################################################

    i.quadrupole = ints.types.Quadrupole(**dd)
    i.build_quadrupole(positions)

    q = i.quadrupole
    assert q is not None
    assert q.matrix is not None


@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_batch(
    dtype: torch.dtype, name1: str, name2: str, force_cpu_for_libcint: bool
) -> None:
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )

    ihelp = IndexHelper.from_numbers(numbers, par)
    i = ints.Integrals(
        numbers,
        par,
        ihelp,
        intlevel=ints.levels.INTLEVEL_QUADRUPOLE,
        force_cpu_for_libcint=force_cpu_for_libcint,
        **dd,
    )

    i.setup_driver(positions)
    assert isinstance(i.driver, IntDriverLibcint)
    assert isinstance(i.driver.drv, list)
    assert isinstance(i.driver.drv[0], libcint.LibcintWrapper)
    assert isinstance(i.driver.drv[1], libcint.LibcintWrapper)

    ################################################

    i.overlap = ints.types.Overlap(**dd)
    i.build_overlap(positions)

    o = i.overlap
    assert o is not None
    assert o.matrix is not None

    ################################################

    i.dipole = ints.types.Dipole(**dd)
    i.build_dipole(positions)

    d = i.dipole
    assert d is not None
    assert d.matrix is not None

    ################################################

    i.quadrupole = ints.types.Quadrupole(**dd)
    i.build_quadrupole(positions)

    q = i.quadrupole
    assert q is not None
    assert q.matrix is not None
