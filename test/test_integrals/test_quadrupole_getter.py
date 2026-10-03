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
"""
Public ``get_quadrupole`` of the calculator: shape, tracelessness, batch
consistency, agreement between integral drivers and the error for a
calculator whose integral level does not include the quadrupole integral.
The comparison with tblite is in ``test_tblite_reference.py``.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

import dxtb
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

OPTS = {"verbosity": 0, "scf_mode": "full", "f_atol": 1e-11, "x_atol": 1e-11}


def _calc(numbers, dd: DD, **opts):
    return dxtb.calculators.GFN2Calculator(numbers, opts={**OPTS, **opts}, **dd)


def test_quadrupole_shape_traceless_and_alias() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(**dd)

    calc = _calc(numbers, dd)
    quad = calc.get_quadrupole(positions)
    assert quad.shape == (6,)
    assert abs(float(quad[0] + quad[2] + quad[5])) < 1e-10
    assert torch.equal(calc.get_quadrupole_moment(positions), quad)


def test_quadrupole_batch_equals_single() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    names = ["SiH4", "H2O"]
    numbers = pack([samples[n]["numbers"] for n in names]).to(DEVICE)
    positions = pack([samples[n]["positions"] for n in names]).to(**dd)

    batch = _calc(numbers, dd).get_quadrupole(positions)
    assert batch.shape == (2, 6)

    for i, name in enumerate(names):
        single = _calc(samples[name]["numbers"].to(DEVICE), dd).get_quadrupole(
            samples[name]["positions"].to(**dd)
        )
        assert torch.allclose(batch[i], single, atol=1e-8, rtol=0.0)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("name", ["LiH", "H2O", "SiH4"])
def test_quadrupole_pytorch_matches_libcint_driver(name: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    ref = _calc(numbers, dd, int_driver="libcint").get_quadrupole(positions)
    for algorithm in ("md", "os"):
        out = _calc(
            numbers, dd, int_driver="pytorch", int_algorithm=algorithm
        ).get_quadrupole(positions)
        assert torch.allclose(out, ref, atol=1e-8, rtol=0.0)


def test_quadrupole_requires_quadrupole_integral() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(**dd)

    calc = dxtb.calculators.GFN1Calculator(numbers, opts={"verbosity": 0}, **dd)
    with pytest.raises(RuntimeError, match="quadrupole integral"):
        calc.get_quadrupole(positions)
