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
The stateless fixed-point map must be bit-identical to the iteration of the
SCF object, and must not touch the SCF object's data.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.data.molecules import mols

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.constants import defaults
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.scf.implicit import SelfConsistentFieldImplicit as SCF

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}


@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_map_equals_iteration(
    scp_mode: str, gfn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stateless_map()` gives exactly the same values as `_fcn`."""
    if gfn == "gfn2" and not has_libcint:
        pytest.skip("libcint not available")

    seen: list[bool] = []
    orig = SCF.scf

    def spy(self, guess, return_charges=True):  # type: ignore[no-untyped-def]
        g = self.stateless_map()
        gen = torch.Generator().manual_seed(0)
        noise = torch.randn(guess.shape, generator=gen, dtype=guess.dtype)
        if guess.dim() >= 2 and guess.shape[-1] == guess.shape[-2]:
            noise = (noise + noise.mT) / 2  # Fock matrix stays symmetric
        pert = guess + 1e-3 * noise * (guess != defaults.PADNZ)

        iter_before = self._data.iter
        for x in (guess, pert):
            assert torch.equal(g(x.clone()), self._fcn(x.clone()))
        seen.append(True)
        # the map must not count iterations or write the SCF data
        assert self._data.iter == iter_before + 2
        return orig(self, guess, return_charges)

    monkeypatch.setattr(SCF, "scf", spy)

    m = mols["H2O"]
    par = GFN1_XTB if gfn == "gfn1" else GFN2_XTB
    opts = {"verbosity": 0, "scf_mode": "nonpure", "scp_mode": scp_mode}
    calc = Calculator(m["numbers"].to(DEVICE), par, opts=opts, **DD)
    calc.singlepoint(m["positions"].to(**DD), torch.tensor(0.0, **DD))
    assert seen == [True]
