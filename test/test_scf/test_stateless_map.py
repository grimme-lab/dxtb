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
    opts = {"verbosity": 0, "scf_mode": "implicit", "scp_mode": scp_mode}
    calc = Calculator(m["numbers"].to(DEVICE), par, opts=opts, **DD)
    calc.singlepoint(m["positions"].to(**DD), torch.tensor(0.0, **DD))
    assert seen == [True]


@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
@pytest.mark.parametrize("mode", ["implicit", "full"])
@pytest.mark.parametrize("create_graph", [False, True])
def test_scf_object_freed_without_gc(
    mode: str,
    create_graph: bool,
    scp_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The SCF object (and with it everything it holds) is freed by reference
    counting alone after forward, gradient and (double) backward.
    """
    import gc
    import weakref

    from dxtb._src.scf.base import BaseSCF

    probes: list[weakref.ref] = []
    orig = BaseSCF.__call__

    def spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        probes.append(weakref.ref(self))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(BaseSCF, "__call__", spy)

    m = mols["H2O"]
    opts = {"verbosity": 0, "scf_mode": mode, "scp_mode": scp_mode}
    calc = Calculator(m["numbers"].to(DEVICE), GFN1_XTB, opts=opts, **DD)
    pos = m["positions"].to(**DD).clone().requires_grad_(True)

    gc.collect()
    gc.disable()
    try:
        e = calc.energy(pos, torch.tensor(0.0, **DD))
        (g,) = torch.autograd.grad(e, pos, create_graph=create_graph)
        if create_graph:
            g.sum().backward()
        del e, g
        # the calculator caches results (and hence the graph) until reset
        calc.reset()
        assert len(probes) == 1
        assert probes[0]() is None
    finally:
        gc.enable()


@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
def test_result_hamiltonian_matches_full(scp_mode: str) -> None:
    """Density, Hamiltonian and orbital energies of the results agree."""
    m = mols["H2O"]
    res = {}
    for mode in ("implicit", "full"):
        tol = 1e-12
        opts = {
            "verbosity": 0,
            "scf_mode": mode,
            "scp_mode": scp_mode,
            "f_atol": tol,
            "x_atol": tol,
            "x_atol_max": tol,
        }
        calc = Calculator(m["numbers"].to(DEVICE), GFN1_XTB, opts=opts, **DD)
        res[mode] = calc.singlepoint(
            m["positions"].to(**DD), torch.tensor(0.0, **DD)
        )
    for key in ("hamiltonian", "density", "emo"):
        a, b = getattr(res["implicit"], key), getattr(res["full"], key)
        assert torch.allclose(a, b, atol=1e-8, rtol=0), key


def test_map_holds_only_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The map (stored in the autograd graph) does not keep the SCF's scratch
    buffers (density, eigenvectors, ...) alive.
    """
    import gc
    import types

    checked: list[bool] = []
    orig = SCF.scf

    def spy(self, guess, return_charges=True):  # type: ignore[no-untyped-def]
        g = self.stateless_map()
        d = self._data
        scratch = {
            id(t) for t in (d.density, d.evecs, d.hamiltonian, d.old_density)
        }

        seen: set[int] = set()
        stack: list[object] = [g]
        while stack:
            obj = stack.pop()
            if id(obj) in seen or isinstance(obj, (types.ModuleType, type)):
                continue
            seen.add(id(obj))
            assert id(obj) not in scratch
            if isinstance(obj, torch.Tensor):
                continue
            if isinstance(obj, types.FunctionType):
                stack.extend(c.cell_contents for c in obj.__closure__ or ())
                continue
            stack.extend(gc.get_referents(obj))
        checked.append(True)
        return orig(self, guess, return_charges)

    monkeypatch.setattr(SCF, "scf", spy)

    m = mols["H2O"]
    opts = {"verbosity": 0, "scf_mode": "implicit"}
    calc = Calculator(m["numbers"].to(DEVICE), GFN1_XTB, opts=opts, **DD)
    calc.singlepoint(m["positions"].to(**DD), torch.tensor(0.0, **DD))
    assert checked == [True]
