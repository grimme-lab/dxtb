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
Correctness matrix for the SCF modes.

Every SCF mode is compared with central finite differences of the quantities
(the reference) for a grid of systems, quantities and derivatives. ``full``
(unrolled) mode is itself one of the tested modes, because its derivatives of
non-variational quantities are not exact. For ``BIG`` systems, position
derivatives are compared as directional derivatives (cost).

Seam under test: the public ``Calculator.singlepoint`` interface and
``torch.autograd``. Nothing inside the SCF is accessed.

Quantities (all reduced to a scalar with fixed per-system weights):

- ``E``: total energy
- ``q``: sum of the atomic charges
- ``q2``: sum of the squared atomic charges
- ``dip``: point-charge dipole moment projected on a fixed direction

Derivatives:

- ``pos``: first derivative w.r.t. the positions
- ``field``: first derivative w.r.t. the electric field (needs libcint)
- ``param``: first derivative w.r.t. one parametrization tensor (``gexp``)
- ``hvp_pos``: Hessian-vector product w.r.t. positions (fixed direction)
- ``mixed``: derivative of the ``pos``-gradient projection w.r.t. the field
  (needs libcint)
"""

from __future__ import annotations

import functools
from typing import Callable

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols

from dxtb import GFN1_XTB, Calculator, ParamModule
from dxtb._src.components.interactions import new_efield
from dxtb._src.exlibs.available import has_libcint

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}

# tightest tolerance at which `full` mode converges for all systems below
SCF_TOL = 1e-12

TOL_FIRST_ABS = 1e-8
TOL_FIRST_REL = 1e-7  # relative to max|ref|, used if it is larger
TOL_SECOND = 1e-6

MODES = ["implicit", "full"]
REFERENCE = "full"  # only used to evaluate values for finite differences

# name -> (molecule names, total charge)
SYSTEMS: dict[str, tuple[list[str], float]] = {
    "H2": (["H2"], 0.0),
    "LiH": (["LiH"], 0.0),
    "H2O": (["H2O"], 0.0),
    "H2O+": (["H2O"], 1.0),
    "LYS_xao": (["LYS_xao"], 0.0),
    "batch_H2O_CH4": (["H2O", "CH4"], 0.0),
    "batch_H2_LYS": (["H2", "LYS_xao"], 0.0),
    # 44 SCF iterations (slowest of all samples), but a HOMO-LUMO gap of only
    # 4.6 mEh: strongly fractional occupations, i.e., it probes the derivatives
    # of the Fermi smearing rather than the SCF
    "slow": (["tmpda"], 0.0),
    # 24 SCF iterations (second slowest), well gapped
    "slow_gap": (["LYS_xao_dist"], 0.0),
}

# systems where only a reduced set of cells is run (cost)
BIG = ("LYS_xao", "batch_H2_LYS", "slow", "slow_gap")

QUANTITIES = ["E", "q", "q2", "dip"]
FIRST = ["pos", "field", "param"]
SECOND = [("hvp_pos", "E"), ("hvp_pos", "q2"), ("mixed", "E")]

FIELD0 = (0.002, -0.001, 0.003)
DIR_C = (0.3, -0.5, 0.7)  # dipole projection

##############################################################################


def _opts(mode: str) -> dict:
    return {
        "verbosity": 0,
        "maxiter": 300,
        "scf_mode": mode,
        "f_atol": SCF_TOL,
        "x_atol": SCF_TOL,
        "x_atol_max": SCF_TOL,
        "force_convergence": True,
    }


def build(system: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Numbers, positions and charge of a system (packed for batches)."""
    names, chrg = SYSTEMS[system]
    nums = [mols[n]["numbers"] for n in names]
    poss = [mols[n]["positions"].to(**DD) for n in names]
    if len(names) == 1:
        return nums[0].to(DEVICE), poss[0], torch.tensor(chrg, **DD)
    return (
        pack(nums).to(DEVICE),
        pack(poss),
        torch.full((len(names),), chrg, **DD),
    )


def weights(nb: int) -> torch.Tensor:
    """Distinct per-system weights, so errors of one system cannot cancel."""
    return 1.0 + 0.5 * torch.arange(nb, **DD)


def quantity(calc, res, pos: torch.Tensor, name: str) -> torch.Tensor:
    """Per-system value of a quantity from a ``singlepoint`` result."""
    if name == "E":
        return res.total.sum(-1)

    q = calc.ihelp.reduce_orbital_to_atom(res.charges.mono)
    if name == "q":
        return q.sum(-1)
    if name == "q2":
        return (q**2).sum(-1)
    if name == "dip":
        mu = (q.unsqueeze(-1) * pos).sum(-2)
        return mu @ torch.tensor(DIR_C, **DD)
    raise ValueError(name)


def direction(shape: tuple[int, ...], numbers: torch.Tensor) -> torch.Tensor:
    """Fixed pseudo-random direction, zero on padding atoms."""
    gen = torch.Generator().manual_seed(1234)
    d = torch.randn(shape, generator=gen, dtype=torch.double).to(DEVICE)
    return d * (numbers != 0).unsqueeze(-1)


class Failed:
    """Stores the exception of a cell that could not be evaluated."""

    def __init__(self, exc: BaseException) -> None:
        # keep only text: a traceback would keep autograd graphs alive
        self.text = f"{type(exc).__name__}: {str(exc)[:120]}"

    def __repr__(self) -> str:
        return self.text


def _try(fcn: Callable[[], torch.Tensor]) -> torch.Tensor | Failed:
    try:
        return fcn().detach().clone()
    except Exception as e:  # pylint: disable=broad-exception-caught
        return Failed(e)


def _cells_for(system: str) -> list[tuple[str, str]]:
    """The (derivative, quantity) cells that are run for a system."""
    quants = ["E", "q2"] if system in BIG else QUANTITIES
    cells = [(d, q) for d in FIRST for q in quants]
    cells += [c for c in SECOND if not (system in BIG and c[0] == "mixed")]
    return cells


@functools.lru_cache(maxsize=None)
def compute(
    mode: str, system: str
) -> dict[tuple[str, str], torch.Tensor | Failed]:
    """
    Evaluate all cells of one (mode, system) pair from one forward pass.

    Only detached numbers are cached, never graphs.
    """
    numbers, pos0, chrg = build(system)
    nb = 1 if numbers.dim() == 1 else numbers.shape[0]
    w = weights(nb)
    dpos = direction(tuple(pos0.shape), numbers)

    pos = pos0.clone().requires_grad_(True)
    field = torch.tensor(FIELD0, **DD).requires_grad_(True)
    par = ParamModule(GFN1_XTB, **DD)
    par.set_differentiable("charge", "effective", "gexp")
    gexp = par.get("charge", "effective", "gexp")

    opts = _opts(mode)
    kwargs = {"interaction": new_efield(field)} if has_libcint else {}
    calc = Calculator(numbers, par, opts=opts, **kwargs, **DD)

    out: dict[tuple[str, str], torch.Tensor | Failed] = {}
    try:
        res = calc.singlepoint(pos, chrg)
        fs = {q: (quantity(calc, res, pos, q) * w).sum() for q in QUANTITIES}
    except Exception as e:  # pylint: disable=broad-exception-caught
        fail = Failed(e)
        return {c: fail for c in _cells_for(system)}

    wrt = {"pos": pos, "field": field, "param": gexp}

    def grad(f, x, **kw):
        return torch.autograd.grad(f, x, retain_graph=True, **kw)[0]

    for d, q in _cells_for(system):
        if d in FIRST:
            out[(d, q)] = _try(lambda: grad(fs[q], wrt[d]))
        elif d == "hvp_pos":

            def hvp(q=q) -> torch.Tensor:
                g = grad(fs[q], pos, create_graph=True)
                return grad((g * dpos).sum(), pos)

            out[(d, q)] = _try(hvp)
        elif d == "mixed":

            def mixed(q=q) -> torch.Tensor:
                g = grad(fs[q], pos, create_graph=True)
                return grad((g * dpos).sum(), field)

            out[(d, q)] = _try(mixed)
    return out


##############################################################################
# reference: central finite differences of the quantities (values only)


def _scalars(system: str, pos=None, field=None, gexp=None) -> dict:
    """Weighted scalar quantities at given inputs (no derivatives)."""
    numbers, pos0, chrg = build(system)
    nb = 1 if numbers.dim() == 1 else numbers.shape[0]
    w = weights(nb)
    pos = pos0 if pos is None else pos
    field = torch.tensor(FIELD0, **DD) if field is None else field
    par = ParamModule(GFN1_XTB, **DD)  # same object type as in `compute`
    if gexp is not None:
        with torch.no_grad():
            par.get("charge", "effective", "gexp").fill_(gexp)
    kw = {"interaction": new_efield(field)} if has_libcint else {}
    calc = Calculator(numbers, par, opts=_opts(REFERENCE), **kw, **DD)
    res = calc.singlepoint(pos, chrg)
    return {
        q: (quantity(calc, res, pos, q) * w).sum().item() for q in QUANTITIES
    }


def _stencil1(fn: Callable[[float], float], h: float) -> float:
    """Fourth-order central first derivative at 0."""
    return (8.0 * (fn(h) - fn(-h)) - (fn(2 * h) - fn(-2 * h))) / (12.0 * h)


def _stencil2(fn: Callable[[float], float], h: float) -> float:
    """Fourth-order central second derivative at 0."""
    return (
        -fn(2 * h) + 16.0 * fn(h) - 30.0 * fn(0.0) + 16.0 * fn(-h) - fn(-2 * h)
    ) / (12.0 * h**2)


def _vec(fn: Callable[[float], dict], h: float, second: bool = False) -> dict:
    """Apply a stencil to every quantity of a dict-valued function."""
    cache: dict[float, dict] = {}

    def at(t: float) -> dict:
        if t not in cache:
            cache[t] = fn(t)
        return cache[t]

    st = _stencil2 if second else _stencil1
    return {q: st(lambda t, q=q: at(t)[q], h) for q in QUANTITIES}


@functools.lru_cache(maxsize=None)
def reference(system: str) -> dict[tuple[str, str], torch.Tensor]:
    """
    Finite-difference reference for all cells of a system.

    For small systems ``pos`` holds the full gradient. For ``BIG`` systems it
    holds the directional derivative ``dpos . grad`` (0-dim), and the mode value
    is projected the same way. ``hvp_pos`` is the scalar ``dpos . H . dpos`` and
    ``mixed`` the vector ``d/dF (dpos . grad_pos E)``.
    """
    numbers, pos0, _ = build(system)
    dpos = direction(tuple(pos0.shape), numbers)
    field0 = torch.tensor(FIELD0, **DD)
    out: dict[tuple[str, str], torch.Tensor] = {}
    cells = _cells_for(system)
    quants = sorted({q for _, q in cells})

    def put(deriv: str, vals: dict) -> None:
        for q in quants:
            out[(deriv, q)] = torch.as_tensor(vals[q], **DD)

    # pos
    if system in BIG:
        put("pos", _vec(lambda t: _scalars(system, pos=pos0 + t * dpos), 1e-3))
    else:
        g = {q: torch.zeros_like(pos0) for q in QUANTITIES}
        for idx in range(pos0.numel()):
            e = torch.zeros_like(pos0)
            e.view(-1)[idx] = 1.0
            v = _vec(lambda t: _scalars(system, pos=pos0 + t * e), 1e-3)
            for q in QUANTITIES:
                g[q].view(-1)[idx] = v[q]
        put("pos", g)

    # field (3) and parameter (1)
    if has_libcint:
        gf = {q: torch.zeros(3, **DD) for q in QUANTITIES}
        for i in range(3):
            e = torch.zeros(3, **DD)
            e[i] = 1.0
            v = _vec(lambda t: _scalars(system, field=field0 + t * e), 1e-4)
            for q in QUANTITIES:
                gf[q][i] = v[q]
        put("field", gf)
    g0 = ParamModule(GFN1_XTB, **DD).get("charge", "effective", "gexp").item()
    put(
        "param",
        _vec(lambda t: _scalars(system, gexp=g0 + t), 1e-3),
    )

    # second derivatives
    put(
        "hvp_pos",
        _vec(
            lambda t: _scalars(system, pos=pos0 + t * dpos), 2e-3, second=True
        ),
    )
    if has_libcint and system not in BIG:
        # E is strongly non-linear in the field: small field step (h scan in
        # the development notes: 1e-3 -> 1e-7 error, 1e-4 -> 1e-11). Position step 2e-3: larger
        # steps lose to truncation, smaller ones to SCF noise (hvp h scan, the development notes)
        hp, hf = 2e-3, 3e-4
        mixed = torch.zeros(3, **DD)
        for i in range(3):
            e = torch.zeros(3, **DD)
            e[i] = 1.0

            def dt(f: float, e=e) -> float:
                return _stencil1(
                    lambda t: _scalars(
                        system, pos=pos0 + t * dpos, field=field0 + f * e
                    )["E"],
                    hp,
                )

            mixed[i] = _stencil1(dt, hf)
        out[("mixed", "E")] = mixed
    return out


def _tol(ref: torch.Tensor, first: bool) -> float:
    if not first:
        return TOL_SECOND
    return max(TOL_FIRST_ABS, TOL_FIRST_REL * ref.abs().max().item())


def project(system: str, deriv: str, val: torch.Tensor) -> torch.Tensor:
    """Bring a mode value into the form of the reference."""
    numbers, pos0, _ = build(system)
    dpos = direction(tuple(pos0.shape), numbers)
    if deriv == "hvp_pos" or (deriv == "pos" and system in BIG):
        return (val * dpos).sum()
    return val


##############################################################################
# the matrix

# Cells that are known to fail, keyed by (mode, system, derivative, quantity).
# `strict=True` forces removal of the marker once a cell is fixed; cells within
# 3x of the tolerance are not strict (their outcome depends on FD/SCF noise).
KNOWN_FAILURES: dict[tuple[str, str, str, str], tuple[str, bool]] = {
    ("full", "H2", "pos", "dip"): (
        "unrolled derivative of a non-variational quantity is inexact (err 1.09e-08)",
        False,
    ),
    ("full", "LiH", "pos", "q2"): (
        "unrolled derivative of a non-variational quantity is inexact (err 2.70e-08)",
        False,
    ),
    ("full", "LiH", "pos", "dip"): (
        "unrolled derivative of a non-variational quantity is inexact (err 5.25e-08)",
        False,
    ),
    ("full", "LiH", "field", "q2"): (
        "unrolled derivative of a non-variational quantity is inexact (err 4.58e-07)",
        False,
    ),
    ("full", "LiH", "field", "dip"): (
        "unrolled derivative of a non-variational quantity is inexact (err 8.92e-07)",
        False,
    ),
    ("full", "batch_H2O_CH4", "hvp_pos", "q2"): (
        "unrolled derivative of a non-variational quantity is inexact (err 3.93e-03)",
        True,
    ),
    ("full", "LYS_xao", "hvp_pos", "q2"): (
        "unrolled derivative of a non-variational quantity is inexact (err 4.16e-03)",
        True,
    ),
    ("full", "batch_H2_LYS", "hvp_pos", "q2"): (
        "unrolled derivative of a non-variational quantity is inexact (err 2.72e-03)",
        True,
    ),
    ("implicit", "slow", "hvp_pos", "E"): (
        "Fermi smearing: HOMO-LUMO gap 4.6 mEh, fractional frontier occupations (inexact in every mode)",
        True,
    ),
    ("implicit", "slow", "hvp_pos", "q2"): (
        "Fermi smearing: HOMO-LUMO gap 4.6 mEh, fractional frontier occupations (inexact in every mode)",
        True,
    ),
    ("full", "slow", "pos", "q2"): (
        "Fermi smearing: HOMO-LUMO gap 4.6 mEh, fractional frontier occupations (inexact in every mode)",
        True,
    ),
    ("full", "slow", "field", "q2"): (
        "Fermi smearing: HOMO-LUMO gap 4.6 mEh, fractional frontier occupations (inexact in every mode)",
        True,
    ),
    ("full", "slow", "param", "q2"): (
        "Fermi smearing: HOMO-LUMO gap 4.6 mEh, fractional frontier occupations (inexact in every mode)",
        True,
    ),
    ("full", "slow", "hvp_pos", "E"): (
        "Fermi smearing: HOMO-LUMO gap 4.6 mEh, fractional frontier occupations (inexact in every mode)",
        True,
    ),
    ("full", "slow", "hvp_pos", "q2"): (
        "Fermi smearing: HOMO-LUMO gap 4.6 mEh, fractional frontier occupations (inexact in every mode)",
        True,
    ),
    ("implicit", "slow_gap", "hvp_pos", "q2"): (
        "FD-reference-noise limited, slow-converging system (err 2.13e-06, tol 1e-06)",
        False,
    ),
    ("full", "slow_gap", "pos", "q2"): (
        "unrolled derivative of a non-variational quantity is inexact (err 4.22e-03)",
        False,
    ),
    ("full", "slow_gap", "field", "q2"): (
        "unrolled derivative of a non-variational quantity is inexact (err 6.21e+00)",
        False,
    ),
    ("full", "slow_gap", "param", "q2"): (
        "unrolled derivative of a non-variational quantity is inexact (err 1.14e-03)",
        False,
    ),
    ("full", "slow_gap", "hvp_pos", "q2"): (
        "unrolled derivative of a non-variational quantity is inexact (err 5.41e+06)",
        False,
    ),
}


def _all_cells() -> list:
    params = []
    for mode in MODES:
        for system in SYSTEMS:
            for d, q in _cells_for(system):
                marks = []
                if d in ("field", "mixed"):
                    marks.append(
                        pytest.mark.skipif(
                            not has_libcint, reason="libcint not available"
                        )
                    )
                key = (mode, system, d, q)
                if key in KNOWN_FAILURES:
                    reason, strict = KNOWN_FAILURES[key]
                    marks.append(
                        pytest.mark.xfail(strict=strict, reason=reason)
                    )
                params.append(
                    pytest.param(
                        mode,
                        system,
                        d,
                        q,
                        marks=marks,
                        id=f"{mode}-{system}-{d}-{q}",
                    )
                )
    return params


@pytest.mark.parametrize("mode,system,deriv,q", _all_cells())
def test_matrix(mode: str, system: str, deriv: str, q: str) -> None:
    """Derivative of a mode agrees with the finite-difference reference."""
    cell = (deriv, q)
    ref = reference(system)[cell]

    val = compute(mode, system)[cell]
    assert isinstance(val, torch.Tensor), f"{mode} failed: {val}"
    val = project(system, deriv, val)

    assert torch.isfinite(ref).all(), "reference is not finite"
    assert torch.isfinite(val).all(), f"{mode} value is not finite"

    first = deriv in FIRST
    err = (val - ref).abs().max().item()
    tol = _tol(ref, first)
    assert err < tol, f"max abs error {err:.3e} (tol {tol:.1e})"


@pytest.mark.parametrize("mode", MODES)
def test_float32_smoke(mode: str) -> None:
    """Single precision runs and roughly agrees with the double reference."""
    numbers, pos0, chrg = build("H2O")
    # energy gradients are variational, hence exact in every mode
    ref = compute("full", "H2O")

    dd32 = {"device": DEVICE, "dtype": torch.float32}
    opts = {"verbosity": 0, "maxiter": 100, "scf_mode": mode}
    pos = pos0.to(**dd32).requires_grad_(True)
    kw = (
        {"interaction": new_efield(torch.tensor(FIELD0, **dd32))}
        if has_libcint
        else {}
    )
    calc = Calculator(numbers, GFN1_XTB, opts=opts, **kw, **dd32)
    res = calc.singlepoint(pos, chrg.to(**dd32))

    e = res.total.sum(-1)
    (g,) = torch.autograd.grad(e, pos)
    assert isinstance(ref[("pos", "E")], torch.Tensor)
    assert (g.double() - ref[("pos", "E")]).abs().max().item() < 5e-4
