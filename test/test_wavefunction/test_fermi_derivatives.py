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
Derivatives of the Fermi occupations without finite differences.

The true occupations fulfil exact identities (the electrons are conserved, a
uniform shift of all orbital energies leaves them unchanged, and the first
derivative has a closed form). ``k`` differentiable Newton steps reproduce
all derivatives up to order ``2**k - 1``, i.e., these identities must hold up
to that order and (generically) be violated at order ``2**k``.

All derivatives are taken along a direction ``v`` with nested autograd.
"""

from __future__ import annotations

from contextlib import contextmanager

import mpmath
import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from tad_mctc.units import KELVIN2AU
from torch.func import jacrev, jvp

from dxtb._src.constants import defaults
from dxtb._src.typing import DD
from dxtb._src.wavefunction import filling

from ..conftest import DEVICE

dd: DD = {"device": DEVICE, "dtype": torch.double}


def _preload_jvp_decompositions() -> None:
    """
    Forward-mode transforms load their decompositions lazily, which needs
    TorchScript. The tests disable it (see ``conftest.py``), so load them now.
    """
    state = torch.jit._state  # type: ignore
    was_enabled = state._enabled.enabled
    state.enable()
    try:
        from torch._decomp import decompositions_for_jvp  # noqa: F401
    finally:
        if not was_enabled:
            state.disable()


_preload_jvp_decompositions()

TEMPERATURES = [300.0, 5000.0, 25000.0]
STEPS = [0, 1, 2]

# Tolerance of the identities relative to the scale of the derivatives.
RTOL = 1e-8
# Additional absolute tolerance in units of THR / kT**n for the n-th
# derivative. It covers channels with (almost) no thermal weight, for which the
# derivative of the Fermi energy is dropped (see `_diff_floor`), an error of the
# order of the (negligible) derivatives themselves.
ATOL = 1e-2
# Violation of the identity at order 2**k relative to the scale, required in at
# least one channel with fractional occupations (sum of f(1-f) of at least
# MIN_WEIGHT) and curvature (see `curvature`, at least MIN_CURVATURE).
LOWER = 1e-8
MIN_WEIGHT = 1e-2
MIN_CURVATURE = 1e-4


###############################################################################
# helpers
###############################################################################


def directional_derivs(fn, eps0: torch.Tensor, v: torch.Tensor, nmax: int):
    """
    Derivatives of every output element along `v` at ``t = 0``, i.e.,
    ``out[n]`` is the n-th derivative with respect to ``t`` of
    ``fn(eps0 + t * v)``, with nested forward-mode autograd.
    """
    one = torch.ones((), **dd)

    def nth(n: int):
        def f(t: torch.Tensor) -> torch.Tensor:
            if n == 0:
                return fn(eps0 + t * v)
            return jvp(nth(n - 1), (t,), (one,))[1]

        return f

    t0 = torch.zeros((), **dd)
    return [nth(n)(t0).detach() for n in range(nmax + 1)]


@contextmanager
def fixed_fermi_energy(offset: float = 0.0):
    """
    Return the Fermi energy of the first search for all later calls (plus an
    optional offset). The search is detached in the implementation, i.e., the
    derivatives do not depend on it. Skipping it allows forward-mode
    autograd, which is much cheaper for high orders.
    """
    real = filling._fermi_energy_search
    stored: list = []

    def search(*args, **kwargs):
        if not stored:
            (ref, e_fermi), invalid, flags = real(*args, **kwargs)
            stored.append(((ref, e_fermi + offset), invalid, flags))
        return stored[0]

    filling._fermi_energy_search = search
    try:
        yield
    finally:
        filling._fermi_energy_search = real


class Case:
    """Occupation problem: electrons per channel, energies and padding."""

    def __init__(
        self,
        nab: list[list[float]],
        emo: torch.Tensor,
        mask: torch.Tensor | None = None,
        generic: bool = True,
    ):
        self.nab = torch.tensor(nab, **dd)
        self.emo = emo.to(**dd)
        self.mask = mask
        # order 2**k is only violated if the spectrum is asymmetric
        self.generic = generic

    def kt(self, ktemp: float) -> torch.Tensor:
        return torch.tensor(ktemp * KELVIN2AU, **dd)

    def occupation(
        self, x: torch.Tensor, kt: torch.Tensor, diff_order: int | None = None
    ) -> torch.Tensor:
        return filling.get_fermi_occupation(
            self.nab, x, kt, mask=self.mask, diff_order=diff_order
        )

    def valid(self) -> torch.Tensor:
        if self.mask is None:
            return torch.ones_like(self.emo, dtype=torch.bool)
        return self.mask != 0

    def direction(self, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(self.emo.shape, generator=g, dtype=torch.double)
        return torch.where(self.valid(), v.to(DEVICE), 0.0)

    def shift(self) -> torch.Tensor:
        return self.valid().to(**dd)


def _random_emo(seed: int, shape: tuple[int, ...], scale: float = 0.3):
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(shape, generator=g, dtype=torch.double) * scale
    return e.sort(-1).values.to(DEVICE)


def _cases() -> dict[str, Case]:
    cases = {}

    # gap of about 20 kT at 300 K, i.e., f(1-f) ~ 3e-5 at the Fermi energy
    e = torch.tensor([-0.9, -0.8, -0.65, -0.62, -0.6, -0.56]).expand(1, 2, -1)
    cases["gap"] = Case([[4.0, 4.0]], e)

    cases["fractional"] = Case([[2.5, 1.5]], _random_emo(1, (1, 2, 7)))

    # threefold degenerate level at the Fermi energy with fractional filling
    e = torch.tensor([-0.6, -0.3, -0.3, -0.3, 0.4, 0.8]).expand(1, 2, -1)
    cases["degenerate"] = Case([[2.0, 3.0]], e)

    # Both channels have integer electrons, and the fourth-order violation of
    # this particular spectrum at 5000 K is only 6e-12 of the scale, i.e., it
    # does not qualify as a generic case for the lower bound.
    cases["open_shell"] = Case(
        [[3.0, 1.0]], _random_emo(2, (1, 2, 6)), generic=False
    )

    # two systems with 6 and 4 orbitals, padding with energies inside the
    # spectrum (must never be occupied)
    e = _random_emo(3, (2, 2, 6))
    e[1, :, 4:] = 0.05
    mask = torch.ones(2, 2, 6, **dd)
    mask[1, :, 4:] = 0.0
    cases["batch_padding"] = Case([[2.0, 2.0], [1.0, 1.0]], e, mask)

    cases["random"] = Case([[3.0, 3.0]], _random_emo(4, (1, 2, 8)))
    return cases


CASES = _cases()


def order_of(steps: int) -> int:
    """Highest exact order of `steps` Newton steps (``2**steps - 1``)."""
    return 2**steps - 1


def _run_order(
    case: Case, ktemp: float, order: int, v: torch.Tensor, nmax: int
):
    """Derivatives of the occupations for a requested derivative order."""
    kt = case.kt(ktemp)
    with fixed_fermi_energy():
        case.occupation(case.emo, kt)  # converged Fermi energy at t = 0
        return directional_derivs(
            lambda x: case.occupation(x, kt, order), case.emo, v, nmax
        )


def _run(case: Case, ktemp: float, steps: int, v: torch.Tensor, nmax: int):
    """Derivatives of the occupations with a given number of Newton steps."""
    return _run_order(case, ktemp, order_of(steps), v, nmax)


_CACHE: dict = {}


def derivs(name: str, ktemp: float, steps: int, kind: str, nmax: int):
    """Cached derivatives along a random (``"rand"``) or shift direction."""
    key = (name, ktemp, steps, kind)
    if key not in _CACHE or len(_CACHE[key]) <= nmax:
        case = CASES[name]
        v = case.direction(7) if kind == "rand" else case.shift()
        _CACHE[key] = _run(case, ktemp, steps, v, nmax)
    return _CACHE[key]


THR = min(
    torch.finfo(torch.double).eps ** 0.5, 1e5 * torch.finfo(torch.double).eps
)


def scale_of(d: torch.Tensor) -> torch.Tensor:
    """Sum of the absolute derivatives of all orbitals (per channel)."""
    return d.abs().sum(-1)


def tolerance(d: torch.Tensor, n: int, kt: torch.Tensor) -> torch.Tensor:
    """Tolerance for the n-th derivative of the occupations (per channel)."""
    return RTOL * scale_of(d) + ATOL * THR / kt.item() ** n


def thermal_weight(case: Case, ktemp: float) -> torch.Tensor:
    """Sum of f(1-f) per channel (fractional occupation)."""
    occ = case.occupation(case.emo, case.kt(ktemp)).detach()
    return (occ * (1.0 - occ)).sum(-1)


def exact_occupation(case: Case, ktemp: float):
    """
    Occupations ``f`` and thermal weights ``w = f (1 - f)`` at the exact
    Fermi energy of each channel (bisection in high precision). The weights
    ``occ * (1 - occ)`` from the occupations lose the tails of the occupied
    orbitals to rounding, which the derivatives with respect to the
    electrons depend on in a gap.
    """
    key = ("weights", id(case), ktemp)
    if key in _CACHE:
        return _CACHE[key]

    kt = mpmath.mpf(ktemp * KELVIN2AU)
    emo = case.emo.expand(*case.nab.shape, -1).cpu()
    valid = case.valid().expand(*case.nab.shape, -1).cpu()
    f = torch.zeros(emo.shape, dtype=torch.double)
    w = torch.zeros(emo.shape, dtype=torch.double)

    with mpmath.workdps(120):
        for idx in np.ndindex(*case.nab.shape):
            nel = mpmath.mpf(case.nab[idx].item())
            e = [
                mpmath.mpf(x)
                for x, v in zip(emo[idx].tolist(), valid[idx])
                if v
            ]

            def count(mu):
                return sum(1 / (1 + mpmath.exp((x - mu) / kt)) for x in e)

            lo, hi = min(e) - 100 * kt, max(e) + 100 * kt
            for _ in range(200):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if count(mid) < nel else (lo, mid)
            mu = (lo + hi) / 2

            fs = [1 / (1 + mpmath.exp((x - mu) / kt)) for x in e]
            hs = [1 / (1 + mpmath.exp((mu - x) / kt)) for x in e]
            f[idx][valid[idx]] = torch.tensor(
                [float(x) for x in fs], dtype=torch.double
            )
            w[idx][valid[idx]] = torch.tensor(
                [float(x * y) for x, y in zip(fs, hs)], dtype=torch.double
            )

    _CACHE[key] = f.to(DEVICE), w.to(DEVICE)
    return _CACHE[key]


def curvature(case: Case, ktemp: float) -> torch.Tensor:
    """
    Relative curvature ``kT g'' / g'`` of the number of electrons per
    channel. The violation at order ``2**k`` is proportional to it: for a
    single orbital at ``f = 1/2`` (e.g., fractional electrons at low
    temperature), Newton is exact to higher orders by symmetry.
    """
    f, w = exact_occupation(case, ktemp)
    return (w * (1.0 - 2.0 * f)).sum(-1).abs() / w.sum(-1)


###############################################################################
# tests
###############################################################################


@pytest.mark.parametrize("ktemp", TEMPERATURES)
@pytest.mark.parametrize("name", CASES)
def test_forward_residual(name: str, ktemp: float):
    """The electrons are conserved within the solver threshold."""
    case = CASES[name]
    occ = case.occupation(case.emo, case.kt(ktemp)).detach()

    eps = torch.finfo(torch.double).eps
    thr = min(eps**0.5, 1e5 * eps)
    assert ((occ.sum(-1) - case.nab).abs() <= thr).all()

    # padding is never occupied
    assert (occ[~case.valid()] == 0).all()


def _electron_conservation(
    name: str, ktemp: float, steps: int, lower: bool = True
):
    order = 2**steps
    case = CASES[name]
    kt = case.kt(ktemp)
    d = derivs(name, ktemp, steps, "rand", order)

    for n in range(1, order):
        total = d[n].sum(-1).abs()
        assert (total <= tolerance(d[n], n, kt)).all(), (n, total)

    fractional = thermal_weight(case, ktemp) >= MIN_WEIGHT
    fractional &= curvature(case, ktemp) >= MIN_CURVATURE
    if lower and case.generic and fractional.any():
        # the coefficient depends on the channel and the direction, i.e., the
        # violation has to show up in at least one channel
        ratio = d[order].sum(-1).abs() / scale_of(d[order])
        assert ratio[fractional].max() >= LOWER, ratio


@pytest.mark.parametrize("ktemp", TEMPERATURES)
@pytest.mark.parametrize("name", CASES)
def test_electron_conservation(name: str, ktemp: float):
    """
    The true number of electrons is constant, i.e., all derivatives of the
    sum of the occupations vanish. It is exact up to order 3 with two steps
    and violated at order 4.
    """
    _electron_conservation(name, ktemp, 2)


@pytest.mark.parametrize("ktemp", TEMPERATURES)
@pytest.mark.parametrize("name", CASES)
def test_shift_invariance(name: str, ktemp: float):
    """
    A uniform shift of all orbital energies (padding excluded) does not
    change any occupation.
    """
    order = 2**2 - 1
    ref = derivs(name, ktemp, 2, "rand", order)
    d = derivs(name, ktemp, 2, "shift", order)

    kt = CASES[name].kt(ktemp)
    for n in range(1, order + 1):
        tol = tolerance(ref[n], n, kt).unsqueeze(-1)
        assert (d[n].abs() <= tol).all(), n


def _fermi_reference(case: Case, ktemp: float):
    """
    Fermi energies of the true occupations with an independent root finder and
    the closed form of the Jacobian, ``w_i / kT * (w_j / sum(w) - delta_ij)``.
    """
    kt = case.kt(ktemp).item()
    emo = case.emo.cpu().numpy()
    valid = case.valid().cpu().numpy()
    nab = case.nab.cpu().numpy()

    jac = np.zeros((*emo.shape, emo.shape[-1]))
    for idx in np.ndindex(*emo.shape[:-1]):
        e, m = emo[idx][valid[idx]], valid[idx]

        def excess(mu: float) -> float:
            f = 1.0 / (1.0 + np.exp(np.clip((e - mu) / kt, -700, 700)))
            return f.sum() - nab[idx]

        mu = brentq(
            excess, e.min() - 60 * kt, e.max() + 60 * kt, xtol=1e-15, rtol=1e-15
        )
        f = 1.0 / (1.0 + np.exp(np.clip((e - mu) / kt, -700, 700)))
        w = f * (1.0 - f)
        j = w[:, None] / kt * (w[None, :] / w.sum() - np.eye(len(e)))
        jac[idx][np.ix_(m, m)] = j
    return jac


@pytest.mark.parametrize("ktemp", TEMPERATURES)
@pytest.mark.parametrize("name", CASES)
def test_first_order_oracle(name: str, ktemp: float):
    """Jacobian of the occupations equals the implicit function theorem."""
    case = CASES[name]
    kt = case.kt(ktemp)

    x = case.emo.clone().requires_grad_()
    occ = case.occupation(x, kt)
    jac = torch.stack(
        [
            torch.autograd.grad(o, x, retain_graph=True)[0]
            for o in occ.reshape(-1)
        ]
    ).reshape(*occ.shape, *x.shape)

    # only the derivatives with respect to the energies of the same channel
    n0, n1, no = occ.shape
    got = torch.stack(
        [jac[b, c, :, b, c, :] for b in range(n0) for c in range(n1)]
    ).reshape(n0, n1, no, no)

    ref = torch.tensor(_fermi_reference(case, ktemp), **dd)
    tol = 1e-10 * ref.abs().amax((-1, -2), keepdim=True) + ATOL * THR / kt
    assert ((got - ref).abs() <= tol).all()


@pytest.mark.parametrize("ktemp", TEMPERATURES)
@pytest.mark.parametrize("name", CASES)
def test_high_step_oracle(name: str, ktemp: float):
    """
    The production step count agrees with five steps (exact up to order 31)
    for the derivatives of the individual orbitals up to third order.
    """
    d2 = derivs(name, ktemp, 2, "rand", 3)
    d5 = derivs(name, ktemp, 5, "rand", 3)

    kt = CASES[name].kt(ktemp)
    for n in (1, 2, 3):
        tol = tolerance(d5[n], n, kt).unsqueeze(-1)
        assert ((d2[n] - d5[n]).abs() <= tol).all(), n


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("name", ["fractional", "random", "batch_padding"])
def test_step_count(name: str, steps: int):
    """
    ``k`` steps are exact through order ``2**k - 1`` and violated at order
    ``2**k``. This pins the rule itself (not only the production value).
    """
    _electron_conservation(name, 5000.0, steps)


@pytest.mark.grad
def test_step_count_three():
    """
    Three steps are exact through order 7. The violation at order 8 is
    only 1e-9 of the scale (or less) and not distinguishable from the
    rounding of eighth derivatives, i.e., it is not checked. Takes about a
    minute because the nested forward-mode derivatives grow exponentially.
    """
    _electron_conservation("fractional", 5000.0, 3, lower=False)


def test_start_error_slope():
    """
    With a start error e0 of the detached solution, the error of the first
    derivative of two steps scales as e0**3 (a straight-through step, which
    does not move the value, would give a slope of 1). The start error is
    injected by shifting the Fermi energy of the search.
    """
    emo = torch.tensor([-0.6, -0.3, -0.25, 0.1, 0.4, 0.8], **dd)
    emo = emo.expand(1, 2, -1).clone()
    nab = torch.tensor([[2.5, 2.5]], **dd)
    kt = torch.tensor(0.02, **dd)
    v = torch.randn(emo.shape, generator=torch.Generator().manual_seed(0))
    v = v.to(**dd)

    def first_derivative(steps: int, offset: float) -> torch.Tensor:
        order = order_of(steps)
        with fixed_fermi_energy(offset):
            filling.get_fermi_occupation(nab, emo, kt, diff_order=order)
            fn = lambda x: filling.get_fermi_occupation(
                nab, x, kt, diff_order=order
            )
            return directional_derivs(fn, emo, v, 1)[1]

    ref = first_derivative(6, 0.0)

    offsets = [1e-2, 1e-3, 1e-4]
    err = [(first_derivative(2, e0) - ref).abs().max().item() for e0 in offsets]
    slope = np.polyfit(np.log(offsets), np.log(err), 1)[0]
    assert 2.7 <= slope <= 3.5, (slope, err)

    # one step: first-order error of the order of e0
    err = [(first_derivative(1, e0) - ref).abs().max().item() for e0 in offsets]
    slope = np.polyfit(np.log(offsets), np.log(err), 1)[0]
    assert 0.8 <= slope <= 1.2, (slope, err)


def test_wide_gap():
    """
    Without thermal weight (a gap of about 950 kT, below the floor of the
    steps), the occupations are integers up to their tails and all
    derivatives are negligible (no NaN from the vanishing derivative of the
    number of electrons).
    """
    emo = torch.tensor([-0.9, -0.8, -0.7, -0.6, 0.3, 0.4], **dd)
    emo = emo.expand(1, 2, -1).clone()
    nab = torch.tensor([[4.0, 4.0]], **dd)
    kt = torch.tensor(300.0 * KELVIN2AU, **dd)

    v = torch.randn(emo.shape, generator=torch.Generator().manual_seed(0))
    d = directional_derivs(
        lambda x: filling.get_fermi_occupation(nab, x, kt),
        emo,
        v.to(**dd),
        3,
    )
    ref = (torch.arange(6, **dd) < 4).expand(1, 2, -1) * 1.0
    assert (d[0] - ref).abs().max() <= 1e-150
    for n in (1, 2, 3):
        assert torch.isfinite(d[n]).all()
        assert d[n].abs().max() <= 1e-150


def test_empty_channel():
    """A channel without electrons has neither occupation nor derivative."""
    nab = torch.tensor([2.0, 0.0], **dd)
    emo = torch.tensor([-0.6, -0.3, -0.25, 0.1], **dd)
    emo = emo.expand(2, -1).clone().requires_grad_()
    kt = torch.tensor(5000.0 * KELVIN2AU, **dd)

    occ = filling.get_fermi_occupation(nab, emo, kt)
    assert (occ[1] == 0).all()

    (g0,) = torch.autograd.grad(
        (occ**2).sum(), emo, create_graph=True, retain_graph=True
    )
    assert torch.isfinite(g0).all()
    assert (g0[1] == 0).all()

    (g1,) = torch.autograd.grad(g0.sum(), emo)
    assert torch.isfinite(g1).all()


def test_temperature_derivative():
    """
    The derivative with respect to the temperature (``kt`` may require a
    gradient) matches the implicit function theorem, and its second
    derivative agrees with five steps.
    """
    emo = torch.tensor([-0.6, -0.3, -0.25, 0.1, 0.4, 0.8], **dd)
    emo = emo.expand(1, 2, -1).clone()
    nab = torch.tensor([[2.5, 1.5]], **dd)
    ktemp = 5000.0

    def occ(kt: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation(nab, emo, kt)

    kt = torch.tensor(ktemp * KELVIN2AU, requires_grad=True, **dd)
    f = occ(kt)
    jac = torch.stack(
        [torch.autograd.grad(x, kt, retain_graph=True)[0] for x in f.flatten()]
    ).reshape(f.shape)

    # closed form: dmu/dkT = sum(w * (mu - e) / kT) / sum(w)
    ref = torch.zeros_like(jac)
    for c in range(2):
        e = emo[0, c].cpu().numpy()
        k = kt.item()

        def fermi(mu: float) -> np.ndarray:
            return 1.0 / (1.0 + np.exp((e - mu) / k))

        mu = brentq(
            lambda m: fermi(m).sum() - nab[0, c].item(), -5, 5, xtol=1e-15
        )
        w = fermi(mu) * (1.0 - fermi(mu))
        dmu = (w * (mu - e) / k).sum() / w.sum()
        ref[0, c] = torch.tensor(w * (dmu / k - (mu - e) / k**2), **dd)
    assert torch.allclose(jac, ref, rtol=1e-9, atol=1e-12)

    def second(steps: int) -> torch.Tensor:
        order = order_of(steps)

        def occ_o(x: torch.Tensor) -> torch.Tensor:
            return filling.get_fermi_occupation(nab, emo, x, diff_order=order)

        with fixed_fermi_energy():
            occ_o(kt.detach())
            one = torch.ones((), **dd)
            first = lambda x: jvp(occ_o, (x,), (one,))[1]
            return jvp(first, (kt.detach(),), (one,))[1]

    assert torch.allclose(second(2), second(5), rtol=1e-8, atol=1e-8)


def test_degenerate_third_derivative_finite_differences():
    """
    Third derivatives of the occupations for a threefold degenerate level
    against central differences of the Hessian.

    A plain central difference fails for this case in the elements whose
    analytic value is zero by symmetry (about 1e-10): the truncation error,
    proportional to the fifth derivative (1e-2 at ``h = 1e-5``), dominates
    there. It is not related to the Newton steps (the error is identical for
    2, 3 and 5 steps, quadruples when doubling ``h`` and the invariants hold
    to 1e-15). Richardson extrapolation removes it without any loosened
    tolerance.
    """
    nab = torch.tensor([2.0, 3.0], **dd)
    emo = torch.tensor([-0.6, -0.3, -0.3, -0.3, 0.4, 0.8], **dd)
    emo = emo.expand(2, -1).clone()
    kt = torch.tensor(1000.0 * KELVIN2AU, **dd)

    def fcn(x: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation(nab, x, kt, thr=1e-14)

    def hess(x: torch.Tensor) -> torch.Tensor:
        return jacrev(jacrev(fcn))(x)

    third = jacrev(hess)(emo)[..., 0, 2]

    def central(h: float) -> torch.Tensor:
        step = torch.zeros_like(emo)
        step[0, 2] = h
        return (hess(emo + step) - hess(emo - step)) / (2 * h)

    h = 1e-5
    coarse, fine = central(h), central(h / 2)
    richardson = (4 * fine - coarse) / 3

    scale = third.abs().max()
    assert ((richardson - third).abs() <= 1e-9 * scale).all()

    # the plain difference is dominated by the h**2 truncation error
    assert ((coarse - third).abs().max() / (fine - third).abs().max()) > 3.9


@pytest.mark.parametrize("ktemp", [300.0, 5000.0])
@pytest.mark.parametrize(
    "nel, energies",
    [
        # 10 Eh gap
        ([4.0, 4.0], [-6.0, -5.5, -5.2, -5.0, 5.0, 5.5]),
        # fractional electrons with levels at |x| >> 50 and levels at x ~ 0
        ([2.5, 1.5], [-6.0, -0.3, -0.2999, 0.4, 5.0, 8.0]),
        # threefold degenerate level far from the others
        ([3.0, 2.0], [-9.0, -0.3, -0.3, -0.3, 4.0, 9.0]),
    ],
)
def test_extreme_spectrum_finite(
    nel: list[float], energies: list[float], ktemp: float
):
    """Occupations and their derivatives up to third order are finite."""
    nab = torch.tensor(nel, **dd)
    emo = torch.tensor(energies, **dd).expand(2, -1).clone().requires_grad_()
    kt = torch.tensor(ktemp * KELVIN2AU, **dd)

    occ = filling.get_fermi_occupation(nab, emo, kt)
    assert torch.isfinite(occ).all()
    assert ((occ >= 0) & (occ <= 1)).all()

    # arbitrary weights of the orbitals
    weights = torch.linspace(0.3, 1.7, occ.numel(), **dd).reshape(occ.shape)
    out = (occ * weights * emo).sum()
    for _ in range(3):
        (grad,) = torch.autograd.grad(out, emo, create_graph=True)
        assert torch.isfinite(grad).all()
        out = (grad * weights).sum()


@pytest.mark.parametrize(
    "order, steps",
    [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (7, 3), (8, 4), (15, 4), (31, 5)],
)
def test_diff_steps(order: int, steps: int):
    """``k`` steps for order ``n``: ``k = ceil(log2(n + 1))``."""
    assert filling._diff_steps(order) == steps
    # exact through 2**k - 1, and one step fewer would not be enough
    assert 2**steps - 1 >= order
    assert steps == 0 or 2 ** (steps - 1) - 1 < order


def test_diff_order_numpy_integer():
    assert filling._diff_steps(np.int64(5)) == 3


def test_diff_order_single_precision_warning():
    """Derivatives beyond the default order need double precision."""
    nab = torch.tensor([2.0, 1.0])
    emo = torch.tensor([-0.6, -0.3, -0.25, 0.1]).expand(2, -1)
    kt = torch.tensor(0.01)

    with pytest.warns(UserWarning, match="double precision"):
        filling.get_fermi_occupation(nab, emo, kt, diff_order=4)


@pytest.mark.parametrize("order", [-1, 1.5, "2", True])
def test_diff_order_invalid(order):
    nab = torch.tensor([2.0, 1.0], **dd)
    emo = torch.tensor([-0.6, -0.3, -0.25, 0.1], **dd).expand(2, -1)
    kt = torch.tensor(0.01, **dd)

    error = ValueError if isinstance(order, int) and order < 0 else TypeError
    with pytest.raises(error):
        filling.get_fermi_occupation(nab, emo, kt, diff_order=order)


def test_diff_order_default():
    """The default is order 3 (two steps), and the values do not change."""
    assert defaults.FERMI_DIFF_ORDER == 3
    assert filling._diff_steps(defaults.FERMI_DIFF_ORDER) == 2

    case = CASES["fractional"]
    kt = case.kt(5000.0)
    ref = case.occupation(case.emo, kt).detach()
    for order in (0, 1, 3, 7, 15):
        occ = case.occupation(case.emo, kt, order).detach()
        # the steps move the value by (much) less than the threshold
        assert (occ - ref).abs().max() <= 1e-10


@pytest.mark.parametrize("order", [2, 4, 5, 6])
@pytest.mark.parametrize("name", ["fractional", "batch_padding"])
def test_diff_order_not_power_of_two(name: str, order: int):
    """
    Orders between ``2**(k-1)`` and ``2**k - 1`` use ``k`` steps and are exact
    through the requested order (the electrons are conserved).
    """
    case = CASES[name]
    kt = case.kt(5000.0)
    v = case.direction(7)
    d = _run_order(case, 5000.0, order, v, order)
    ref = _run_order(case, 5000.0, 15, v, order)

    for n in range(1, order + 1):
        total = d[n].sum(-1).abs()
        assert (total <= tolerance(d[n], n, kt)).all(), (n, total)

        # each orbital agrees with four steps (exact through order 15)
        tol = tolerance(ref[n], n, kt).unsqueeze(-1)
        assert ((d[n] - ref[n]).abs() <= tol).all(), n


###############################################################################
# derivatives with respect to the number of electrons
###############################################################################


def _electron_derivs(case: Case, ktemp: float, steps: int, nmax: int):
    """Derivatives of the occupations along a direction of `nel`."""
    kt = case.kt(ktemp)
    g = torch.Generator().manual_seed(11)
    v = torch.rand(case.nab.shape, generator=g, dtype=torch.double) + 0.5
    v = v.to(DEVICE)

    def fcn(n: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation(
            n, case.emo, kt, mask=case.mask, diff_order=order_of(steps)
        )

    with fixed_fermi_energy():
        fcn(case.nab)  # converged Fermi energy at t = 0
        return v, directional_derivs(fcn, case.nab, v, nmax)


def _natural_scale(v: torch.Tensor, weight: torch.Tensor, n: int):
    """
    Scale of the n-th derivative along `v` of the electrons of a channel
    with the thermal weight `weight`: each derivative brings a factor of
    ``d mu / d N ~ kT / weight``, and each derivative of the Fermi function
    one of ``1 / kT``.
    """
    return v.abs() * (v.abs() / weight) ** (n - 1)


def _above_floor(case: Case, ktemp: float, steps: int) -> torch.Tensor:
    """
    Channels whose derivative of the number of electrons is above the floor
    of the differentiable Newton steps (see `_diff_floor`), with a factor of
    10 as a margin for the difference between the weights.
    """
    _, w = exact_occupation(case, ktemp)
    deriv = w.sum(-1) / case.kt(ktemp)
    return deriv >= 10 * filling._diff_floor(torch.double, steps)


@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("ktemp", TEMPERATURES)
@pytest.mark.parametrize("name", ["fractional", "degenerate", "batch_padding"])
def test_electron_number_derivatives(name: str, ktemp: float, steps: int):
    """
    The occupations follow the number of electrons: the first derivative of
    their sum is the direction itself, all higher ones vanish. ``k`` steps
    are exact through order ``2**k - 1`` (the proof covers any parameter of
    the Newton map) and violate the identity at order ``2**k``. This also
    holds in gaps (no thermal weight within the threshold), down to the floor
    of the steps. Below it, the derivatives are finite.
    """
    order = 2**steps
    case = CASES[name]
    v, d = _electron_derivs(case, ktemp, steps, order)
    _, w = exact_occupation(case, ktemp)
    weight = w.sum(-1)
    exact = _above_floor(case, ktemp, steps)

    total = d[1].sum(-1)
    tol = RTOL * torch.maximum(scale_of(d[1]), v.abs())
    assert ((total - v).abs() <= tol)[exact].all(), (total, v)

    for n in range(2, order):
        total = d[n].sum(-1).abs()
        nat = _natural_scale(v, weight, n)
        tol = RTOL * torch.maximum(scale_of(d[n]), nat)
        assert (total <= tol)[exact].all(), (n, total)

    for n in range(1, order + 1):
        assert torch.isfinite(d[n]).all(), n

    # a single orbital at f = 1/2 has vanishing even derivatives (no scale)
    scale = scale_of(d[order])
    measurable = exact & (weight >= MIN_WEIGHT) & (scale > 0.0)
    measurable &= curvature(case, ktemp) >= MIN_CURVATURE
    if case.generic and measurable.any():
        ratio = d[order].sum(-1).abs() / torch.where(measurable, scale, 1.0)
        assert ratio[measurable].max() >= LOWER, ratio


@pytest.mark.parametrize("ktemp", TEMPERATURES)
@pytest.mark.parametrize("name", ["fractional", "degenerate", "batch_padding"])
def test_electron_number_first_order(name: str, ktemp: float):
    """
    First derivative in closed form: an additional electron is distributed
    according to the thermal weights, ``d f_i / d N = w_i / sum_j w_j`` with
    ``w = f (1 - f)``, and it does not change the other channels. In a gap,
    the weights are tails, and the ratio depends on the position of the
    Fermi energy, which the threshold of the search does not determine.
    """
    case = CASES[name]
    kt = case.kt(ktemp)
    exact = _above_floor(case, ktemp, filling._diff_steps(3))

    def fcn(n: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation(n, case.emo, kt, mask=case.mask)

    jac = jacrev(fcn)(case.nab)  # [b, 2, n, b, 2]
    assert torch.isfinite(jac).all()

    _, w = exact_occupation(case, ktemp)
    ref = w / w.sum(-1, keepdim=True)
    nb, nc = case.nab.shape
    for b in range(nb):
        for c in range(nc):
            if not exact[b, c]:
                continue
            expected = torch.zeros_like(ref)
            expected[b, c] = ref[b, c]
            assert (jac[..., b, c] - expected).abs().max() <= 1e-10, (b, c)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("gap", [60.0, 150.0, 300.0])
def test_electron_number_gap(dtype: torch.dtype, gap: float):
    """
    Integer electrons in a gap of `gap` kT: the thermal weight is far below
    the threshold (1e-12 to 1e-64), but an additional electron still goes to
    the HOMO and LUMO in the exact ratio. Beyond the floor of the steps
    (here: all gaps in single precision), the derivative is zero, and all
    derivatives are finite.
    """
    kt = 1e-3
    half = gap * kt / 2
    energies = [-half - 0.3, -half - 0.05, -half, half, half + 0.02, half + 0.4]
    case = Case([[3.0, 3.0]], torch.tensor(energies).expand(1, 2, -1))

    emo = case.emo.to(dtype)
    ktt = torch.tensor(kt, dtype=dtype, device=DEVICE)

    def fcn(n: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation(n, emo, ktt)

    jac = jacrev(fcn)(case.nab.to(dtype))[0, 0, :, 0, 0]  # alpha by alpha
    assert torch.isfinite(jac).all()

    _, w = exact_occupation(case, kt / KELVIN2AU)
    ref = (w / w.sum(-1, keepdim=True))[0, 0]
    floor = filling._diff_floor(dtype, filling._diff_steps(3))
    if w[0, 0].sum() / kt >= 10 * floor:
        tol = 1e-12 if dtype == torch.double else 1e-5
        assert (jac.double() - ref).abs().max() <= tol
    else:
        assert (jac == 0).all()

    # higher orders along the electrons are finite, too
    v = torch.ones_like(case.nab).to(dtype)
    one = torch.ones((), dtype=dtype, device=DEVICE)

    def nth(n: int):
        def f(t: torch.Tensor) -> torch.Tensor:
            if n == 0:
                return fcn(case.nab.to(dtype) + t * v)
            return jvp(nth(n - 1), (t,), (one,))[1]

        return f

    for n in range(1, 5):
        assert torch.isfinite(
            nth(n)(torch.zeros((), **{**dd, "dtype": dtype}))
        ).all()


def test_electron_number_graph():
    """
    The graph of `nel` is kept (it was detached before): the chemical
    potential and Fukui functions are available by autograd. Channels without
    electrons or completely filled channels do not depend on it.
    """
    emo = torch.tensor([-0.6, -0.3, -0.25, 0.1], **dd).expand(3, 2, -1)
    nab = torch.tensor([[2.5, 1.5], [4.0, 1.0], [2.0, 0.0]], **dd)
    nab.requires_grad_()
    kt = torch.tensor(5000.0 * KELVIN2AU, **dd)

    occ = filling.get_fermi_occupation(nab, emo, kt)
    assert occ.requires_grad

    (grad,) = torch.autograd.grad(occ.sum(), nab)
    ref = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], **dd)
    assert (grad - ref).abs().max() <= 1e-12


@pytest.mark.parametrize("padding", [False, True])
def test_electron_number_zero_temperature(padding: bool):
    """
    At zero temperature, only the partially occupied orbital changes. For
    integer electrons, the derivative belongs to the lowest empty orbital
    (adding electrons), i.e., the derivatives sum to one in both cases.
    """
    emo = torch.tensor([-0.6, -0.3, 0.0, -0.25, 0.1], **dd).expand(2, -1)
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0], **dd).expand(2, -1)
    nab = torch.tensor([2.5, 1.0], **dd)
    kt = torch.tensor(0.0, **dd)

    def fcn(n: torch.Tensor) -> torch.Tensor:
        m = mask if padding else None
        return filling.get_fermi_occupation(n, emo, kt, mask=m)

    jac = jacrev(fcn)(nab)  # [2, n, 2]
    # the padded orbital (index 2) is skipped with a mask
    alpha, beta = (3, 1) if padding else (2, 1)
    ref = torch.zeros_like(jac)
    ref[0, alpha, 0] = 1.0
    ref[1, beta, 1] = 1.0
    assert torch.equal(jac, ref)
