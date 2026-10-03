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
Orientation-aware shell-pair builder
====================================

Enumerates shell pairs for a single (unbatched) molecule and assembles a full
integral matrix (overlap, dipole, quadrupole) through the
shared ``compute_1d``-contract pipeline, with an out-of-place ``index_copy``
scatter of the lower triangle followed by a symmetrization (no in-place
writes into the result).

Pairs are grouped by the *ordered* unique-shell pair ``(ushell_bra,
ushell_ket)``. Merging ``(a, b)`` with ``(b, a)`` would be harmless for the
overlap, which depends only on the interatomic distance, but would transpose
the shells of odd-parity multipole blocks (a sign flip for the dipole).

Every shell pair with ``bra >= ket`` (shell index) is computed once and
mirrored; this relies on the multipole integrals being symmetric, which holds
for a common gauge origin. Same-atom pairs are *included* in the integrals
(they are load-bearing for GFN2's AES2 term), and every shell is normalized
exactly, so the overlap has a unit diagonal without forcing it.
"""

from __future__ import annotations

from math import comb, gamma, pi
from typing import Sequence

import numpy as np
import torch

from dxtb import IndexHelper
from dxtb._src.typing import Tensor

from .pipeline import (
    OVERLAP_COMPONENTS,
    Kernel1D,
    assemble_multipole_1d,
    assemble_overlap_1d,
)
from .trafo import TRAFO

__all__ = [
    "assemble_matrix",
    "prepare",
    "select_pairs",
]

_SCREEN_CHUNK = 20000


def _to_ints(t: Tensor) -> list[int]:
    """
    Integer tensor to a list of Python ints.

    Inside a ``torch.func`` transform, ``Tensor.tolist()`` raises ``Cannot
    access data pointer of Tensor that doesn't have storage`` on some torch
    versions (observed on 2.4), while extracting single elements works. The
    fast path is tried first, so the element-wise fallback only costs time
    within a transform.
    """
    try:
        return t.tolist()
    except RuntimeError:
        return [int(v) for v in t]


class _Class:
    """Structural data of one ordered unique-shell-pair class."""

    def __init__(self, ub, uk, lb, lk, ib, jk, idx, weight):
        self.ub, self.uk, self.lb, self.lk = ub, uk, lb, lk
        self.ib, self.jk, self.idx, self.weight = ib, jk, idx, weight


class _Plan:
    """
    Everything of :func:`assemble_matrix` that depends only on the structure
    (angular momenta, unique shells, shell-to-atom map), not on positions or
    basis parameters.
    """

    def __init__(self, ihelp: IndexHelper, dev: torch.device) -> None:
        # All index bookkeeping is done with numpy on the helper's constant
        # index arrays. This keeps it out of the autograd/`torch.func`
        # machinery, which cannot evaluate `unique`/`nonzero` on tensors
        # inside a transform on every supported torch version (see
        # `_to_ints`).
        ang_np = np.asarray(_to_ints(ihelp.angular))
        ush_np = np.asarray(_to_ints(ihelp.shells_to_ushell))
        nsh = ang_np.shape[0]
        nsph_np = 2 * ang_np + 1
        off_np = np.cumsum(nsph_np) - nsph_np

        self.nao = int(nsph_np.sum())
        self.atom = ihelp.shells_to_atom.to(dev)
        self.unique_angular = _to_ints(ihelp.unique_angular)

        # ordered shell pairs with bra >= ket
        bra_np, ket_np = np.tril_indices(nsh)
        key_np = ush_np[bra_np] * (int(ush_np.max()) + 1) + ush_np[ket_np]

        self.classes: list[_Class] = []
        for k in np.unique(key_np).tolist():
            cls_np = np.nonzero(key_np == k)[0]
            first = int(cls_np[0])
            ib_np, jk_np = bra_np[cls_np], ket_np[cls_np]
            lb, lk = int(ang_np[bra_np[first]]), int(ang_np[ket_np[first]])

            rows = (
                off_np[ib_np][:, None, None]
                + np.arange(2 * lb + 1)[None, :, None]
            )
            cols = (
                off_np[jk_np][:, None, None]
                + np.arange(2 * lk + 1)[None, None, :]
            )
            idx = rows * self.nao + cols  # broadcasts to (npairs, nb, nk)

            self.classes.append(
                _Class(
                    int(ush_np[bra_np[first]]),
                    int(ush_np[ket_np[first]]),
                    lb,
                    lk,
                    torch.as_tensor(ib_np, dtype=torch.long, device=dev),
                    torch.as_tensor(jk_np, dtype=torch.long, device=dev),
                    torch.as_tensor(idx, dtype=torch.long, device=dev),
                    torch.as_tensor(
                        np.where(ib_np == jk_np, 0.5, 1.0), device=dev
                    ),
                )
            )


_PLANS: dict[tuple, _Plan] = {}
_PLANS_BY_OBJECT: dict[tuple, tuple[IndexHelper, _Plan]] = {}
"""
Plans by identity of the index helper (the helper itself is kept alive, so
its ``id`` cannot be reused). The lookup does not read any tensor data and can
therefore be traced by ``torch.compile``.
"""


def _plan(ihelp: IndexHelper, dev: torch.device) -> _Plan:
    """
    Structural plan of `ihelp`, cached on the bytes of its index arrays.

    Inside ``torch.compile`` the plan cannot be built (it reads the index
    arrays on the host): it must exist already, i.e., the function has to be
    called once eagerly (or :func:`prepare` called) with the same helper.
    """
    obj = _PLANS_BY_OBJECT.get((id(ihelp), dev))
    if obj is not None and obj[0] is ihelp:
        return obj[1]

    if torch.compiler.is_compiling():
        raise RuntimeError(
            "The pair plan of this IndexHelper has not been built. Call "
            "`prepare(ihelp, device)` (or the function eagerly) before "
            "compiling."
        )

    key = (
        dev,
        tuple(_to_ints(ihelp.angular)),
        tuple(_to_ints(ihelp.shells_to_ushell)),
        tuple(_to_ints(ihelp.shells_to_atom)),
    )
    plan = _PLANS.get(key)
    if plan is None:
        if len(_PLANS) >= 32:
            _PLANS.clear()
        plan = _PLANS[key] = _Plan(ihelp, dev)

    if len(_PLANS_BY_OBJECT) >= 32:
        _PLANS_BY_OBJECT.clear()
    _PLANS_BY_OBJECT[(id(ihelp), dev)] = (ihelp, plan)
    return plan


def prepare(ihelp: IndexHelper, dev: torch.device) -> None:
    """
    Build the structural plan of `ihelp` on `dev`, so that
    :func:`assemble_matrix` can be traced by ``torch.compile`` (``fullgraph``).
    """
    _plan(ihelp, dev)


def _normalized(
    kernel: Kernel1D, plan: _Plan, alphas: list[Tensor], coeffs: list[Tensor]
) -> list[Tensor]:
    """
    Contraction coefficients scaled to exactly unit self-overlap of every
    shell, computed with the kernel that assembles the integrals.

    ``Basis.create_cgtos`` already normalizes every contracted shell, so for
    the built-in bases this changes the coefficients only at the level of
    rounding errors. It keeps the integrals normalized when the exponents or
    coefficients are modified after ``create_cgtos``, e.g., while fitting
    basis parameters. The norm is uniform over the ``2l+1`` components of a
    shell, so it is folded into the coefficients (bra and ket each pick up
    one factor) instead of scaling the AO matrix afterwards.
    """
    out = []
    for l, a, c in zip(plan.unique_angular, alphas, coeffs):
        vec0 = torch.zeros((1, 3), dtype=a.dtype, device=a.device)
        s = assemble_overlap_1d(kernel, (l, l), (a, a), (c, c), vec0)
        # reciprocal square root with a clamp and an eps shift; the eps is a
        # Python float, which `torch.compile` can trace
        eps = torch.finfo(a.dtype).eps
        out.append(c / (torch.sqrt(torch.clamp(s[0, 0, 0], min=eps)) + eps))
    return out


def _mirror(
    idx: list[Tensor],
    val: list[Tensor],
    ncomp: int,
    nao: int,
    like: Tensor,
) -> Tensor:
    """
    Scatter the lower-triangular shell blocks into an ``(ncomp, nao, nao)``
    matrix (dtype and device of ``like``) and add its transpose.
    """
    flat = torch.zeros(ncomp, nao * nao, dtype=like.dtype, device=like.device)
    if idx:
        # A single out-of-place scatter; every AO pair occurs at most once,
        # so no accumulation is needed and the backward is a gather.
        flat = flat.index_copy(1, torch.cat(idx), torch.cat(val, dim=1))

    tri = flat.reshape(ncomp, nao, nao)
    return tri + tri.transpose(-1, -2)


def assemble_matrix(
    kernel: Kernel1D,
    ihelp: IndexHelper,
    alphas: list[Tensor],
    coeffs: list[Tensor],
    positions: Tensor,
    components: tuple[tuple[int, int, int], ...] | None = None,
    origin: Tensor | None = None,
    screening_threshold: float | None = None,
    chunk_size: int | None = None,
    checkpoint: bool = False,
    pairs: Sequence[Tensor] | None = None,
) -> Tensor:
    """
    Assemble the full AO integral matrix of one molecule. Every shell is
    normalized to exactly unit self-overlap.

    Parameters
    ----------
    kernel : Kernel1D
        ``compute_1d``-contract kernel (OS supports ``emax > 0``).
    ihelp : IndexHelper
        Index helper of the (unbatched) molecule.
    alphas, coeffs : list[Tensor]
        Per-unique-shell exponents and coefficients (``Basis.create_cgtos``).
    positions : Tensor
        Cartesian coordinates, shape ``(nat, 3)``.
    components : tuple of (int, int, int) | None
        Per-axis multipole orders of each output component. ``None`` gives
        the overlap (single component, no ``sij``-free shortcut).
    origin : Tensor | None
        Multipole origin (default: Cartesian origin).
    screening_threshold : float | None
        Drop shell pairs on different atoms whose rigorous upper bound on the
        largest element of the block (:func:`_bounds`) is below this value for
        the overlap, dipole and quadrupole alike (:func:`_screen`), i.e., every
        dropped element is smaller than the threshold in absolute value. Pairs on the same atom are always kept. The mask is computed
        from detached positions, so autograd is unaffected (dropped blocks get
        no gradient). ``None`` (default) computes every pair. The mask has a
        data-dependent size: it works eagerly and under ``jacrev``/``jacfwd``,
        but not under ``vmap`` or ``torch.compile``; use ``pairs`` there.
    chunk_size : int | None
        Maximum number of shell pairs of one class evaluated at once, to
        bound the size of the intermediate tensors. ``None``: no chunking.
    checkpoint : bool
        Recompute the intermediate tensors of every chunk in the backward pass
        (``torch.utils.checkpoint``) instead of storing them: less memory for
        about one extra forward evaluation. Only affects autograd.
    pairs : Sequence[Tensor] | None
        Shell pairs to compute, one index tensor per pair class, as returned
        by :func:`select_pairs` (excludes ``screening_threshold``). The
        indices are constants, so everything is static and the function works
        under ``vmap`` and ``torch.compile`` as well. The error bound of the
        screening only holds for the geometries the selection was made for.

    Returns
    -------
    Tensor
        Shape ``(ncomp, nao, nao)``; ``ncomp = 1`` for the overlap.
    """
    dev, dt = positions.device, positions.dtype

    plan = _plan(ihelp, dev)
    nao, atom = plan.nao, plan.atom
    if components is None:
        components = OVERLAP_COMPONENTS
    ncomp = len(components)

    coeffs = _normalized(kernel, plan, alphas, coeffs)

    if pairs is not None:
        if screening_threshold is not None:
            raise ValueError(
                "Pass either `screening_threshold` or `pairs`, not both."
            )
        if len(pairs) != len(plan.classes):
            raise ValueError(
                f"Expected {len(plan.classes)} pair selections (one per "
                f"class), got {len(pairs)}."
            )
    elif screening_threshold is not None and torch.compiler.is_compiling():
        raise RuntimeError(
            "`screening_threshold` selects pairs depending on the data, which "
            "`torch.compile` cannot trace. Select the pairs eagerly with "
            "`select_pairs` and pass them as `pairs`."
        )

    all_idx: list[Tensor] = []
    all_val: list[Tensor] = []

    for k, cl in enumerate(plan.classes):
        ub, uk, lb, lk = cl.ub, cl.uk, cl.lb, cl.lk
        keep: Tensor | None = None

        if pairs is not None:
            keep = pairs[k]
        elif screening_threshold is not None:
            keep = _screen(
                torch.arange(cl.ib.numel(), device=dev),
                cl.ib,
                cl.jk,
                atom,
                positions.detach(),
                alphas[ub].detach(),
                alphas[uk].detach(),
                coeffs[ub].detach(),
                coeffs[uk].detach(),
                (lb, lk),
                screening_threshold,
                None if origin is None else origin.detach(),
            )

        selections: list[Tensor | slice]
        if keep is None and chunk_size is None:
            # all pairs in one go: views instead of identity gathers
            selections = [slice(None)]
        else:
            if keep is None:
                keep = torch.arange(cl.ib.numel(), device=dev)
            step = keep.numel() if chunk_size is None else int(chunk_size)
            step = max(step, 1)
            selections = [
                keep[start : start + step]
                for start in range(0, keep.numel(), step)
            ]

        for sel in selections:
            ib, jk = cl.ib[sel], cl.jk[sel]

            pos_a = positions[atom[ib]]
            vec = positions[atom[jk]] - pos_a  # B - A

            args = (
                vec,
                pos_a,
                alphas[ub],
                alphas[uk],
                coeffs[ub],
                coeffs[uk],
            )

            def _blocks(vec, pos_a, a_b, a_k, c_b, c_k, lb=lb, lk=lk):
                return assemble_multipole_1d(
                    kernel,
                    (lb, lk),
                    (a_b, a_k),
                    (c_b, c_k),
                    vec,
                    pos_a,
                    components,
                    origin,
                )

            if checkpoint and torch.is_grad_enabled():
                # pylint: disable=import-outside-toplevel
                from torch.utils.checkpoint import checkpoint as _ckpt

                blocks = _ckpt(_blocks, *args, use_reentrant=False)
            else:
                blocks = _blocks(*args)

            # same-shell blocks are complete: halve them for the symmetrization
            blocks = blocks * cl.weight[sel].to(dt)[:, None, None, None]

            all_idx.append(cl.idx[sel].reshape(-1))
            all_val.append(blocks.permute(1, 0, 2, 3).reshape(ncomp, -1))

    return _mirror(all_idx, all_val, ncomp, nao, positions)


_SCREEN_ORDERS = (0, 1, 2)
"""
Multipole orders (overlap, dipole, quadrupole) that share one screening mask.
"""


_SPH_NORM = tuple(t.abs().sum(-1).max().item() for t in TRAFO)
"""Largest absolute row sum of the Cartesian-to-spherical transform of each l."""


def _bounds(
    pa: Tensor,
    pb: Tensor,
    alpha_a: Tensor,
    alpha_b: Tensor,
    coeff_a: Tensor,
    coeff_b: Tensor,
    angular: tuple[int, int],
    orders: Sequence[int],
    origin: Tensor | None,
) -> Tensor:
    r"""
    Rigorous upper bounds on the largest absolute element of the spherical
    integral block of each shell pair of one class (centers ``pa`` and ``pb``
    of shape ``(npairs, 3)``), for every multipole order ``e`` in ``orders``,
    as a tensor of shape ``(len(orders), npairs)``. The quantities that do not
    depend on the order are computed only once.

    For a Cartesian monomial of the pair with total degrees ``(la, lb, e)``
    about the centers :math:`A`, :math:`B` and the multipole origin :math:`C`,
    :math:`|\prod (x-X)^n| \le |r-A|^{l_a} |r-B|^{l_b} |r-C|^e`, and
    :math:`|r-X| \le \rho + |P-X|` with :math:`\rho = |r-P|` of the product
    Gaussian center :math:`P`. With
    :math:`\int \rho^k e^{-p\rho^2} d^3r = 2\pi\Gamma(\frac{k+3}{2})
    p^{-(k+3)/2}`, the primitive pair is bounded by

    .. math::

        |c_i c_j| e^{-\mu R^2} \sum_{a,b,c} \binom{l_a}{a}
        \binom{l_b}{b} \binom{e}{c} |PA|^a |PB|^b |PC|^c
        \, 2\pi \Gamma(\tfrac{n+3}{2}) p^{-(n+3)/2},

    with :math:`n = (l_a - a) + (l_b - b) + (e - c)`, :math:`|PA| = bR/p`,
    :math:`|PB| = aR/p` and :math:`|PC| \le \max(|AC|, |BC|)`. The sum over
    all primitive pairs is multiplied by the largest absolute row sums of the
    two spherical transforms. With ``e = 0`` this is the overlap.
    """
    la, lb = angular
    a, b = alpha_a[:, None], alpha_b[None, :]
    p = a + b
    cc = (coeff_a[:, None] * coeff_b[None, :]).abs()
    mu = a * b / p
    sph = _SPH_NORM[la] * _SPH_NORM[lb] * (1.0 + 1e-10)  # rounding margin

    c = torch.zeros(3, dtype=pa.dtype, device=pa.device)
    if origin is not None:
        c = origin.to(pa.dtype).reshape(-1, 3)[0]

    r = (pb - pa).norm(dim=-1)[:, None, None]
    xc = torch.maximum((pa - c).norm(dim=-1), (pb - c).norm(dim=-1))
    xc = xc[:, None, None]
    xa, xb = b / p * r, a / p * r

    # everything but the polynomial is independent of the order
    prefactor = cc * torch.exp(-mu * r * r)
    emax = max(orders)
    mom = [
        2.0 * pi * gamma((n + 3) / 2) * p ** (-(n + 3) / 2)
        for n in range(la + lb + emax + 1)
    ]
    xa_pow = [xa**i for i in range(la + 1)]
    xb_pow = [xb**j for j in range(lb + 1)]
    xc_pow = [xc**k for k in range(emax + 1)]

    bounds = []
    for e in orders:
        poly = 0.0
        for i in range(la + 1):
            for j in range(lb + 1):
                for k in range(e + 1):
                    n = (la - i) + (lb - j) + (e - k)
                    poly = poly + (
                        comb(la, i)
                        * comb(lb, j)
                        * comb(e, k)
                        * xa_pow[i]
                        * xb_pow[j]
                        * xc_pow[k]
                        * mom[n]
                    )
        bounds.append((prefactor * poly).sum((-2, -1)) * sph)

    return torch.stack(bounds)


def _screen(
    cls: Tensor,
    bra: Tensor,
    ket: Tensor,
    atom: Tensor,
    positions: Tensor,
    alpha_a: Tensor,
    alpha_b: Tensor,
    coeff_a: Tensor,
    coeff_b: Tensor,
    angular: tuple[int, int],
    threshold: float,
    origin: Tensor | None = None,
) -> Tensor:
    """
    Indices (into ``bra``/``ket``) of the pairs of one class to keep: those
    on the same atom and those for which the :func:`_bounds` of *any* multipole
    order in :data:`_SCREEN_ORDERS` exceeds ``threshold``.

    The mask is the same for the overlap, dipole and quadrupole. Dropping a
    pair from the overlap but not from the dipole would leave an error of up
    to ``|R| * threshold`` (``|R|^2`` for the quadrupole) once the multipoles
    are shifted from the gauge origin to the atoms, which grows with the
    distance of the molecule from the origin.
    """
    kept = []
    for start in range(0, cls.numel(), _SCREEN_CHUNK):
        sel = cls[start : start + _SCREEN_CHUNK]
        pa, pb = positions[atom[bra[sel]]], positions[atom[ket[sel]]]
        keep = (pb == pa).all(-1)
        bound = _bounds(
            pa,
            pb,
            alpha_a,
            alpha_b,
            coeff_a,
            coeff_b,
            angular,
            _SCREEN_ORDERS,
            origin,
        )
        keep = keep | (bound > threshold).any(0)
        kept.append(sel[keep])

    return torch.cat(kept) if kept else cls[:0]


def select_pairs(
    kernel: Kernel1D,
    ihelp: IndexHelper,
    alphas: list[Tensor],
    coeffs: list[Tensor],
    positions: Tensor,
    threshold: float,
    origin: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """
    Shell pairs to compute for the given geometries, for the ``pairs``
    argument of :func:`assemble_matrix`: one index tensor per pair class.

    The selection is made eagerly (the number of pairs depends on the data)
    with the shared bound of :func:`_screen`, so it is valid for the overlap,
    dipole and quadrupole. It is the union over all geometries in
    ``positions``, which keeps the error bound for every one of them (a pair
    that is kept although it is not needed is exact).

    Parameters
    ----------
    kernel : Kernel1D
        ``compute_1d``-contract kernel (normalizes the shells).
    ihelp : IndexHelper
        Index helper of the (unbatched) molecule.
    alphas, coeffs : list[Tensor]
        Per-unique-shell exponents and coefficients (``Basis.create_cgtos``).
    positions : Tensor
        Cartesian coordinates, shape ``(nat, 3)`` or ``(nbatch, nat, 3)``.
    threshold : float
        Drop pairs whose bound is below this value.
    origin : Tensor | None
        Multipole origin (default: Cartesian origin).

    Returns
    -------
    tuple[Tensor, ...]
        Indices of the kept pairs of each class.
    """
    dev = positions.device
    plan = _plan(ihelp, dev)
    coeffs = [c.detach() for c in _normalized(kernel, plan, alphas, coeffs)]
    geometries = positions.detach().reshape(-1, *positions.shape[-2:])
    org = None if origin is None else origin.detach()

    out = []
    for cl in plan.classes:
        every = torch.arange(cl.ib.numel(), device=dev)
        kept = [
            _screen(
                every,
                cl.ib,
                cl.jk,
                plan.atom,
                pos,
                alphas[cl.ub].detach(),
                alphas[cl.uk].detach(),
                coeffs[cl.ub],
                coeffs[cl.uk],
                (cl.lb, cl.lk),
                threshold,
                org,
            )
            for pos in geometries
        ]
        out.append(torch.unique(torch.cat(kept)))
    return tuple(out)
