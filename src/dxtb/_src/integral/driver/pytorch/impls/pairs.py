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
integral matrix through the shared ``compute_1d``-contract pipeline, with an
out-of-place ``index_copy`` scatter of the lower triangle followed by a
symmetrization (no in-place writes into the result).

Pairs are grouped by the *ordered* unique-shell pair ``(ushell_bra,
ushell_ket)``. The pre-existing ``BaseBasis.unique_shell_pairs`` groups by an
unordered prime product, which merges ``(a, b)`` and ``(b, a)``. That is
harmless for the overlap (it depends only on the interatomic distance) but
silently transposes the shells of odd-parity multipole blocks (a sign flip
for the dipole), which is why this builder does not reuse it.

Every shell pair with ``bra >= ket`` (shell index) is computed once and
mirrored; this relies on the multipole integrals being symmetric, which holds
for a common gauge origin. Same-atom
pairs are *included* (they are load-bearing for GFN2's AES2 term); the
legacy overlap's same-atom exclusion and forced unit diagonal are not
applied here.
"""

from __future__ import annotations

from math import pi
from typing import Callable

import numpy as np
import torch
from tad_mctc.batch import pack

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.param import Param
from dxtb._src.typing import Tensor

from .normalize import shell_self_overlap_norm
from .pipeline import Kernel1D, assemble_multipole_1d, assemble_overlap_1d

__all__ = ["assemble_matrix", "assemble_matrix_batch"]

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


def _plan(ihelp: IndexHelper, dev: torch.device) -> _Plan:
    """Structural plan of `ihelp`, cached on the bytes of its index arrays."""
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
    return plan


def assemble_matrix(
    kernel: Kernel1D,
    ihelp: IndexHelper,
    alphas: list[Tensor],
    coeffs: list[Tensor],
    positions: Tensor,
    components: tuple[tuple[int, int, int], ...] | None = None,
    origin: Tensor | None = None,
    normalize: bool = True,
    screening_threshold: float | None = None,
    chunk_size: int | None = None,
    block_fn: Callable[..., Tensor] | None = None,
    checkpoint: bool = False,
) -> Tensor:
    """
    Assemble the full AO integral matrix of one molecule.

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
    normalize : bool
        Renormalize every shell to exactly unit self-overlap
        (:func:`shell_self_overlap_norm`), matching libcint/PySCF.
    screening_threshold : float | None
        Drop shell pairs whose prefactor bound
        ``max |c_a c_b| (pi/p)^1.5 exp(-mu R^2) (1 + R)^(la + lb + emax)`` is
        below this value (same-atom pairs are always kept). The mask is
        computed from detached positions, so autograd is unaffected. ``None``
        (default) computes every pair.
    chunk_size : int | None
        Maximum number of shell pairs of one class evaluated at once, to
        bound the size of the intermediate tensors. ``None``: no chunking.
    block_fn : Callable | None
        Replacement for :func:`assemble_multipole_1d` (same signature), e.g.
        its ``torch.compile``-d version. Also used for the overlap, which is
        then computed as the ``emax = 0`` multipole.
    checkpoint : bool
        Recompute the intermediate tensors of every chunk in the backward pass
        (``torch.utils.checkpoint``) instead of storing them: less memory for
        about one extra forward evaluation. Only affects autograd.

    Returns
    -------
    Tensor
        Shape ``(ncomp, nao, nao)``; ``ncomp = 1`` for the overlap.
    """
    dev, dt = positions.device, positions.dtype

    plan = _plan(ihelp, dev)
    nao, atom = plan.nao, plan.atom
    ncomp = 1 if components is None else len(components)
    emax = 0 if components is None else max(max(c) for c in components)

    if normalize:
        # The norm is uniform over the 2l+1 components of a shell, so it is
        # folded into the contraction coefficients (bra and ket each pick up
        # one factor), instead of scaling the nao x nao matrix afterwards.
        coeffs = [
            c * shell_self_overlap_norm(kernel, int(l), a, c)[0]
            for l, a, c in zip(plan.unique_angular, alphas, coeffs)
        ]

    all_idx: list[Tensor] = []
    all_val: list[Tensor] = []

    for cl in plan.classes:
        ub, uk, lb, lk = cl.ub, cl.uk, cl.lb, cl.lk
        keep = torch.arange(cl.ib.numel(), device=dev)

        if screening_threshold is not None:
            keep = _screen(
                keep,
                cl.ib,
                cl.jk,
                atom,
                positions.detach(),
                alphas[ub].detach(),
                alphas[uk].detach(),
                coeffs[ub].detach(),
                coeffs[uk].detach(),
                lb + lk + emax,
                screening_threshold,
            )

        step = keep.numel() if chunk_size is None else max(int(chunk_size), 1)
        for start in range(0, keep.numel(), max(step, 1)):
            sel = keep[start : start + step]
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
                if components is None and block_fn is None:
                    return assemble_overlap_1d(
                        kernel, (lb, lk), (a_b, a_k), (c_b, c_k), vec
                    ).unsqueeze(1)

                fn = assemble_multipole_1d if block_fn is None else block_fn
                return fn(
                    kernel,
                    (lb, lk),
                    (a_b, a_k),
                    (c_b, c_k),
                    vec,
                    pos_a,
                    ((0, 0, 0),) if components is None else components,
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

    flat = torch.zeros(ncomp, nao * nao, dtype=dt, device=dev)
    if all_idx:
        # A single out-of-place scatter of the lower-triangular shell blocks;
        # every AO pair occurs at most once, so no accumulation is needed and
        # the backward is a gather.
        flat = flat.index_copy(1, torch.cat(all_idx), torch.cat(all_val, dim=1))

    tri = flat.reshape(ncomp, nao, nao)
    return tri + tri.transpose(-1, -2)


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
    degree: int,
    threshold: float,
) -> Tensor:
    """Indices (into ``bra``/``ket``) of the pairs of one class to keep."""
    a, b = alpha_a[:, None], alpha_b[None, :]
    p = a + b
    pref = (coeff_a[:, None] * coeff_b[None, :]).abs() * (pi / p) ** 1.5
    mu = a * b / p

    kept = []
    for start in range(0, cls.numel(), _SCREEN_CHUNK):
        sel = cls[start : start + _SCREEN_CHUNK]
        d = positions[atom[ket[sel]]] - positions[atom[bra[sel]]]
        r2 = (d * d).sum(-1)
        bound = (pref[None] * torch.exp(-mu[None] * r2[:, None, None])).amax(
            (-2, -1)
        )
        bound = bound * (1.0 + r2.sqrt()) ** degree
        kept.append(sel[(bound > threshold) | (r2 == 0)])

    return torch.cat(kept) if kept else cls[:0]


def assemble_matrix_batch(
    kernel: Kernel1D,
    numbers: Tensor,
    positions: Tensor,
    par: Param,
    components: tuple[tuple[int, int, int], ...] | None = None,
    origin: Tensor | None = None,
    normalize: bool = True,
) -> Tensor:
    """
    Padded-batch version of :func:`assemble_matrix`.

    Padded atoms are identified from ``numbers == 0`` -- never from the
    coordinates, which are ambiguous for molecules lying in a coordinate
    plane (see AGENTS.md, issue #263) -- and are removed *before* any basis
    or pair is built, so the kernels never see them. Each molecule is
    assembled separately and packed with zero padding, so padded rows and
    columns are exactly zero and no gradient can flow from padded atoms.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(nbatch, nat)``, zero-padded.
    positions : Tensor
        Coordinates, shape ``(nbatch, nat, 3)``.
    par : Param
        Parametrization.

    Returns
    -------
    Tensor
        Shape ``(nbatch, ncomp, nao_max, nao_max)``.
    """
    mats = []
    for nums, pos in zip(numbers, positions):
        keep = nums != 0
        nums, pos = nums[keep], pos[keep]
        ihelp = IndexHelper.from_numbers(nums, par)
        alphas, coeffs = Basis(
            nums, par, ihelp, dtype=pos.dtype, device=pos.device
        ).create_cgtos()
        mats.append(
            assemble_matrix(
                kernel,
                ihelp,
                alphas,
                coeffs,
                pos,
                components,
                origin,
                normalize,
            )
        )
    return pack(mats)
