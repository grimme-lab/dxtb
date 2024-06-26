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
PyTorch-based overlap implementations.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import deflate, pack
from tad_mctc.math import einsum

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.constants import defaults
from dxtb._src.typing import Any, Literal, Tensor
from dxtb._src.utils import t2int

from .md import overlap_gto, overlap_gto_grad
from .md.utils import get_pairs, get_subblock_start

__all__ = ["OverlapAG_V1", "OverlapAG_V2", "overlap", "overlap_gradient"]


class OverlapAGBase(torch.autograd.Function):
    """
    Base class for the version-specific autograd function for the Overlap.
    Different PyTorch versions only require different `forward()` signatures.
    """

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[
        None | Tensor,  # positions
        None,  # bas
        None,  # ihelp
        None,  # uplo
        None,  # cutoff
    ]:
        # initialize gradients with ``None``
        positions_bar = None

        # check which of the input variables of `forward()` requires gradients
        grad_positions, _, _, _, _ = ctx.needs_input_grad

        positions: Tensor = ctx.saved_tensors[0]
        bas: Basis = ctx.bas
        ihelp: IndexHelper = ctx.ihelp
        uplo: Literal["n", "u", "l"] = ctx.uplo
        cutoff: Tensor | float | int | None = ctx.cutoff

        # analytical gradient for positions
        if grad_positions:
            # We only only calculate the gradient of the overlap w.r.t. one
            # center, but the overlap w.r.t the second center is simply the
            # transpose of the first. The shape of the returned gradient is
            # (nbatch, norb, norb, 3).
            grad_i = overlap_gradient(positions, bas, ihelp, uplo, cutoff)
            grad_j = grad_i.transpose(-3, -2)

            # vjp: (nb, norb, norb) * (nb, norb, norb, 3) -> (nb, norb, 3)
            _gi = einsum("...ij,...ijd->...id", grad_out, grad_i)
            _gj = einsum("...ij,...ijd->...jd", grad_out, grad_j)
            positions_bar = _gi + _gj

            # Finally, we need to reduce the gradient to conform with the
            # expected shape, which is equal to that of the variable w.r.t.
            # which we differentiate, i.e., the positions. Hence, the final
            # reduction does: (nb, norb, 3) -> (nb, natom, 3)
            positions_bar = ihelp.reduce_orbital_to_atom(
                positions_bar, dim=-2, extra=True
            )

        return positions_bar, None, None, None, None


class OverlapAG_V1(OverlapAGBase):
    """
    Autograd function for overlap integral evaluation.
    """

    # TODO: For a more efficient gradient computation, we could also calculate
    # the gradient in the forward pass according to
    # https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html#saving-intermediate-results
    @staticmethod
    def forward(
        ctx: Any,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
    ) -> Tensor:
        ctx.save_for_backward(positions)
        ctx.bas = bas
        ctx.ihelp = ihelp
        ctx.uplo = uplo
        ctx.cutoff = cutoff

        return overlap(positions, bas, ihelp, uplo, cutoff)


class OverlapAG_V2(OverlapAGBase):
    """
    Autograd function for overlap integral evaluation.
    """

    generate_vmap_rule = True
    # https://pytorch.org/docs/master/notes/extending.func.html#automatically-generate-a-vmap-rule
    # should work since we only use PyTorch operations

    @staticmethod
    def forward(
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
    ) -> Tensor:
        return overlap(positions, bas, ihelp, uplo, cutoff)

    @staticmethod
    def setup_context(ctx, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        positions, bas, ihelp, uplo, cutoff = inputs

        ctx.save_for_backward(positions)
        ctx.bas = bas
        ctx.ihelp = ihelp
        ctx.uplo = uplo
        ctx.cutoff = cutoff


def overlap(
    positions: Tensor,
    bas: Basis,
    ihelp: IndexHelper,
    uplo: Literal["n", "u", "l"] = "l",
    cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
) -> Tensor:
    """
    Calculate the full overlap matrix.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    bas : Basis
        Basis set information.
    ihelp : IndexHelper
        Helper class for indexing.
    uplo : Literal['n';, 'u', 'l'], optional
        Whether the matrix of unique shell pairs should be create as a
        triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
        Defaults to `l` (lower triangular matrix).
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff for integral calculation in Bohr. Defaults to
        `constants.defaults.INTCUTOFF`.

    Returns
    -------
    Tensor
        Orbital-resolved overlap matrix of shape `(nb, norb, norb)`.
    """
    alphas, coeffs = bas.create_cgtos()

    # spread stuff to orbitals for indexing
    alpha = ihelp.spread_ushell_to_orbital(pack(alphas), dim=-2, extra=True)
    coeff = ihelp.spread_ushell_to_orbital(pack(coeffs), dim=-2, extra=True)
    pos = ihelp.spread_atom_to_orbital(positions, dim=-2, extra=True)
    ang = ihelp.spread_shell_to_orbital(ihelp.angular)

    # real-space integral cutoff; assumes orthogonalization of basis
    # functions as "self-overlap" is  explicitly removed with `dist > 0`
    if cutoff is None:
        mask = None
    else:
        # cdist does not return zero for distance between same vectors
        # https://github.com/pytorch/pytorch/issues/57690
        dist = storch.cdist(pos, pos)
        mask = (dist < cutoff) & (dist > 0.1)

    umap, n_unique_pairs = bas.unique_shell_pairs(mask=mask, uplo=uplo)

    # overlap calculation
    ovlp = torch.zeros(*umap.shape, dtype=positions.dtype, device=positions.device)

    for uval in range(n_unique_pairs):
        pairs = get_pairs(umap, uval)
        first_pair = pairs[0]

        angi, angj = ang[first_pair]
        norbi = 2 * t2int(angi) + 1
        norbj = 2 * t2int(angj) + 1

        # collect [0, 0] entry of each subblock
        upairs = get_subblock_start(umap, uval, norbi, norbj)

        # we only require one pair as all have the same basis function
        alpha_tuple = (
            deflate(alpha[first_pair][0]),
            deflate(alpha[first_pair][1]),
        )
        coeff_tuple = (
            deflate(coeff[first_pair][0]),
            deflate(coeff[first_pair][1]),
        )
        ang_tuple = (angi, angj)

        vec = pos[upairs][:, 0, :] - pos[upairs][:, 1, :]
        stmp = overlap_gto(ang_tuple, alpha_tuple, coeff_tuple, -vec)

        # write overlap of unique pair to correct position in full overlap matrix
        for r, pair in enumerate(upairs):
            ovlp[
                pair[0] : pair[0] + norbi,
                pair[1] : pair[1] + norbj,
            ] = stmp[r]

    # fill empty triangular matrix
    if uplo == "l":
        ovlp = torch.tril(ovlp, diagonal=-1) + torch.triu(ovlp.mT)
    elif uplo == "u":
        ovlp = torch.triu(ovlp, diagonal=1) + torch.tril(ovlp.mT)

    # fix diagonal as "self-overlap" was removed via mask earlier
    ovlp.fill_diagonal_(1.0)
    return ovlp


def overlap_gradient(
    positions: Tensor,
    bas: Basis,
    ihelp: IndexHelper,
    uplo: Literal["n", "u", "l"] = "l",
    cutoff: Tensor | float | int | None = None,
) -> Tensor:
    """
    Calculate the gradient of the overlap.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    bas : Basis
        Basis set information.
    ihelp : IndexHelper
        Helper class for indexing.
    uplo : Literal['n';, 'u', 'l'], optional
        Whether the matrix of unique shell pairs should be create as a
        triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
        Defaults to `l` (lower triangular matrix).
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff for integral calculation in Angstrom. Defaults to
        `constants.defaults.INTCUTOFF` (50.0).

    Returns
    -------
    Tensor
        Orbital-resolved overlap gradient of shape `(nb, norb, norb, 3)`.
    """
    alphas, coeffs = bas.create_cgtos()

    # spread stuff to orbitals for indexing
    alpha = ihelp.spread_ushell_to_orbital(pack(alphas), dim=-2, extra=True)
    coeff = ihelp.spread_ushell_to_orbital(pack(coeffs), dim=-2, extra=True)
    pos = ihelp.spread_atom_to_orbital(positions, dim=-2, extra=True)
    ang = ihelp.spread_shell_to_orbital(ihelp.angular)

    # real-space integral cutoff; assumes orthogonalization of basis
    # functions as "self-overlap" is explicitly removed with `dist > 0`
    if cutoff is None:
        mask = None
    else:
        # cdist does not return zero for distance between same vectors
        # https://github.com/pytorch/pytorch/issues/57690
        dist = storch.cdist(pos, pos)
        mask = (dist < cutoff) & (dist > 0.1)

    umap, n_unique_pairs = bas.unique_shell_pairs(mask=mask, uplo=uplo)

    # overlap calculation
    ds = torch.zeros((3, *umap.shape), dtype=positions.dtype, device=positions.device)

    # loop over unique pairs
    for uval in range(n_unique_pairs):
        pairs = get_pairs(umap, uval)
        first_pair = pairs[0]

        li, lj = ang[first_pair]
        norbi = 2 * t2int(li) + 1
        norbj = 2 * t2int(lj) + 1

        # collect [0, 0] entry of each subblock
        upairs = get_subblock_start(umap, uval, norbi, norbj, uplo)

        # we only require one pair as all have the same basis function
        alpha_tuple = (
            deflate(alpha[first_pair][0]),
            deflate(alpha[first_pair][1]),
        )
        coeff_tuple = (
            deflate(coeff[first_pair][0]),
            deflate(coeff[first_pair][1]),
        )
        ang_tuple = (li, lj)

        vec = pos[upairs][:, 0, :] - pos[upairs][:, 1, :]

        # NOTE: We could also use the overlap (from the grad func) here, but
        # the program is structured differently at the moment. Hence, we do not
        # need the overlap here and save us the writing step to the full
        # matrix, which is actually the bottleneck of the overlap routines.
        dstmp = overlap_gto_grad(ang_tuple, alpha_tuple, coeff_tuple, -vec)

        # Write overlap of unique pair to correct position in full matrix
        # This loop is the bottleneck of the whole integral evaluation.
        for r, pair in enumerate(upairs):
            ds[
                :,
                pair[0] : pair[0] + norbi,
                pair[1] : pair[1] + norbj,
            ] = dstmp[r]

    # fill empty triangular matrix
    if uplo == "l":
        ds = torch.tril(ds, diagonal=-1) - torch.triu(ds.mT)
    elif uplo == "u":
        ds = torch.triu(ds, diagonal=1) - torch.tril(ds.mT)

    # (3, norb, norb) -> (norb, norb, 3)
    return einsum("xij->ijx", ds)
