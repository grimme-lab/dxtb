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
Repulsion: Classes
==================

This module implements the classical repulsion energy term in two flavors. The
first class, class:`.Repulsion`, provides the gradient using PyTorch's
autograd. The second class, class:`.RepulsionAnalytical`, provides an custom
backward with an analytical derivative.

Note
----
The Repulsion class has a cache scope that goes beyond single-point
calculations (geometry optimization, numerical gradients). The atomic numbers
are set upon instantiation (``numbers`` is a property), and the parameters in
the cache are created for only those atomic numbers. The positions, however,
must be supplied to the ``get_energy`` method. Hence, the cache does not become
invalid for different geometries, but only for different atomic numbers.
"""

from __future__ import annotations

import torch
from tad_mctc._version import __tversion__
from tad_mctc.math import einsum

from dxtb._src.typing import Any, Tensor, override

from .base import (
    BaseRepulsion,
    BaseRepulsionCache,
    repulsion_energy,
    repulsion_gradient,
)

__all__ = ["LABEL_REPULSION", "Repulsion", "RepulsionAnalytical"]


LABEL_REPULSION = "Repulsion"
"""Label for the 'Repulsion' component, coinciding with the class name."""


class Repulsion(BaseRepulsion):
    """
    Representation of the classical repulsion.
    """

    @override
    def get_energy(
        self, positions: Tensor, cache: BaseRepulsionCache, **kwargs: Any
    ) -> Tensor:
        """
        Get repulsion energy.

        Parameters
        ----------
        cache : Repulsion.Cache
            Cache for repulsion.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        atom_resolved : bool
            Whether to return atom-resolved energy (True) or full matrix
            (False).

        Returns
        -------
        Tensor
            (Atom-resolved) repulsion energy.
        """
        e = repulsion_energy(
            positions,
            cache.mask,
            cache.arep,
            cache.kexp,
            cache.zeff,
            self.cutoff,
        )

        if kwargs.get("atom_resolved", True) is True:
            return 0.5 * torch.sum(e, dim=-1)
        return e


class RepulsionAnalytical(Repulsion):
    """
    Representation of the classical repulsion.
    """

    @override
    def get_energy(
        self,
        positions: Tensor,
        cache: BaseRepulsionCache,
        atom_resolved: bool = True,
    ) -> Tensor:
        """
        Get repulsion energy. This function employs the custom autograd class
        to provide an analytical first derivative.

        Parameters
        ----------
        cache : Repulsion.Cache
            Cache for repulsion.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        atom_resolved : bool
            Whether to return atom-resolved energy (True) or full matrix (False).

        Returns
        -------
        Tensor
            (Atom-resolved) repulsion energy.
        """
        _RepulsionAG = (
            RepulsionAG_V1 if __tversion__ < (2, 0, 0) else RepulsionAG_V2
        )  # pragma: no cover

        e = _RepulsionAG.apply(
            positions,
            cache.mask,
            cache.arep,
            cache.kexp,
            cache.zeff,
            self.cutoff,
        )
        assert e is not None

        if atom_resolved is True:
            return 0.5 * torch.sum(e, dim=-1)
        return e


class RepulsionAGBase(torch.autograd.Function):
    """
    Base class for the version-specific autograd function for repulsion energy.
    Different PyTorch versions only require different `forward()` signatures.
    """

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[
        None | Tensor,  # positions
        None,  # mask
        None | Tensor,  # arep
        None | Tensor,  # kexp
        None | Tensor,  # zeff
        None,  # cutoff
    ]:
        # initialize gradients with ``None``
        positions_bar = arep_bar = kexp_bar = zeff_bar = None

        # check which of the input variables of `forward()` requires gradients
        grad_positions, _, grad_arep, grad_kexp, grad_zeff, _ = ctx.needs_input_grad

        erep, positions, mask, arep, kexp, zeff = ctx.saved_tensors

        # analytical gradient for positions
        if grad_positions:
            # (n_batch, n_atoms, n_atoms, 3)
            g = repulsion_gradient(
                erep,
                positions,
                mask,
                arep,
                kexp,
                reduced=False,
            )

            # vjp: (nb, na, na) * (nb, na, na, 3) -> (nb, na, 3)
            _gi = einsum("...ij,...ijd->...id", grad_out, g)
            _gj = einsum("...ij,...ijd->...jd", grad_out, g)
            positions_bar = _gi - _gj

        # automatic gradient for parameters
        if grad_arep:
            (arep_bar,) = torch.autograd.grad(
                erep,
                arep,
                grad_outputs=grad_out,
                create_graph=True,
            )
        if grad_kexp:
            (kexp_bar,) = torch.autograd.grad(
                erep,
                kexp,
                grad_outputs=grad_out,
                create_graph=True,
            )
        if grad_zeff:
            (zeff_bar,) = torch.autograd.grad(
                erep,
                zeff,
                grad_outputs=grad_out,
                create_graph=True,
            )

        return positions_bar, None, arep_bar, kexp_bar, zeff_bar, None


class RepulsionAG_V1(RepulsionAGBase):
    """
    Autograd function for repulsion energy.
    """

    @staticmethod
    def forward(
        ctx,
        positions: Tensor,
        mask: Tensor,
        arep: Tensor,
        kexp: Tensor,
        zeff: Tensor,
        cutoff: float,
    ) -> Tensor:
        with torch.enable_grad():
            erep = repulsion_energy(positions, mask, arep, kexp, zeff, cutoff)

        ctx.mark_non_differentiable(mask)
        ctx.save_for_backward(erep, positions, mask, arep, kexp, zeff)

        return erep.clone()


class RepulsionAG_V2(RepulsionAGBase):
    """
    Autograd function for repulsion energy.
    """

    generate_vmap_rule = True
    # https://pytorch.org/docs/master/notes/extending.func.html#automatically-generate-a-vmap-rule
    # should work since we only use PyTorch operations

    @staticmethod
    def forward(
        positions: Tensor,
        mask: Tensor,
        arep: Tensor,
        kexp: Tensor,
        zeff: Tensor,
        cutoff: float,
    ) -> Tensor:
        with torch.enable_grad():
            erep = repulsion_energy(positions, mask, arep, kexp, zeff, cutoff)

        return erep.clone()

    @staticmethod
    def setup_context(ctx, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        positions, mask, arep, kexp, zeff, _ = inputs
        erep = output

        ctx.mark_non_differentiable(mask)
        ctx.save_for_backward(erep, positions, mask, arep, kexp, zeff)
