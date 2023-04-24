"""
Classical repulsion energy contribution
=======================================

This module implements the classical repulsion energy term.

Note
----
The Repulsion class is constructed for geometry optimization, i.e., the atomic
numbers are set upon instantiation (`numbers` is a property), and the parameters
in the cache are created for only those atomic numbers. The positions, however,
must be supplied to the `get_energy` (or `get_grad`) method.

Example
-------
>>> import torch
>>> from xtbml.basis import IndexHelper
>>> from xtbml.classical import new_repulsion
>>> from xtbml.param import GFN1_XTB, get_elem_param
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [+0.00000000000000, +0.00000000000000, +0.00000000000000],
...     [+1.61768389755830, +1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [+1.61768389755830, -1.61768389755830, +1.61768389755830],
...     [-1.61768389755830, +1.61768389755830, +1.61768389755830],
... ])
>>> rep = new_repulsion(numbers, positions, GFN1_XTB)
>>> ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
>>> cache = rep.get_cache(numbers, ihelp)
>>> energy = rep.get_energy(positions, cache)
>>> print(energy.sum(-1))
tensor(0.0303)
"""
from __future__ import annotations

import warnings

import torch

from .._types import Tensor, TensorLike
from ..basis import IndexHelper
from ..constants import xtb
from ..param import Param, get_elem_param
from ..utils import ParameterWarning, real_pairs
from .base import Classical

__all__ = ["Repulsion", "new_repulsion"]


class Repulsion(Classical, TensorLike):
    """
    Representation of the classical repulsion.
    """

    arep: Tensor
    """Atom-specific screening parameters for unique species."""

    zeff: Tensor
    """Effective nuclear charges for unique species."""

    kexp: Tensor
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy.
    """

    klight: Tensor | None
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy for light elements, i.e., H and He (only GFN2).
    """

    cutoff: float
    """Real space cutoff for repulsion interactions (default: 25.0)."""

    __slots__ = ["arep", "zeff", "kexp", "klight", "cutoff"]

    def __init__(
        self,
        arep: Tensor,
        zeff: Tensor,
        kexp: Tensor,
        klight: Tensor | None = None,
        cutoff: float = xtb.DEFAULT_REPULSION_CUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.arep = arep.to(self.device).type(self.dtype)
        self.zeff = zeff.to(self.device).type(self.dtype)
        self.kexp = kexp.to(self.device).type(self.dtype)
        self.cutoff = cutoff

        if klight is not None:
            klight = klight.to(self.device).type(self.dtype)
        self.klight = klight

    class Cache(TensorLike):
        """
        Cache for the repulsion parameters.
        """

        arep: Tensor
        """Atom-specific screening parameters."""

        zeff: Tensor
        """Effective nuclear charges."""

        kexp: Tensor
        """
        Scaling of the interatomic distance in the exponential damping function
        of the repulsion energy.
        """

        mask: Tensor
        """Mask for padding from numbers."""

        __slots__ = ["mask", "alpha", "zeff", "kexp"]

        def __init__(
            self,
            mask: Tensor,
            alpha: Tensor,
            zeff: Tensor,
            kexp: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            super().__init__(
                device=device if device is None else alpha.device,
                dtype=dtype if dtype is None else alpha.dtype,
            )
            self.mask = mask
            self.alpha = alpha
            self.zeff = zeff
            self.kexp = kexp

    def get_cache(self, numbers: Tensor, ihelp: IndexHelper) -> Repulsion.Cache:
        """
        Store variables for energy and gradient calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms.
        ihelp : IndexHelper
            Helper class for indexing.

        Returns
        -------
        Repulsion.Cache
            Cache for repulsion.

        Note
        ----
        The cache of a classical contribution does not require `positions` as
        it only becomes useful if `numbers` remain unchanged and `positions`
        vary, i.e., during geometry optimization.
        """

        # spread
        arep = ihelp.spread_uspecies_to_atom(self.arep)
        zeff = ihelp.spread_uspecies_to_atom(self.zeff)
        kexp = ihelp.spread_uspecies_to_atom(
            self.kexp.expand(torch.unique(numbers).shape)
        )

        # mask for padding
        mask = real_pairs(numbers, diagonal=True)

        # create matrices
        a = torch.sqrt(arep.unsqueeze(-1) * arep.unsqueeze(-2)) * mask
        z = zeff.unsqueeze(-1) * zeff.unsqueeze(-2) * mask
        k = kexp.unsqueeze(-1) * kexp.new_ones(kexp.shape).unsqueeze(-2) * mask

        # GFN2 uses a different value for H and He
        if self.klight is not None:
            kmask = ~real_pairs(numbers <= 2)
            k = torch.where(kmask, k, self.klight) * mask

        return self.Cache(mask, a, z, k)

    def get_energy(
        self, positions: Tensor, cache: Repulsion.Cache, atom_resolved: bool = True
    ) -> Tensor:
        """
        Get repulsion energy.

        Parameters
        ----------
        cache : Repulsion.Cache
            Cache for repulsion.
        positions : Tensor
            Cartesian coordinates of all atoms.
        atom_resolved : bool
            Whether to return atom-resolved energy (True) or full matrix (False).

        Returns
        -------
        Tensor
            (Atom-resolved) repulsion energy.
        """

        distances = torch.where(
            cache.mask,
            torch.cdist(
                positions,
                positions,
                p=2,
                compute_mode="use_mm_for_euclid_dist",
            ),
            # add epsilon to avoid zero division in some terms
            positions.new_tensor(torch.finfo(self.dtype).eps),
        )

        # Eq.13: R_AB ** k_f
        r1k = torch.pow(distances, cache.kexp)

        # Eq.13: exp(- (alpha_A * alpha_B)**0.5 * R_AB ** k_f )
        exp_term = torch.exp(-cache.alpha * r1k)

        # Eq.13: repulsion energy
        e = torch.where(
            cache.mask * (distances <= distances.new_tensor(self.cutoff)),
            cache.zeff * exp_term / distances,
            distances.new_tensor(0.0),
        )

        if atom_resolved is True:
            return 0.5 * torch.sum(e, dim=-1)
        else:
            return e

    def get_gradient_analytical(
        self, positions: Tensor, cache: Repulsion.Cache
    ) -> Tensor:
        """
        Get analytical gradient of repulsion energy.

        Parameters
        ----------
        cache : Repulsion.Cache
            Cache for repulsion.
        positions : Tensor
            Cartesian coordinates of all atoms.

        Returns
        -------
        Tensor
            Atom-resolved repulsion gradient, size (n_batch, n_atoms, 3).
        """

        distances = torch.where(
            cache.mask,
            torch.cdist(
                positions,
                positions,
                p=2,
                compute_mode="use_mm_for_euclid_dist",
            ),
            # add epsilon to avoid zero division in some terms
            positions.new_tensor(torch.finfo(self.dtype).eps),
        )

        r1k = torch.pow(distances, cache.kexp)

        # (n_batch, n_atoms, n_atoms)
        e = self.get_energy(positions, cache, False)
        grad = -(cache.alpha * r1k * cache.kexp + 1.0) * e

        # (n_batch, n_atoms, n_atoms, 3)
        rij = torch.where(
            cache.mask.unsqueeze(-1),
            positions.unsqueeze(-2) - positions.unsqueeze(-3),
            distances.new_tensor(0.0),
        )

        # (n_batch, n_atoms, n_atoms)
        r2 = torch.pow(distances, 2)

        # (n_batch, n_atoms, n_atoms, 3)
        grad = grad / r2
        grad = grad.unsqueeze(-1) * rij

        # (n_batch, n_atoms, 3)
        return torch.sum(grad, dim=-2)


def new_repulsion(
    numbers: Tensor,
    par: Param,
    cutoff: float = xtb.DEFAULT_REPULSION_CUTOFF,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Repulsion | None:
    """
    Create new instance of Repulsion class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    par : Param
        Representation of an extended tight-binding model.
    cutoff : float
        Real space cutoff for repulsion interactions (default: 25.0).

    Returns
    -------
    Repulsion | None
        Instance of the Repulsion class or `None` if no repulsion is used.

    Raises
    ------
    ValueError
        If parametrization does not contain a halogen bond correction.
    """

    if hasattr(par, "repulsion") is False or par.repulsion is None:
        # TODO: Repulsion is used in all models, so error or just warning?
        warnings.warn("No repulsion scheme found.", ParameterWarning)
        return None

    kexp = torch.tensor(par.repulsion.effective.kexp)
    klight = (
        torch.tensor(par.repulsion.effective.klight)
        if par.repulsion.effective.klight
        else None
    )

    # get parameters for unique species
    unique = torch.unique(numbers)
    arep = get_elem_param(unique, par.element, "arep", pad_val=0)
    zeff = get_elem_param(unique, par.element, "zeff", pad_val=0)

    return Repulsion(arep, zeff, kexp, klight, cutoff, device=device, dtype=dtype)
