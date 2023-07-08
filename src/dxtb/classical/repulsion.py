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
from ..utils import ParameterWarning, cdist, real_pairs
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

        __slots__ = ["mask", "arep", "zeff", "kexp"]

        def __init__(
            self,
            mask: Tensor,
            arep: Tensor,
            zeff: Tensor,
            kexp: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            super().__init__(
                device=device if device is None else arep.device,
                dtype=dtype if dtype is None else arep.dtype,
            )
            self.mask = mask
            self.arep = arep
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

        # Without the eps, the first backward returns nan's as described in
        # https://github.com/pytorch/pytorch/issues/2421. The second backward
        # gives nan's in gradgradcheck, because the epsilon is smaller than the
        # step size. But the actual gradient should be correct.
        eps = torch.finfo(arep.dtype).tiny
        a = torch.where(
            mask,
            torch.sqrt(arep.unsqueeze(-1) * arep.unsqueeze(-2) + eps),
            torch.tensor(0.0, device=self.device, dtype=self.dtype),
        )

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
        e = RepulsionAG.apply(
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


def repulsion_energy(
    positions: Tensor,
    mask: Tensor,
    arep: Tensor,
    kexp: Tensor,
    zeff: Tensor,
    cutoff: float = xtb.DEFAULT_REPULSION_CUTOFF,
) -> Tensor:
    """
    Clasical repulsion energy.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms.
    mask : Tensor
        Mask for padding.
    arep : Tensor
        Atom-specific screening parameters.
    kexp : Tensor
        Scaling of the interatomic distance in the exponential damping function
        of the repulsion energy.
    zeff : Tensor
        Effective nuclear charges.
    cutoff : float, optional
        Real-space cutoff. Defaults to `xtb.DEFAULT_REPULSION_CUTOFF`.

    Returns
    -------
    Tensor
        Atom-resolved repulsion energy
    """
    eps = torch.tensor(
        torch.finfo(positions.dtype).eps,
        dtype=positions.dtype,
        device=positions.device,
    )
    zero = torch.tensor(
        0.0,
        dtype=positions.dtype,
        device=positions.device,
    )
    _cutoff = torch.tensor(
        cutoff,
        dtype=positions.dtype,
        device=positions.device,
    )

    distances = torch.where(
        mask,
        cdist(positions, positions, p=2),
        eps,
    )

    # Eq.13: R_AB ** k_f
    r1k = torch.pow(distances, kexp)

    # Eq.13: exp(- (alpha_A * alpha_B)**0.5 * R_AB ** k_f )
    exp_term = torch.exp(-arep * r1k)

    # Eq.13: repulsion energy
    erep = torch.where(
        mask * (distances <= _cutoff),
        zeff * exp_term / (distances + eps),
        zero,
    )
    return erep


def repulsion_gradient(
    erep: Tensor,
    positions: Tensor,
    mask: Tensor,
    arep: Tensor,
    kexp: Tensor,
    *,
    reduced: bool = False,
) -> Tensor:
    """
    Nuclear gradient of classical repulsion energy.

    Parameters
    ----------
    erep : Tensor
        Atom-resolved repulsion energy (from `repulsion_energy`).
    positions : Tensor
        Cartesian coordinates of all atoms.
    mask : Tensor
        Mask for padding.
    arep : Tensor
        Atom-specific screening parameters.
    kexp : Tensor
        Scaling of the interatomic distance in the exponential damping function
        of the repulsion energy.
    reduced : bool, optional
        Shape of the output gradient. Defaults to `False`, which returns a
        gradient of shape `(natoms, natoms, 3)`. This is required for the custom
        backward function. If `reduced=True`, the output gradient has the
        typical shape `(natoms, 3)`.

    Returns
    -------
    Tensor
        Nuclear gradient of repulsion energy. The shape is specified by the
        `reduced` keyword argument.
    """
    eps = torch.tensor(
        torch.finfo(positions.dtype).eps,
        dtype=positions.dtype,
        device=positions.device,
    )

    distances = torch.where(
        mask,
        cdist(positions, positions, p=2),
        eps,
    )

    r1k = torch.pow(distances, kexp)

    # (n_batch, n_atoms, n_atoms)
    grad = -(arep * r1k * kexp + 1.0) * erep

    # (n_batch, n_atoms, n_atoms, 3)
    rij = torch.where(
        mask.unsqueeze(-1),
        positions.unsqueeze(-2) - positions.unsqueeze(-3),
        eps,
    )

    # (n_batch, n_atoms, n_atoms)
    r2 = torch.pow(distances, 2)

    # (n_batch, n_atoms, n_atoms, 3)
    grad = torch.where(mask, grad / (r2 + eps), eps)
    grad = grad.unsqueeze(-1) * rij

    # reduction gives (n_batch, n_atoms, 3)
    return grad if reduced is False else torch.sum(grad, dim=-2)


class RepulsionAG(torch.autograd.Function):
    """
    Autograd function for repulsion energy.
    """

    # generate_vmap_rule = True
    # https://pytorch.org/docs/master/notes/extending.func.html#automatically-generate-a-vmap-rule
    # should work since we only use PyTorch operations

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

    # @staticmethod
    # def forward(
    #     positions: Tensor,
    #     mask: Tensor,
    #     arep: Tensor,
    #     kexp: Tensor,
    #     zeff: Tensor,
    #     cutoff: float,
    # ) -> Tensor:
    #     with torch.enable_grad():
    #         erep = repulsion_energy(positions, mask, arep, kexp, zeff, cutoff)

    #     return erep.clone()

    # @staticmethod
    # def setup_context(ctx, inputs: tuple[Tensor, ...], output: Tensor) -> None:
    #     positions, mask, arep, kexp, zeff, _ = inputs
    #     erep = output

    #     ctx.mark_non_differentiable(mask)
    #     ctx.save_for_backward(erep, positions, mask, arep, kexp, zeff)

    @staticmethod
    def backward(
        ctx, grad_out: Tensor
    ) -> tuple[
        None | Tensor,  # positions
        None,  # mask
        None | Tensor,  # arep
        None | Tensor,  # kexp
        None | Tensor,  # zeff
        None,  # cutoff
    ]:
        # initialize gradients with `None`
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
            _gi = torch.einsum("...ij,...ijd->...id", grad_out, g)
            _gj = torch.einsum("...ij,...ijd->...jd", grad_out, g)
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
