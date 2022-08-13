"""
Definition of repulsion energy terms.
"""

from __future__ import annotations
import warnings

import torch

from ..basis import IndexHelper
from ..exceptions import ParameterWarning
from ..param import Param, get_elem_param
from ..typing import Tensor, TensorLike
from ..utils import real_pairs


class Repulsion(TensorLike):
    """Representation of the classical repulsion."""

    class Cache:
        """Cache for the repulsion energy and gradient."""

        __slots__ = ["alpha", "zeff", "kexp", "mask", "distances"]

        def __init__(
            self,
            alpha: Tensor,
            zeff: Tensor,
            kexp: Tensor,
            mask: Tensor,
            distances: Tensor,
        ):
            self.alpha = alpha
            self.zeff = zeff
            self.kexp = kexp
            self.mask = mask
            self.distances = distances

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        par: Param,
        cutoff: Tensor,
    ) -> None:
        if par.repulsion is None:
            raise ValueError("No repulsion parameters provided")

        self.numbers = numbers
        self.positions = positions
        self.par = par
        self.cutoff = cutoff

        super().__init__(self.positions.device, self.positions.dtype)

    def get_cache(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> "Repulsion.Cache":
        # get parameters for unique species
        unique = torch.unique(numbers)
        arep = get_elem_param(unique, self.par.element, "arep", pad_val=0)
        zeff = get_elem_param(unique, self.par.element, "zeff", pad_val=0)

        _kexp = self.par.repulsion.effective.kexp
        _kexp_light = self.par.repulsion.effective.kexp_light
        kexp = torch.full(unique.shape, _kexp, dtype=self.dtype, device=self.device)
        if _kexp_light is not None:
            kexp = torch.where(unique > 2, kexp.new_tensor(_kexp_light), kexp)

        # spread
        arep = ihelp.spread_uspecies_to_atom(arep)
        zeff = ihelp.spread_uspecies_to_atom(zeff)
        kexp = ihelp.spread_uspecies_to_atom(kexp)

        # mask for padding
        mask = real_pairs(self.numbers, diagonal=True)

        # create matrices
        a = torch.sqrt(arep.unsqueeze(-1) * arep.unsqueeze(-2)) * mask
        z = zeff.unsqueeze(-2) * zeff.unsqueeze(-1) * mask
        k = kexp.unsqueeze(-2) * kexp.new_ones(kexp.shape).unsqueeze(-1) * mask

        distances = torch.where(
            mask,
            torch.cdist(
                positions,
                positions,
                p=2,
                compute_mode="use_mm_for_euclid_dist",
            ),
            # add epsilon to avoid zero division in some terms
            positions.new_tensor(torch.finfo(self.dtype).eps),
        )

        return self.Cache(a, z, k, mask, distances)

    def get_energy(self, cache: "Repulsion.Cache") -> Tensor:
        """
        Get dispersion energy.

        Returns
        -------
        Tensor
            Atom-resolved dispersion energy.
        """

        # Eq.13: R_AB ** k_f
        r1k = torch.pow(cache.distances, cache.kexp)

        # Eq.13: exp(- (alpha_A * alpha_B)**0.5 * R_AB ** k_f )
        exp_term = torch.exp(-cache.alpha * r1k)

        # Eq.13: repulsion energy
        # mask for padding
        dE = torch.where(
            cache.mask * (cache.distances <= self.cutoff),
            cache.zeff * exp_term / cache.distances,
            cache.distances.new_tensor(0.0),
        )

        # atom resolved energies
        return 0.5 * torch.sum(dE, dim=-1)

    def get_grad(self, cache: "Repulsion.Cache") -> Tensor:
        r1k = torch.pow(cache.distances, cache.kexp)

        dG = -(cache.alpha * r1k * cache.kexp + 1.0) * self.get_energy(cache)
        # >>> print(dG.shape)
        # torch.Size([n_batch, n_atoms, n_atoms])

        rij = torch.where(
            cache.mask.unsqueeze(-1),
            self.positions.unsqueeze(-2) - self.positions.unsqueeze(-3),
            cache.distances.new_tensor(0.0),
        )
        # >>> print(rij.shape)
        # torch.Size([n_batch, n_atoms, n_atoms, 3])

        r2 = torch.pow(cache.distances, 2)
        # >>> print(r2.shape)
        # torch.Size([n_batch, n_atoms, n_atoms])

        dG = dG / r2
        dG = dG.unsqueeze(-1) * rij
        # >>> print(dG.shape)
        # torch.Size([n_batch, n_atoms, n_atoms, 3])

        dG = torch.sum(dG, dim=-2)
        # >>> print(dG.shape)
        # torch.Size([n_batch, n_atoms, 3])

        return dG


def new_repulsion(
    numbers: Tensor,
    positions: Tensor,
    par: Param,
    cutoff: Tensor = torch.tensor(25.0),
) -> Repulsion | None:
    """
    Create new instance of Repulsion class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    positions : Tensor
        Cartesian coordinates of all atoms.
    par : Param
        Representation of an extended tight-binding model.


    Returns
    -------
    Repulsion | None
        Instance of the Repulsion class or `None` if no repulsion is used.

    Raises
    ------
    ValueError
        If parametrization does not contain a halogen bond correction.
    """

    if par.repulsion is None:
        # TODO: Repulsion is used in all models, so error or just warning?
        warnings.warn("No repulsion scheme found.", ParameterWarning)
        return None

    return Repulsion(numbers, positions, par, cutoff)
