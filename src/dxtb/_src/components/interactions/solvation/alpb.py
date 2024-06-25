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
Analytical linearized Poisson-Boltzmann model
=============================================

This module implements implicit solvation models of the generalized Born type.

Example
-------
.. code-block:: python

    import torch
    from dxtb.solvation.alpb import GeneralizedBorn

    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [0.00000000000000, -0.00000000000000, 0.00000000000000],
        [1.61768389755830, 1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [1.61768389755830, -1.61768389755830, 1.61768389755830],
        [-1.61768389755830, 1.61768389755830, 1.61768389755830],
    ])
    charges = torch.tensor([
        -8.41282505804719e-2,
        2.10320626451180e-2,
        2.10320626451178e-2,
        2.10320626451179e-2,
        2.10320626451179e-2,
    ])

    # Initialize the GeneralizedBorn model with a solvent dielectric constant
    gb = GeneralizedBorn(numbers, torch.tensor(78.9), kernel="still")

    # Build cache and use it for energy calculation
    cache = gb.get_cache(numbers, positions)
    energy = gb.get_atom_energy(charges, cache)

    total_energy = energy.sum(-1)
    print(total_energy)  # Output: tensor(-5.0762e-05)

"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.data import VDW_D3
from tad_mctc.math import einsum

from dxtb._src.param import Param
from dxtb._src.typing import DD, Any, Tensor, TensorLike, get_default_dtype
from dxtb._src.typing.exceptions import DeviceError

from ..base import Interaction, InteractionCache
from .born import get_born_radii

alpha = 0.571412

DEFAULT_KERNEL = "p16"
DEFAULT_ALPB = True

__all__ = ["GeneralizedBorn", "new_solvation"]


@torch.jit.script
def p16_kernel(r1: Tensor, ab: Tensor) -> Tensor:
    """
    Evaluate P16 interaction kernel: 1 / (R + √ab / (1 + ζR/(16·√ab))¹⁶)

    Parameters
    ----------
    r1 : Tensor
        Distance between all atom pairs
    ab : Tensor
        Product of Born radii

    Returns
    -------
    Tensor
        Interaction kernel between all atom pairs
    """

    ab = torch.sqrt(ab)
    arg = torch.pow(ab / (ab + r1 * 1.028 / 16), 16)

    return 1.0 / (r1 + ab * arg)


@torch.jit.script
def still_kernel(r1: Tensor, ab: Tensor) -> Tensor:
    """
    Evaluate Still interaction kernel: 1 / √(R² + ab · exp[R²/(4·ab)])

    Parameters
    ----------
    r1 : Tensor
        Distance between all atom pairs
    ab : Tensor
        Product of Born radii

    Returns
    -------
    Tensor
        Interaction kernel between all atom pairs
    """

    r2 = torch.pow(r1, 2)
    arg = torch.exp(-0.25 * r2 / ab)

    return 1.0 / torch.sqrt(r2 + ab * arg)


born_kernel = {"p16": p16_kernel, "still": still_kernel}


def get_adet(positions: Tensor, rad: Tensor) -> Tensor:
    """
    Calculate electrostatic shape function based on the moments of inertia
    of solid spheres.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    rad : Tensor
        Radii of all atoms.

    Returns
    -------
    Tensor
        Electrostatic shape function.
    """

    vol = torch.pow(rad, 3)
    center = (positions * vol.unsqueeze(-1)).sum(-2) / vol.sum(-1)

    displ = positions - center.unsqueeze(-2)
    diag = torch.pow(displ, 2).sum(-1) + 2 * torch.pow(rad, 2) / 5
    inertia = (
        vol.unsqueeze(-1).unsqueeze(-2)
        * (
            -displ.unsqueeze(-1) * displ.unsqueeze(-2)
            + torch.diag_embed(diag.unsqueeze(-1).expand(*positions.shape))
        )
    ).sum(-3)

    adet = (
        +inertia[..., 0, 0] * inertia[..., 1, 1] * inertia[..., 2, 2]
        - inertia[..., 0, 0] * inertia[..., 1, 2] * inertia[..., 2, 1]
        - inertia[..., 0, 1] * inertia[..., 1, 0] * inertia[..., 2, 2]
        + inertia[..., 0, 1] * inertia[..., 1, 2] * inertia[..., 2, 0]
        + inertia[..., 0, 2] * inertia[..., 1, 0] * inertia[..., 2, 1]
        - inertia[..., 0, 2] * inertia[..., 1, 1] * inertia[..., 2, 0]
    )

    return torch.sqrt(5 * torch.pow(adet, 1 / 3) / (2 * vol.sum(-1)))


class GeneralizedBornCache(InteractionCache, TensorLike):
    """
    Restart data for the generalized Born solvation model.
    """

    __slots__ = ["mat"]

    mat: Tensor
    """Coulomb matrix."""

    def __init__(
        self,
        mat: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            device=device if device is None else mat.device,
            dtype=dtype if dtype is None else mat.dtype,
        )
        self.mat = mat


class GeneralizedBorn(Interaction):
    """
    Implicit solvation model for describing the interaction with a dielectric continuum.
    """

    kernel: str
    """Interaction kernel."""

    alpbet: Tensor
    """Finite dielectric constant correction."""

    keps: Tensor
    """Dielectric function."""

    born_kwargs: dict[str, Any]
    """Parameters for Born radii integration."""

    def __init__(
        self,
        numbers: Tensor,
        dielectric_constant: Tensor,
        alpb: bool = DEFAULT_ALPB,
        kernel: str = DEFAULT_KERNEL,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device, dtype)

        self.alpbet = (
            alpha / dielectric_constant if alpb else torch.tensor(0.0, **self.dd)
        )
        self.keps = (1 / dielectric_constant - 1) / (1 + self.alpbet)
        self.kernel = kernel

        if "rvdw" not in kwargs:
            kwargs["rvdw"] = VDW_D3.to(**self.dd)[numbers]
        self.born_kwargs = kwargs

    def get_cache(
        self, numbers: Tensor, positions: Tensor, **_
    ) -> GeneralizedBornCache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

        Returns
        -------
        GeneralizedBornCache
            Cache object for second order electrostatics.

        Note
        ----
        If the `GeneralizedBorn` interaction is evaluated within the
        `InteractionList`, the `IndexHelper` will be passed as argument.
        Hence, the `**_` in the argument list is necessary to absorb it.
        """
        cachvars = (numbers.detach().clone(), positions.detach().clone())

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, GeneralizedBornCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        # if the cache is built, store the positions for validation
        self._cachevars = cachvars

        born = get_born_radii(numbers, positions, **self.born_kwargs)
        eps = torch.tensor(torch.finfo(positions.dtype).eps, **self.dd)

        mask = real_pairs(numbers, mask_diagonal=False)

        dist = torch.where(mask, storch.cdist(positions, p=2), eps)
        ab = torch.where(mask, born.unsqueeze(-1) * born.unsqueeze(-2), eps)

        mat = self.keps * born_kernel[self.kernel](dist, ab)

        if self.alpbet > 0:
            adet = get_adet(positions, self.born_kwargs["rvdw"])
            mat += self.keps * self.alpbet * adet.unsqueeze(-1).unsqueeze(-2)

        self.cache = GeneralizedBornCache(mat)
        return self.cache

    def get_atom_energy(self, charges: Tensor, cache: GeneralizedBornCache) -> Tensor:
        return 0.5 * charges * self.get_atom_potential(charges, cache)

    def get_atom_potential(
        self, charges: Tensor, cache: GeneralizedBornCache
    ) -> Tensor:
        return einsum("...ik,...k->...i", cache.mat, charges)

    # TODO: Implement gradient before using solvation in SCF
    def get_atom_gradient(
        self,
        numbers: Tensor,
        positions: Tensor,
        charges: Tensor,
        cache: GeneralizedBornCache,
    ) -> Tensor:
        raise NotImplementedError("Solvation gradient not implemented")


def new_solvation(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> GeneralizedBorn | None:
    """
    Create new instance of the generalized Born solvation model.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    GeneralizedBorn | None
        Instance of the `GeneralizedBorn` class or ``None`` if no solvation model
        is used.
    """
    if hasattr(par, "solvation") is False or par.solvation is None:
        return None

    if device is not None:
        if device != numbers.device:
            raise DeviceError(
                f"Passed device ({device}) and device of electric field "
                f"({numbers.device}) do not match."
            )

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    s = par.solvation  # type: ignore
    epsilon = torch.tensor(s.epsilon.gexp, **dd)
    alpb = s.alpb if hasattr(s, "alpb") else DEFAULT_ALPB
    kernel = s.kernel if hasattr(s, "kernel") else DEFAULT_KERNEL

    return GeneralizedBorn(
        numbers, dielectric_constant=epsilon, alpb=alpb, kernel=kernel, **dd
    )
