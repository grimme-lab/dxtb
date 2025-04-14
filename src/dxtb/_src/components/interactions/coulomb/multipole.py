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
Coulomb: Anisotropic second-order electrostatics (AES2)
=======================================================

This module implements the anisotropic second-order electrostatic energy
that uses a damped multipole expansion.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import eye, real_pairs
from tad_mctc.exceptions import DeviceError
from tad_mctc.math import einsum

from dxtb import IndexHelper
from dxtb._src.param import Param, get_elem_param
from dxtb._src.typing import (
    DD,
    Slicers,
    Tensor,
    TensorLike,
    get_default_dtype,
    override,
)

from ..base import Interaction, InteractionCache

__all__ = ["AES2", "LABEL_AES2", "new_aes2"]


LABEL_AES2 = "AES2"
"""Label for the 'AES2' interaction, coinciding with the class name."""


class AES2Cache(InteractionCache, TensorLike):
    """
    Cache for AES2 interaction.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    mrad: Tensor
    """Multipole damping radii for all atoms."""

    dkernel: Tensor
    """Kernel for on-site dipole exchange-correlation."""

    qkernel: Tensor
    """Kernel for on-site quadrupole exchange-correlation."""

    amat_sd: Tensor
    """
    Interation matrix for charges and dipoles
    (shape: ``(..., nat, nat, 3)``).
    """

    amat_dd: Tensor
    """
    Interation matrix for dipoles and dipoles
    (shape: ``(..., nat, nat, 3, 3)``).
    """

    amat_sq: Tensor
    """
    Interation matrix for charges and quadrupoles
    (shape: ``(..., nat, nat, 6)``).
    """

    __slots__ = [
        "__store",
        "mrad",
        "dkernel",
        "qkernel",
        "amat_sd",
        "amat_dd",
        "amat_sq",
    ]

    def __init__(
        self,
        mrad: Tensor,
        dkernel: Tensor,
        qkernel: Tensor,
        amat_sd: Tensor,
        amat_dd: Tensor,
        amat_sq: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else mrad.device,
            dtype=dtype if dtype is None else mrad.dtype,
        )
        self.mrad = mrad
        self.dkernel = dkernel
        self.qkernel = qkernel
        self.amat_sd = amat_sd
        self.amat_dd = amat_dd
        self.amat_sq = amat_sq

        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        mrad: Tensor
        """Multipole damping radii for all atoms."""

        dkernel: Tensor
        """Kernel for on-site dipole exchange-correlation."""

        qkernel: Tensor
        """Kernel for on-site quadrupole exchange-correlation."""

        amat_sd: Tensor
        """Interation matrix for charges and dipoles."""

        amat_dd: Tensor
        """Interation matrix for dipoles and dipoles."""

        amat_sq: Tensor
        """Interation matrix for charges and quadrupoles."""

        def __init__(
            self,
            mrad: Tensor,
            dkernel: Tensor,
            qkernel: Tensor,
            amat_sd: Tensor,
            amat_dd: Tensor,
            amat_sq: Tensor,
        ) -> None:
            self.mrad = mrad
            self.dkernel = dkernel
            self.qkernel = qkernel
            self.amat_sd = amat_sd
            self.amat_dd = amat_dd
            self.amat_sq = amat_sq

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(
                self.mrad,
                self.dkernel,
                self.qkernel,
                self.amat_sd,
                self.amat_dd,
                self.amat_sq,
            )

        slicer = slicers["atom"]
        self.mrad = self.mrad[[~conv, *slicer]]
        self.dkernel = self.dkernel[[~conv, *slicer]]
        self.qkernel = self.qkernel[[~conv, *slicer]]
        self.amat_sd = self.amat_sd[[~conv, *slicer, *slicer]]
        self.amat_dd = self.amat_dd[[~conv, *slicer, *slicer]]
        self.amat_sq = self.amat_sq[[~conv, *slicer, *slicer]]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.mrad = self.__store.mrad
        self.dkernel = self.__store.dkernel
        self.qkernel = self.__store.qkernel
        self.amat_sd = self.__store.amat_sd
        self.amat_dd = self.__store.amat_dd
        self.amat_sq = self.__store.amat_sq


class AES2(Interaction):
    """
    Isotropic second-order electrostatic energy (ES2).
    """

    dmp3: Tensor
    """Damping function for inverse quadratic contributions."""

    dmp5: Tensor
    """Damping function for inverse cubic contributions."""

    dkernel: Tensor
    """Kernel for on-site dipole exchange-correlation."""

    qkernel: Tensor
    """Kernel for on-site quadrupole exchange-correlation."""

    shift: Tensor
    """Shift for the generation of the multipolar damping radii."""

    kexp: Tensor
    """Exponent for the generation of the multipolar damping radii."""

    rmax: Tensor
    """Maximum radius for the multipolar damping radii."""

    rad: Tensor
    """Base radii for the multipolar damping radii."""

    vcn: Tensor
    """Valence coordination number."""

    __slots__ = [
        "dmp3",
        "dmp5",
        "dkernel",
        "qkernel",
        "shift",
        "kexp",
        "rmax",
        "rad",
        "vcn",
    ]

    def __init__(
        self,
        dmp3: Tensor,
        dmp5: Tensor,
        dkernel: Tensor,
        qkernel: Tensor,
        shift: Tensor,
        kexp: Tensor,
        rmax: Tensor,
        rad: Tensor,
        vcn: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        # scalar parameters
        self.dmp3 = dmp3.to(**self.dd)
        self.dmp5 = dmp5.to(**self.dd)
        self.shift = shift.to(**self.dd)
        self.kexp = kexp.to(**self.dd)
        self.rmax = rmax.to(**self.dd)

        # element-wise parameters
        self.dkernel = dkernel.to(**self.dd)
        self.qkernel = qkernel.to(**self.dd)
        self.rad = rad.to(**self.dd)
        self.vcn = vcn.to(**self.dd)

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> AES2Cache:
        """
        Obtain the cache object.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        AES2Cache
            Cache object for anisotropic second order electrostatics.

        Note
        ----
        The cache of an interaction requires ``positions`` as they do not change
        during the self-consistent charge iterations.
        """
        if numbers is None:
            raise ValueError("Atomic numbers are required for AES2 cache.")
        if positions is None:
            raise ValueError("Positions are required for AES2 cache.")
        if ihelp is None:
            raise ValueError("IndexHelper is required for AES2 cache creation.")

        cachvars = (numbers.detach().clone(), positions.detach().clone())

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, AES2Cache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        # if the cache is built, store the cachvar for validation
        self._cachevars = cachvars

        dkernel = ihelp.spread_uspecies_to_atom(self.dkernel).unsqueeze(-1)
        qkernel = ihelp.spread_uspecies_to_atom(self.qkernel).unsqueeze(-1)

        from tad_mctc.ncoord import cn_d3, gfn2_count

        vcn = ihelp.spread_uspecies_to_atom(self.vcn)
        rad = ihelp.spread_uspecies_to_atom(self.rad)

        cn = cn_d3(numbers, positions, counting_function=gfn2_count)

        # tblite: coulomb/multipole.f90::get_mrad
        t1 = torch.exp(-self.kexp * (cn - vcn - self.shift))
        t2 = (self.rmax - rad) / (1.0 + t1)
        mrad = rad + t2
        # dmradcn = -self.kexp * t2 * t1 / (1 + t1)

        amat_sd, amat_dd, amat_sq = self.get_atom_coulomb_matrix(
            numbers, positions, mrad
        )

        self.cache = AES2Cache(
            mrad=mrad,
            dkernel=dkernel,
            qkernel=qkernel,
            amat_sd=amat_sd,
            amat_dd=amat_dd,
            amat_sq=amat_sq,
        )

        return self.cache

    def get_atom_coulomb_matrix(
        self, numbers: Tensor, positions: Tensor, rad: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the atom-resolved interaction matrices.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Interaction matrices for:
            - charges and dipoles (shape: ``(..., nat, nat, 3)``),
            - dipoles and dipoles (shape: ``(..., nat, nat, 3, 3)``),
            - charges and quadrupoles (shape: ``(..., nat, nat, 6)``).
        """
        eps = torch.tensor(torch.finfo(positions.dtype).eps, **self.dd)
        mask = real_pairs(numbers, mask_diagonal=True)

        dist = storch.cdist(positions, positions, p=2)

        # (nb, nat, nat)
        g1 = storch.reciprocal(dist)
        g3 = g1 * g1 * g1
        g5 = g3 * g1 * g1

        # (nb, nat, nat)
        rr = 0.5 * (rad.unsqueeze(-1) + rad.unsqueeze(-2)) * g1
        fdmp3 = 1.0 / (1.0 + 6.0 * rr**self.dmp3)
        fdmp5 = 1.0 / (1.0 + 6.0 * rr**self.dmp5)

        # (nb, nat, nat, 3)
        rij = torch.where(
            mask.unsqueeze(-1),
            positions.unsqueeze(-2) - positions.unsqueeze(-3),
            eps,
        )

        # Monopole / Dipole

        # (nb, nat, nat, 1)
        _g3 = g3.unsqueeze(-1)
        _fdmp3 = fdmp3.unsqueeze(-1)

        # (nb, nat, nat, 3) * (nb, nat, nat, 1) -> (nb, nat, nat, 3)
        sd = rij * _g3 * _fdmp3

        # Dipole / Dipole

        # (nb, nat, nat, 1)
        _g5 = g5.unsqueeze(-1)
        _fdmp5 = fdmp5.unsqueeze(-1)
        g5_fdmp5 = _g5 * _fdmp5

        # (nb, nat, nat, 1, 1)
        _g5_fdmp5 = g5_fdmp5.unsqueeze(-1)

        # (nb, 1, 1, 3, 3)
        unity = (
            eye((*numbers.shape[:-2], 3, 3), **self.dd)
            .unsqueeze(-3)
            .unsqueeze(-3)
        )

        # (nb, nat, nat, 3, 3)
        dd = (
            unity * _g3.unsqueeze(-1) * _fdmp5.unsqueeze(-1)
            - rij.unsqueeze(-1) * rij.unsqueeze(-2) * 3 * _g5_fdmp5
        )

        # Monopole / Quadrupole

        # (nb, nat, nat, 6)
        sq = torch.empty(
            (*rij.shape[:-1], 6),
            device=positions.device,
            dtype=positions.dtype,
        )

        sq[..., 0] = rij[..., 0] * rij[..., 0]
        sq[..., 2] = rij[..., 1] * rij[..., 1]
        sq[..., 5] = rij[..., 2] * rij[..., 2]

        sq[..., 1] = 2 * rij[..., 0] * rij[..., 1]
        sq[..., 3] = 2 * rij[..., 0] * rij[..., 2]
        sq[..., 4] = 2 * rij[..., 1] * rij[..., 2]

        # (nb, nat, nat, 6) * (nb, nat, nat, 1) -> (nb, nat, nat, 6)
        sq = sq * g5_fdmp5

        return sd, dd, sq

    @override
    def get_dipole_atom_energy(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate atom-resolved dipolar energy.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved shadow charges (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved dipolar energy.
        """
        if qdp is None:
            raise RuntimeError(
                "Dipole moments are required for dipolar energy calculation."
            )

        # tblite: coulomb/multipole.f90::get_energy
        vdp_sd = einsum("...ijx,...i->...jx", cache.amat_sd, qat)
        vdp_dd = 0.5 * einsum("...ijxy,...ix->...jy", cache.amat_dd, qdp)

        # tblite: coulomb/multipole.f90::get_kernel_energy
        # (ke = dk * dot(qdp, qdp))
        # Remember: cache.dkernel was unsqueezed in `get_cache`!
        ke = einsum("...ix,...ix,...ix->...i", cache.dkernel, qdp, qdp)

        return ke + einsum("...ix,...ix->...i", vdp_sd + vdp_dd, qdp)

    @override
    def get_quadrupole_atom_energy(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate atom-resolved dipolar energy.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved shadow charges (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved dipolar energy.
        """
        assert qqp is not None

        # tblite: coulomb/multipole.f90::get_energy
        vqp = einsum("...ijx,...i->...jx", cache.amat_sq, qat)

        # tblite: coulomb/multipole.f90::get_kernel_energy
        # (ke = dk * dot(qdp * scale, qdp))
        # Remember: cache.dkernel was unsqueezed in `get_cache`!
        scale = torch.tensor([1, 2, 1, 2, 2, 1], **self.dd)
        ke = einsum("...ix,...ix,x,...ix->...i", cache.qkernel, qqp, scale, qqp)

        return ke + einsum("...ix,...ix->...i", vqp, qqp)

    @override
    def get_monopole_atom_potential(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate atom-resolved potential.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved monopolar potential.
        """
        assert qdp is not None
        assert qqp is not None

        vat_sd = einsum("...ijx,...jx->...i", cache.amat_sd, qdp)
        vat_sq = einsum("...ijx,...jx->...i", cache.amat_sq, qqp)

        return vat_sd + vat_sq

    @override
    def get_dipole_atom_potential(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate atom-resolved dipolar potential.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved monopolar potential.
        """
        assert qdp is not None

        vdp_sd = einsum("...ijx,...i->...jx", cache.amat_sd, qat)
        vdp_dd = einsum("...ijxy,...ix->...jy", cache.amat_dd, qdp)

        # tblite: coulomb/multipole.f90::get_kernel_potential
        kernel_pot_dp = 2 * cache.dkernel * qdp

        return vdp_sd + vdp_dd + kernel_pot_dp

    @override
    def get_quadrupole_atom_potential(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        r"""
        Calculate atom-resolved quadrupolar potential.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved monopolar potential.
        """
        assert qqp is not None

        vqp = einsum("...ijx,...i->...jx", cache.amat_sq, qat)

        # tblite: coulomb/multipole.f90::get_kernel_potential
        scale = torch.tensor([1, 2, 1, 2, 2, 1], **self.dd)
        kernel_pot_dp = 2 * cache.qkernel * qqp * scale

        return vqp + kernel_pot_dp


def new_aes2(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> AES2 | None:
    """
    Create new instance of :class:`.AES2`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    AES2 | None
        Instance of the AES2 class or ``None`` if no AES2 is used.
    """
    if hasattr(par, "multipole") is False or par.multipole is None:
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

    unique = torch.unique(numbers)

    dkernel = get_elem_param(unique, par.element, "dkernel", **dd)
    qkernel = get_elem_param(unique, par.element, "qkernel", **dd)
    rad = get_elem_param(unique, par.element, "mprad", **dd)
    vcn = get_elem_param(unique, par.element, "mpvcn", **dd)

    dmp3 = torch.tensor(par.multipole.damped.dmp3, **dd)
    dmp5 = torch.tensor(par.multipole.damped.dmp5, **dd)
    kexp = torch.tensor(par.multipole.damped.kexp, **dd)
    shift = torch.tensor(par.multipole.damped.shift, **dd)
    rmax = torch.tensor(par.multipole.damped.rmax, **dd)

    return AES2(
        dmp3=dmp3,
        dmp5=dmp5,
        dkernel=dkernel,
        qkernel=qkernel,
        shift=shift,
        kexp=kexp,
        rmax=rmax,
        rad=rad,
        vcn=vcn,
        **dd,
    )
