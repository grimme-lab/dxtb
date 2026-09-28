# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Isotropic Electrostatics: Class
==============================

This module implements the isotropic electrostatics class. The
:class:`dxtb.components.IES` class is constructed similar to the
:class:`dxtb.components.Halogen` class.
"""

from __future__ import annotations

import torch
from tad_mctc.convert import any_to_tensor
from tad_mctc.ncoord import coordination_number, erf_count
from tad_multicharge.model.eeq import EEQModel

from dxtb import IndexHelper
from dxtb._src.typing import Any, Tensor, override

from ..base import Classical, ClassicalCache, ComponentCache

__all__ = ["IES", "LABEL_IES"]


LABEL_IES = "IES"
"""Label for the :class:`.IES` component, coinciding with the class name."""


class IESCache(ClassicalCache):
    """Coordinate-independent data for the EEQ solve."""

    numbers: Tensor
    """Atomic numbers, including batch padding."""

    eeq: EEQModel
    """ main electronegativity-equilibration model."""

    rcov: Tensor
    """Atom-resolved D3 covalent radii from tad-mctc."""

    __slots__ = ["numbers", "eeq", "rcov"]

    def __init__(
        self,
        numbers: Tensor,
        eeq: EEQModel,
        rcov: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is not None else rcov.device,
            dtype=dtype if dtype is not None else rcov.dtype,
        )
        self.numbers = numbers
        self.eeq = eeq
        self.rcov = rcov


class IES(Classical):
    """
    Representation of the isotropic electrostatics (IES) component in a tight-binding model.
    """

    chi: Tensor
    """Element-resolved electronegativity lookup table."""

    eeq_kcn: Tensor
    """Element-resolved coordination-number dependence."""

    eta: Tensor
    """Element-resolved chemical hardness."""

    rad: Tensor
    """Element-resolved Gaussian charge width."""

    rcov: Tensor
    """D3 covalent-radius lookup table from tad-mctc."""

    cutoff: Tensor
    """Coordination-number real-space cutoff."""

    cn_max: Tensor
    """Smooth coordination-number cap."""

    cn_kcn: Tensor
    """Steepness of the erf counting function."""

    __slots__ = [
        "chi",
        "eeq_kcn",
        "eta",
        "rad",
        "rcov",
        "cutoff",
        "cn_max",
        "cn_kcn",
    ]

    def __init__(
        self,
        chi: Tensor,
        eeq_kcn: Tensor,
        eta: Tensor,
        rad: Tensor,
        rcov: Tensor,
        cutoff: Tensor | float | int,
        cn_max: Tensor | float | int,
        cn_kcn: Tensor | float | int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.chi = chi.to(**self.dd)
        self.eeq_kcn = eeq_kcn.to(**self.dd)
        self.eta = eta.to(**self.dd)
        self.rad = rad.to(**self.dd)
        self.rcov = rcov.to(**self.dd)
        self.cutoff = any_to_tensor(cutoff, **self.dd)
        self.cn_max = any_to_tensor(cn_max, **self.dd)
        self.cn_kcn = any_to_tensor(cn_kcn, **self.dd)

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> IESCache:
        """
        Store variables that are independent of the atomic positions in a cache object.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of the system (shape: ``(natom,)``).
        ihelp : IndexHelper | None
            Helper class for indexing.

        Returns
        -------
        IESCache
            Cache object containing coordinate-independent data for the EEQ solve.
        """
        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars):
            if not isinstance(self.cache, IESCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        eeq = EEQModel(
            self.chi,
            self.eeq_kcn,
            self.eta,
            self.rad,
            **self.dd,
        )
        self.cache = IESCache(numbers, eeq, self.rcov[numbers], **self.dd)
        return self.cache

    @override
    def get_energy(
        self,
        positions: Tensor,
        cache: ComponentCache,
        charge: Tensor | float | int | None = None,
        **_: Any,
    ) -> Tensor:
        """
        Calculate the isotropic electrostatics energy using the EEQ model.

        Parameters
        ----------
        positions : Tensor
            Atomic positions of the system (shape: ``(natom, 3)``).
        cache : ComponentCache
            Cache object containing coordinate-independent data for the EEQ solve.
        charge : Tensor | float | int | None
            Total molecular charge. If None, the energy is not computed.

        Returns
        -------
        Tensor
            Isotropic electrostatics energy of the system (shape: ``()``).
        """
        if not isinstance(cache, IESCache):
            raise TypeError(f"Cache in {self.label} is not of type 'IESCache'.")
        if charge is None:
            raise ValueError("Total molecular charge is required for IES.")

        total_charge = any_to_tensor(
            charge,
            device=positions.device,
            dtype=positions.dtype,
        )
        cn = coordination_number(
            cache.numbers,
            positions,
            counting_function=erf_count,
            rcov=cache.rcov,
            cutoff=self.cutoff,
            cn_max=self.cn_max,
            kcn=self.cn_kcn,
        )
        _charges, energy = cache.eeq.solve(
            cache.numbers,
            positions,
            total_charge,
            cn,
            return_energy=True,
        )
        return energy
