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
Dispersion: D4
==============

DFT-D4 dispersion model.
"""

from __future__ import annotations

from typing import Any

import tad_dftd4 as d4
import torch
from tad_mctc.data import radii
from tad_mctc.ncoord import erf_count
from tad_mctc.typing import CountingFunction, Tensor, override

from dxtb import IndexHelper

from ..base import ClassicalCache
from .base import Dispersion

__all__ = ["DispersionD4", "DispersionD4Cache"]


class DispersionD4Cache(ClassicalCache):
    """
    Cache for the dispersion settings.

    Note
    ----
    The dispersion parameters (a1, a2, ...) are given in the dispersion
    class constructor.
    """

    __slots__ = [
        "q",
        "model",
        "rcov",
        "r4r2",
        "cutoff",
        "counting_function",
        "damping_function",
    ]

    def __init__(
        self,
        q: Tensor | None,
        model: d4.model.D4Model,
        rcov: Tensor,
        r4r2: Tensor,
        cutoff: d4.cutoff.Cutoff,
        counting_function: CountingFunction,
        damping_function: d4.damping.Damping,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.q = q
        self.model = model
        self.rcov = rcov
        self.r4r2 = r4r2
        self.cutoff = cutoff
        self.counting_function = counting_function
        self.damping_function = damping_function


class DispersionD4(Dispersion):
    """
    Representation of the DFT-D4 dispersion correction (:class:`.DispersionD4`).
    """

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **kwargs: Any
    ) -> DispersionD4Cache:
        """
        Obtain cache for storage of settings.

        Settings can be passed as `kwargs`. The available optional parameters
        are the same as in `tad_dftd4.dftd4`, i.e., "model", "rcov", "r4r2",
        "cutoff", "counting_function", and "damping_function". Only the charges

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        charge : Tensor
            Total charge of the system.

        Returns
        -------
        DispersionD4Cache
            Cache for the D4 dispersion.
        """
        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, DispersionD4Cache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        model = kwargs.pop("model", None)
        if model is not None and not isinstance(model, d4.model.D4Model):
            raise TypeError("D4: Model is not of type 'd4.model.D4Model'.")
        if model is None:
            model = d4.model.D4Model(
                numbers, ref_charges=self.ref_charges, **self.dd
            )
        else:
            model = model.type(self.dtype).to(self.device)

        rcov = kwargs.pop("rcov", None)
        if rcov is not None and not isinstance(rcov, Tensor):
            raise TypeError("D4: 'rcov' is not of type 'Tensor'.")
        if rcov is None:
            rcov = radii.COV_D3(**self.dd)[numbers]
        else:
            rcov = rcov.to(**self.dd)

        r4r2 = kwargs.pop("r4r2", None)
        if r4r2 is not None and not isinstance(r4r2, Tensor):
            raise TypeError("D4: 'r4r2' is not of type 'Tensor'.")
        if r4r2 is None:
            r4r2 = d4.data.R4R2(**self.dd)[numbers]
        else:
            r4r2 = r4r2.to(**self.dd)

        cutoff = kwargs.pop("cutoff", None)
        if cutoff is not None and not isinstance(cutoff, d4.Cutoff):
            raise TypeError("D4: 'cutoff' is not of type 'd4.Cutoff'.")
        if cutoff is None:
            cutoff = d4.Cutoff(**self.dd)
        else:
            cutoff = cutoff.type(self.dtype).to(self.device)

        q = kwargs.pop("q", None)

        cf = kwargs.pop("counting_function", erf_count)
        df = kwargs.pop("damping_function", d4.damping.RationalDamping())

        self.cache = DispersionD4Cache(q, model, rcov, r4r2, cutoff, cf, df)
        return self.cache

    @override
    def get_energy(
        self,
        positions: Tensor,
        cache: DispersionD4Cache,
        q: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Get D4 dispersion energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : DispersionD4Cache
            Dispersion cache containing settings.
        q : Tensor | None, optional
            Atomic partial charges. Defaults to ``None`` (EEQ charges).

        Returns
        -------
        Tensor
            Atom-resolved D4 dispersion energy.
        """
        # FIXME: Charge should be REQUIRED for D4!
        if self.charge is None and "charge" not in kwargs:
            charge = torch.tensor(0.0, **self.dd)
        else:
            charge = kwargs.pop("charge", self.charge)

        return d4.dftd4(
            self.numbers,
            positions,
            charge,
            self.param,
            model=cache.model,
            rcov=cache.rcov,
            r4r2=cache.r4r2,
            q=cache.q if q is None else q,
            cutoff=cache.cutoff,
            counting_function=cache.counting_function,
            damping_function=cache.damping_function,
        )
