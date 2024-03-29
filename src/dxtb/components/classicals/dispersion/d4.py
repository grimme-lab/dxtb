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
DFT-D4 dispersion model.
"""

from __future__ import annotations

import tad_dftd4 as d4
from tad_mctc.typing import Any, Tensor

from .base import Dispersion


class DispersionD4(Dispersion):
    """Representation of the DFT-D4 dispersion correction."""

    charge: Tensor
    """Total charge of the system."""

    class Cache:
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
            counting_function: d4.typing.CountingFunction,
            damping_function: d4.typing.DampingFunction,
        ) -> None:
            self.q = q
            self.model = model
            self.rcov = rcov
            self.r4r2 = r4r2
            self.cutoff = cutoff
            self.counting_function = counting_function
            self.damping_function = damping_function

    def get_cache(self, numbers: Tensor, **kwargs: Any) -> DispersionD4.Cache:
        """
        Obtain cache for storage of settings.

        Settings can be passed as `kwargs`. The available optional parameters
        are the same as in `tad_dftd4.dftd4`, i.e., "model", "rcov", "r4r2",
        "cutoff", "counting_function", and "damping_function". Only the charges

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        charge : Tensor
            Total charge of the system.

        Returns
        -------
        DispersionD4.Cache
            Cache for the D4 dispersion.
        """

        model: d4.model.D4Model = (
            kwargs.pop(
                "model",
                d4.model.D4Model(numbers, device=self.device, dtype=self.dtype),
            )
            .type(self.dtype)
            .to(self.device)
        )
        rcov: Tensor = (
            kwargs.pop(
                "rcov",
                d4.data.COV_D3[numbers],
            )
            .type(self.dtype)
            .to(self.device)
        )
        q = kwargs.pop("q", None)
        r4r2: Tensor = (
            kwargs.pop(
                "r4r2",
                d4.data.R4R2[numbers],
            )
            .type(self.dtype)
            .to(self.device)
        )
        cutoff: d4.cutoff.Cutoff = (
            (
                kwargs.pop(
                    "cutoff", d4.cutoff.Cutoff(device=self.device, dtype=self.dtype)
                )
            )
            .type(self.dtype)
            .to(self.device)
        )
        cf = kwargs.pop("counting_function", d4.ncoord.erf_count)
        df = kwargs.pop("damping_function", d4.damping.rational_damping)

        return self.Cache(q, model, rcov, r4r2, cutoff, cf, df)

    def get_energy(
        self, positions: Tensor, cache: DispersionD4.Cache, q: Tensor | None = None
    ) -> Tensor:
        """
        Get D4 dispersion energy.ci

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        cache : DispersionD4.Cache
            Dispersion cache containing settings.
        q : Tensor | None, optional
            Atomic partial charges. Defaults to `None` (EEQ charges).

        Returns
        -------
        Tensor
            Atom-resolved D4 dispersion energy.
        """

        return d4.dftd4(
            self.numbers,
            positions,
            self.charge,
            self.param,
            model=cache.model,
            rcov=cache.rcov,
            r4r2=cache.r4r2,
            q=cache.q if q is None else q,
            cutoff=cache.cutoff,
            counting_function=cache.counting_function,
            damping_function=cache.damping_function,
        )
