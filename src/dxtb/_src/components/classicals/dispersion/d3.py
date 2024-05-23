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
Dispersion: D3
==============

The DFT-D3 dispersion model.
"""
from __future__ import annotations

import tad_dftd3 as d3
import torch
from tad_mctc.ncoord import cn_d3, exp_count

from dxtb._src.typing import Any, CountingFunction, Tensor

from .base import ClassicalCache, Dispersion

__all__ = ["DispersionD3", "DispersionD3Cache"]


class DispersionD3Cache(ClassicalCache):
    """
    Cache for the dispersion settings.

    Note
    ----
    The dispersion parameters (a1, a2, ...) are given in the constructor.
    """

    __slots__ = [
        "ref",
        "rcov",
        "rvdw",
        "r4r2",
        "counting_function",
        "weighting_function",
        "damping_function",
    ]

    def __init__(
        self,
        ref: d3.reference.Reference,
        rcov: Tensor,
        rvdw: Tensor,
        r4r2: Tensor,
        counting_function: CountingFunction,
        weighting_function: d3.typing.WeightingFunction,
        damping_function: d3.typing.DampingFunction,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else rcov.device,
            dtype=dtype if dtype is None else rcov.dtype,
        )
        self.ref = ref
        self.rcov = rcov
        self.rvdw = rvdw
        self.r4r2 = r4r2
        self.counting_function = counting_function
        self.weighting_function = weighting_function
        self.damping_function = damping_function


class DispersionD3(Dispersion):
    """Representation of the DFT-D3(BJ) dispersion correction."""

    def get_cache(self, numbers: Tensor, **kwargs: Any) -> DispersionD3Cache:
        """
        Obtain cache for storage of settings.

        Settings can be passed as ``kwargs``. The available optional parameters
        are the same as in :func:`tad_dftd3.dftd3`, i.e., "ref", "rcov",
        "rvdw", and "r4r2".

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).

        Returns
        -------
        DispersionD3Cache
            Cache for the D3 dispersion.
        """
        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, DispersionD3Cache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        ref = kwargs.pop(
            "ref",
            d3.reference.Reference(),
        ).to(**self.dd)

        rcov = kwargs.pop(
            "rcov",
            d3.data.COV_D3.to(**self.dd)[numbers],
        ).to(**self.dd)

        rvdw = kwargs.pop(
            "rvdw",
            d3.data.VDW_D3.to(**self.dd)[numbers.unsqueeze(-1), numbers.unsqueeze(-2)],
        ).to(**self.dd)

        r4r2 = kwargs.pop(
            "r4r2",
            d3.data.R4R2.to(**self.dd)[numbers],
        ).to(**self.dd)

        cf = kwargs.pop("counting_function", exp_count)
        wf = kwargs.pop("weighting_function", d3.model.gaussian_weight)
        df = kwargs.pop("damping_function", d3.damping.rational_damping)

        self.cache = DispersionD3Cache(ref, rcov, rvdw, r4r2, cf, wf, df)
        return self.cache

    def get_energy(
        self, positions: Tensor, cache: DispersionD3Cache, **kwargs: Any
    ) -> Tensor:
        """
        Get D3 dispersion energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : DispersionD3Cache
            Dispersion cache containing settings.

        Returns
        -------
        Tensor
            Atom-resolved D3 dispersion energy.
        """

        cn = cn_d3(
            self.numbers,
            positions,
            counting_function=cache.counting_function,
            rcov=cache.rcov,
        )
        weights = d3.model.weight_references(
            self.numbers, cn, cache.ref, cache.weighting_function
        )

        chunk_size = kwargs.pop("chunk_size", None)
        c6 = d3.model.atomic_c6(self.numbers, weights, cache.ref, chunk_size=chunk_size)

        return d3.disp.dispersion(
            self.numbers,
            positions,
            self.param,
            c6,
            cache.rvdw,
            cache.r4r2,
            cache.damping_function,
        )
