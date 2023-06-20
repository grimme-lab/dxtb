"""
DFT-D3 dispersion model.
"""
from __future__ import annotations

import tad_dftd3 as d3
import torch

from .._types import Any, CountingFunction, Tensor, TensorLike
from ..ncoord import exp_count, get_coordination_number
from .base import Dispersion


class DispersionD3(Dispersion):
    """Representation of the DFT-D3(BJ) dispersion correction."""

    class Cache(TensorLike):
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
            weighting_function: d3.WeightingFunction,
            damping_function: d3.DampingFunction,
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

    def get_cache(self, numbers: Tensor, **kwargs: Any) -> DispersionD3.Cache:
        """
        Obtain cache for storage of settings.

        Settings can be passed as `kwargs`. The available optional parameters
        are the same as in `tad_dftd3.dftd3`, i.e., "ref", "rcov", "rvdw", and
        "r4r2".

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms.

        Returns
        -------
        DispersionD3.Cache
            Cache for the D3 dispersion.
        """

        ref = (
            kwargs.pop(
                "ref",
                d3.reference.Reference(),
            )
            .type(self.dtype)
            .to(self.device)
        )
        rcov = (
            kwargs.pop(
                "rcov",
                d3.data.covalent_rad_d3[numbers],
            )
            .type(self.dtype)
            .to(self.device)
        )
        rvdw = (
            kwargs.pop(
                "rvdw",
                d3.data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)],
            )
            .type(self.dtype)
            .to(self.device)
        )
        r4r2 = (
            kwargs.pop(
                "r4r2",
                d3.data.sqrt_z_r4_over_r2[numbers],
            )
            .type(self.dtype)
            .to(self.device)
        )
        cf = kwargs.pop("counting_function", exp_count)
        wf = kwargs.pop("weighting_function", d3.model.gaussian_weight)
        df = kwargs.pop("damping_function", d3.disp.rational_damping)

        return self.Cache(ref, rcov, rvdw, r4r2, cf, wf, df)

    def get_energy(self, positions: Tensor, cache: DispersionD3.Cache) -> Tensor:
        """
        Get D3 dispersion energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms.
        cache : DispersionD3.Cache
            Dispersion cache containing settings.

        Returns
        -------
        Tensor
            Atom-resolved D3 dispersion energy.
        """

        cn = get_coordination_number(
            self.numbers, positions, cache.counting_function, cache.rcov
        )
        weights = d3.model.weight_references(
            self.numbers, cn, cache.ref, cache.weighting_function
        )
        c6 = d3.model.atomic_c6(self.numbers, weights, cache.ref)

        return d3.disp.dispersion(
            self.numbers,
            positions,
            self.param,
            c6,
            cache.rvdw,
            cache.r4r2,
            cache.damping_function,
        )
