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
Interactions: Self-consistent D4 Dispersion
===========================================

Self-consistent D4 dispersion correction.
"""

from __future__ import annotations

import tad_dftd4 as d4
import torch
from tad_mctc.exceptions import DeviceError
from tad_mctc.math import einsum

from dxtb import IndexHelper
from dxtb._src.param import Param
from dxtb._src.typing import (
    DD,
    Any,
    Slicers,
    Tensor,
    TensorLike,
    get_default_dtype,
    override,
)
from dxtb._src.utils import convert_float_tensor

from ..base import Interaction, InteractionCache

__all__ = ["DispersionD4SC", "LABEL_DISPERSIOND4SC", "new_d4sc"]


LABEL_DISPERSIOND4SC = "DispersionD4SC"
"""Label for the :class:`.DispersionD4SC` interaction, coinciding with the class name."""


class DispersionD4SCCache(InteractionCache, TensorLike):
    """
    Restart data for the :class:`.DispersionD4SC` interaction.

    Note
    ----
    The dispersion parameters (a1, a2, ...) are given in the dispersion
    class constructor.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    cn: Tensor
    """Coordination number of every atom."""

    dispmat: Tensor
    """
    Dispersion matrix. This quantity is almost equal to the dispersion energy,
    except for multiplication with C6 and C8.
    """

    __slots__ = ["__store", "cn", "dispmat"]

    def __init__(
        self,
        cn: Tensor,
        dispmat: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else cn.device,
            dtype=dtype if dtype is None else cn.dtype,
        )

        self.cn = cn
        self.dispmat = dispmat

        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        cn: Tensor
        """Coordination number of every atom."""

        dispmat: Tensor
        """
        Dispersion matrix. This quantity is almost equal to the dispersion
        energy, except for multiplication with C6 and C8.
        """

        def __init__(self, cn: Tensor, dispmat: Tensor) -> None:
            self.cn = cn
            self.dispmat = dispmat

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(self.cn, self.dispmat)

        slicer = slicers["atom"]
        self.cn = self.cn[[~conv, *slicer]]
        self.dispmat = self.dispmat[[~conv, *slicer, *slicer]]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.cn = self.__store.cn
        self.dispmat = self.__store.dispmat


class DispersionD4SC(Interaction):
    """
    Self-consistent D4 dispersion correction (:class:`.DispersionD4SC`).
    """

    param: dict[str, Tensor]
    """Dispersion parameters."""

    model: d4.model.D4Model
    """Model for the D4 dispersion correction."""

    rcov: Tensor
    """Covalent radii of all atoms."""

    r4r2: Tensor
    """R4/R2 ratio of all atoms."""

    cutoff: d4.cutoff.Cutoff
    """Real-space cutoff for the D4 dispersion correction."""

    counting_function: d4.typing.CountingFunction
    """
    Counting function for the coordination number.

    :default: :func:`d4.ncoord.erf_count`
    """

    damping_function: d4.typing.DampingFunction
    """
    Damping function for the dispersion correction.

    :default: :func:`d4.damping.rational_damping`
    """

    __slots__ = [
        "param",
        "model",
        "rcov",
        "r4r2",
        "cutoff",
        "counting_function",
        "damping_function",
    ]

    def __init__(
        self,
        param: dict[str, Tensor],
        model: d4.model.D4Model,
        rcov: Tensor,
        r4r2: Tensor,
        cutoff: d4.cutoff.Cutoff,
        counting_function: d4.typing.CountingFunction = d4.ncoord.erf_count,
        damping_function: d4.typing.DampingFunction = d4.damping.rational_damping,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.param = param
        self.model = model
        self.rcov = rcov
        self.r4r2 = r4r2
        self.cutoff = cutoff
        self.counting_function = counting_function
        self.damping_function = damping_function

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
        **_,
    ) -> DispersionD4SCCache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        DispersionD4SCCache
            Restart data for the interaction.

        Note
        ----
        If the :class:`.DispersionD4SC` interaction is evaluated within the
        :class:`dxtb.components.InteractionList`, ``positions`` will be passed
        as an argument, too. Hence, it is necessary to absorb the ``positions``
        in the signature of the function (also see
        :meth:`dxtb.components.Interaction.get_cache`).
        """
        if numbers is None:
            raise ValueError(
                "Atomic numbers are required for DispersionD4SC cache."
            )
        if positions is None:
            raise ValueError("Positions are required for ES2 cache.")

        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, DispersionD4SCCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        # if the cache is built, store the cachevar for validation
        self._cachevars = cachvars

        cn = d4.ncoord.cn_d4(
            numbers,
            positions,
            counting_function=self.counting_function,
            rcov=self.rcov,
            cutoff=self.cutoff.cn,
        )

        # tblite: disp/d4.f90::get_dispersion_matrix
        # Instead of multiplying with C6 (from `get_atomic_c6`), we multiply
        # with the reference C6 coefficients that have not been multiplied with
        # the Gaussian weights yet. Correspondingly, we have to set the C6
        # argument of `dispersion2` to 1.
        edisp = d4.disp.dispersion2(
            numbers,
            positions,
            self.param,
            torch.ones((*numbers.shape, numbers.shape[-1]), **self.dd),
            self.r4r2,
            as_matrix=True,
        )
        dispmat = edisp.unsqueeze(-1).unsqueeze(-1) * self.model.rc6

        self.cache = DispersionD4SCCache(cn, dispmat)

        return self.cache

    @override
    def get_monopole_atom_energy(
        self, cache: DispersionD4SCCache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the D4 dispersion correction energy.

        Parameters
        ----------
        cache : DispersionD4SCCache
            Restart data for the interaction.
        qat : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atomwise D4 dispersion correction energies.
        """
        weights = self.model.weight_references(cache.cn, qat)

        return 0.5 * einsum(
            "...ijab,...ia,...jb->...j",
            *(cache.dispmat, weights, weights),
            optimize=[(0, 1), (0, 1)],
        )

    @override
    def get_monopole_atom_potential(
        self, cache: DispersionD4SCCache, qat: Tensor, *_: Any, **__: Any
    ) -> Tensor:
        """
        Calculate the D4 dispersion correction potential.

        Parameters
        ----------
        cache : DispersionD4SCCache
            Restart data for the interaction.
        qat : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atomwise dispersion correction potential.
        """
        weights, dgwdq = self.model.weight_references(
            cache.cn, qat, with_dgwdq=True
        )

        return einsum(
            "...ijab,...jb,...ia->...i", cache.dispmat, weights, dgwdq
        )


def new_d4sc(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> DispersionD4SC | None:
    """
    Create new instance of :class:`.DispersionD4SC`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    DispersionD4SC | None
        Instance of the :class:`.DispersionD4SC` class or ``None`` if no :class:`.DispersionD4SC` is
        used.
    """
    if hasattr(par, "dispersion") is False or par.dispersion is None:
        return None

    if par.dispersion.d4 is None:
        return None

    if par.dispersion.d4.sc is False:
        return None

    if device is not None:
        if device != numbers.device:
            raise DeviceError(
                f"Passed device ({device}) and device of `numbers` tensor "
                f"({numbers.device}) do not match."
            )

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    param = convert_float_tensor(
        {
            "a1": par.dispersion.d4.a1,
            "a2": par.dispersion.d4.a2,
            "s6": par.dispersion.d4.s6,
            "s8": par.dispersion.d4.s8,
            "s9": par.dispersion.d4.s9,
            "s10": par.dispersion.d4.s10,
        },
        **dd,
    )

    rcov = d4.data.COV_D3.to(**dd)[numbers]
    r4r2 = d4.data.R4R2.to(**dd)[numbers]
    model = d4.model.D4Model(numbers, ref_charges="gfn2", **dd)
    cutoff = d4.cutoff.Cutoff(disp2=50.0, disp3=25.0, **dd)

    return DispersionD4SC(
        param, model=model, rcov=rcov, r4r2=r4r2, cutoff=cutoff, **dd
    )
