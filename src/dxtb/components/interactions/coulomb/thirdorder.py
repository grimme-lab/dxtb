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
On-site third-order electrostatic energy (ES3)
==============================================

This module implements the third-order electrostatic energy for GFN1-xTB.

Example
-------
>>> import torch
>>> import dxtb.coulomb.thirdorder as es3
>>> from dxtb.param import GFN1_XTB, get_element_param
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [0.00000000000000, -0.00000000000000, 0.00000000000000],
...     [1.61768389755830, 1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [1.61768389755830, -1.61768389755830, 1.61768389755830],
...     [-1.61768389755830, 1.61768389755830, 1.61768389755830],
... ])
>>> qat = torch.tensor([
...     -8.41282505804719e-2,
...     2.10320626451180e-2,
...     2.10320626451178e-2,
...     2.10320626451179e-2,
...     2.10320626451179e-2,
... ])
>>> hubbard_derivs = get_element_param(GFN1_XTB.element, "gam3")
>>> ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
>>> es = es3.ES3(positions, hubbard_derivs)
>>> cache = es.get_cache(ihelp)
>>> e = es.get_atom_energy(qat, cache)
>>> torch.set_printoptions(precision=7)
>>> print(torch.sum(e, dim=-1))
tensor(0.0155669)
"""

from __future__ import annotations

import torch
from tad_mctc.exceptions import DeviceError

from dxtb.basis import IndexHelper
from dxtb.param import Param, get_elem_param
from dxtb.typing import DD, Slicers, Tensor, TensorLike, get_default_dtype

from .. import Interaction

__all__ = ["ES3", "LABEL_ES3", "new_es3"]


LABEL_ES3 = "ES3"
"""Label for the 'ES3' interaction, coinciding with the class name."""


class ES3(Interaction):
    """
    On-site third-order electrostatic energy.
    """

    hubbard_derivs: Tensor
    """Hubbard derivatives of all atoms."""

    __slots__ = ["hubbard_derivs"]

    class Cache(Interaction.Cache, TensorLike):
        """
        Restart data for the ES3 interaction.
        """

        __store: Store | None
        """Storage for cache (required for culling)."""

        hd: Tensor
        """Spread Hubbard derivatives of all atoms (not only unique)."""

        __slots__ = ["__store", "hd"]

        def __init__(
            self,
            hd: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            super().__init__(
                device=device if device is None else hd.device,
                dtype=dtype if dtype is None else hd.dtype,
            )
            self.hd = hd
            self.__store = None

        class Store:
            """
            Storage container for cache containing `__slots__` before culling.
            """

            hd: Tensor
            """Spread Hubbard derivatives of all atoms (not only unique)."""

            def __init__(self, hd: Tensor) -> None:
                self.hd = hd

        def cull(self, conv: Tensor, slicers: Slicers) -> None:
            if self.__store is None:
                self.__store = self.Store(self.hd)

            slicer = slicers["atom"]
            self.hd = self.hd[[~conv, *slicer]]

        def restore(self) -> None:
            if self.__store is None:
                raise RuntimeError("Nothing to restore. Store is empty.")

            self.hd = self.__store.hd

    def __init__(
        self,
        hubbard_derivs: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.hubbard_derivs = hubbard_derivs

    def get_cache(self, ihelp: IndexHelper, **_) -> Cache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Interaction.Cache
            Restart data for the interaction.

        Note
        ----
        If this `ES3` interaction is evaluated within the `InteractionList`,
        `numbers` and `positions` will be passed as argument, too. The `**_` in
        the argument list will absorb those unnecessary arguments which are
        given as keyword-only arguments (see `Interaction.get_cache()`).
        """

        return self.Cache(ihelp.spread_uspecies_to_atom(self.hubbard_derivs))

    def get_atom_energy(self, charges: Tensor, cache: Cache) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Implements Eq.30 of the following paper:
        - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht,
        J. Seibert, S. Spicher and S. Grimme, *WIREs Computational Molecular
        Science*, **2020**, 11, e1493. DOI: `10.1002/wcms.1493
        <https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493>`__

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atomwise third-order Coulomb interaction energies.
        """

        return cache.hd * torch.pow(charges, 3.0) / 3.0

    def get_atom_potential(self, charges: Tensor, cache: Cache) -> Tensor:
        """Calculate the third-order electrostatic potential.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atomwise third-order Coulomb interaction potential.
        """

        return cache.hd * torch.pow(charges, 2.0)


def new_es3(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ES3 | None:
    """
    Create new instance of ES3.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    ES3 | None
        Instance of the ES3 class or `None` if no ES3 is used.
    """

    if hasattr(par, "thirdorder") is False or par.thirdorder is None:
        return None

    if par.thirdorder.shell is True:
        raise NotImplementedError(
            "Shell-resolved third order electrostatics are not implemented. "
            "Set `thirdorder.shell` parameter to `False`."
        )

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

    hubbard_derivs = get_elem_param(torch.unique(numbers), par.element, "gam3", **dd)

    return ES3(hubbard_derivs, **dd)
