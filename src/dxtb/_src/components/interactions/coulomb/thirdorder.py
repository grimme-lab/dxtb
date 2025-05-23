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
Coulomb: On-site third-order electrostatic energy (ES3)
=======================================================

This module implements the third-order electrostatic energy for GFN1-xTB.

Example
-------

.. code-block:: python

    import torch
    import dxtb.coulomb.thirdorder as es3
    from dxtb import GFN1_XTB, get_element_param
    from dxtb import IndexHelper

    # Define atomic numbers and their positions
    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [+0.00000000000000, -0.00000000000000, +0.00000000000000],
        [+1.61768389755830, +1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [+1.61768389755830, -1.61768389755830, +1.61768389755830],
        [-1.61768389755830, +1.61768389755830, +1.61768389755830],
    ])

    # Atomic charges
    qat = torch.tensor([
        -8.41282505804719e-2,
        2.10320626451180e-2,
        2.10320626451178e-2,
        2.10320626451179e-2,
        2.10320626451179e-2,
    ])

    # Initialize the ES3 calculation class with Hubbard derivatives parameter
    hubbard_derivs = get_element_param(GFN1_XTB.element, "gam3")
    es = es3.ES3(positions, hubbard_derivs)

    # Create an index helper from atomic numbers
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    # Generate the cache and carry out the energy calculation
    cache = es.get_cache(ihelp)
    e = es.get_atom_energy(qat, cache)

    # Print the summed energy
    torch.set_printoptions(precision=7)
    print(torch.sum(e, dim=-1))  # tensor(0.0155669)
"""

from __future__ import annotations

import torch
from tad_mctc.exceptions import DeviceError

from dxtb import IndexHelper
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import (
    DD,
    Any,
    Slicers,
    Tensor,
    TensorLike,
    get_default_dtype,
    override,
)

from ..base import Interaction, InteractionCache

__all__ = ["ES3", "LABEL_ES3", "new_es3"]


LABEL_ES3 = "ES3"
"""Label for the :class:`.ES3` interaction, coinciding with the class name."""


class ES3Cache(InteractionCache, TensorLike):
    """
    Restart data for the :class:`.ES3` interaction.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    hd: Tensor
    """Spread Hubbard derivatives of all atoms (not only unique)."""

    shell_resolved: bool
    """Whether the third-order electrostatics are shell-resolved."""

    __slots__ = ["__store", "hd", "shell_resolved"]

    def __init__(
        self,
        hd: Tensor,
        shell_resolved: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            device=device if device is None else hd.device,
            dtype=dtype if dtype is None else hd.dtype,
        )
        self.hd = hd
        self.shell_resolved = shell_resolved
        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        hd: Tensor
        """Spread Hubbard derivatives of all atoms (not only unique)."""

        def __init__(self, hd: Tensor) -> None:
            self.hd = hd

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(self.hd)

        slicer = slicers["shell"] if self.shell_resolved else slicers["atom"]
        self.hd = self.hd[[~conv, *slicer]]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.hd = self.__store.hd


class ES3(Interaction):
    """
    On-site third-order electrostatic energy (:class:`.ES3`).
    """

    hubbard_derivs: Tensor
    """Hubbard derivatives of all atoms."""

    shell_scale: Tensor | None
    """
    Scaling factors for shell-resolved third-order electrostatics.

    In GFN2-xTB, this is a tensor of shape ``(3,)`` containing the scaling
    factors for the s, p, and d shells.

    :default: ``None``
    """

    __slots__ = ["hubbard_derivs", "shell_scale"]

    def __init__(
        self,
        hubbard_derivs: Tensor,
        shell_scale: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.hubbard_derivs = hubbard_derivs
        self.shell_scale = shell_scale

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> ES3Cache:
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
        ES3Cache
            Restart data for the interaction.

        Note
        ----
        If the :class:`.ES3` interaction is evaluated within the
        :class:`dxtb.components. InteractionList`, ``positions`` will be
        passed as an argument, too. Hence, it is necessary to absorb
        the ``positions`` in the signature of the function (also see
        :meth:`dxtb.components.Interaction.get_cache`).
        """
        if numbers is None:
            raise ValueError("Atomic numbers are required for ES3 cache.")
        if ihelp is None:
            raise ValueError("IndexHelper is required for ES3 cache.")

        cachvars = (numbers.detach().clone(),)

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, ES3Cache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        # if the cache is built, store the cachevar for validation
        self._cachevars = cachvars

        if self.shell_scale is None:
            hd = ihelp.spread_uspecies_to_atom(self.hubbard_derivs)
        else:
            scale = ihelp.spread_ushell_to_shell(
                self.shell_scale[ihelp.unique_angular]
            )
            hd = ihelp.spread_uspecies_to_shell(self.hubbard_derivs) * scale

        self.cache = ES3Cache(
            hd, shell_resolved=(self.shell_scale is not None), **self.dd
        )

        return self.cache

    @override
    def get_monopole_atom_energy(
        self, cache: ES3Cache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Implements Eq.30 of the following paper:

        - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht,
          J. Seibert, S. Spicher and S. Grimme, *WIREs Computational Molecular
          Science*, **2020**, 11, e1493. DOI: `10.1002/wcms.1493
          <https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493>`__

        Parameters
        ----------
        cache : ES3Cache
            Restart data for the interaction.
        charges : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atom-wise third-order Coulomb interaction energies.
        """
        return (
            cache.hd * torch.pow(qat, 3.0) / 3.0
            if self.shell_scale is None
            else torch.zeros_like(qat)
        )

    @override
    def get_monopole_shell_energy(
        self, cache: ES3Cache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Parameters
        ----------
        cache : ES3Cache
            Restart data for the interaction.
        qat : Tensor
            Shell charges of all atoms.

        Returns
        -------
        Tensor
            Shell-wise third-order Coulomb interaction energy.
        """
        return (
            torch.zeros_like(qat)
            if self.shell_scale is None
            else cache.hd * torch.pow(qat, 3.0) / 3.0
        )

    @override
    def get_monopole_atom_potential(
        self,
        cache: ES3Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate the third-order electrostatic potential.
        Zero if this interaction is shell-resolved.

        Parameters
        ----------
        qat : ES3Cache
            Restart data for the interaction.
        charges : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atom-wise third-order Coulomb interaction potential.
        """
        return (
            cache.hd * torch.pow(qat, 2.0)
            if self.shell_scale is None
            else torch.zeros_like(qat)
        )

    @override
    def get_monopole_shell_potential(
        self, cache: ES3Cache, qsh: Tensor, *_: Any, **__: Any
    ) -> Tensor:
        """
        Calculate the third-order electrostatic potential.
        Zero if this interaction is atom-resolved.

        Parameters
        ----------
        qsh : Tensor
            Shell charges of all atoms.
        cache : ES3Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Shell-wise third-order Coulomb interaction potential.
        """
        return (
            torch.zeros_like(qsh)
            if self.shell_scale is None
            else cache.hd * torch.pow(qsh, 2.0)
        )


def new_es3(
    unique: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ES3 | None:
    """
    Create new instance of :class:`.ES3`.

    Parameters
    ----------
    unique : Tensor
        Unique elements in the system (shape: ``(nunique,)``).
    par : Param | ParamModule
        Representation of an extended tight-binding model.

    Returns
    -------
    ES3 | None
        Instance of the :class:`.ES3` class or ``None`` if no :class:`.ES3` is
        used.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    # compatibility with previous version based on `Param`
    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "thirdorder" not in par or par.is_none("thirdorder"):
        return None

    if device is not None:
        if device != unique.device:
            raise DeviceError(
                f"Passed device ({device}) and device of `unique` tensor "
                f"({unique.device}) do not match."
            )

    hubbard_derivs = par.get_elem_param(unique, "gam3")

    shell_scale = (
        None
        if par.is_false("thirdorder", "shell")
        else torch.cat(
            [
                torch.atleast_1d(par.get("thirdorder.shell.s")),
                torch.atleast_1d(par.get("thirdorder.shell.p")),
                torch.atleast_1d(par.get("thirdorder.shell.d")),
            ],
            dim=0,
        )
    )

    return ES3(hubbard_derivs, shell_scale=shell_scale, **dd)
