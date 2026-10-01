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
r"""
Spin Polarization
=================

On-site, shell-resolved interaction of collinear spin-polarized
tight-binding (spGFN).

The spin-polarization energy is given in terms of the shell-resolved
magnetization charges :math:`m_{l}` of each atom

.. math::

    E_\text{spin} = \dfrac{1}{2} \sum_A \sum_{l, l' \in A}
    m_{l} W_{ll'}^A m_{l'}

with the element-specific spin constants :math:`W_{ll'}^A`. The interaction
only acts on the magnetization channel and requires an unrestricted (UHF)
calculation with two spin channels.

Example
-------

.. code-block:: python

    import torch
    from dxtb import GFN1_XTB, Calculator
    from dxtb.components.spin import new_spin_polarization

    numbers = torch.tensor([8, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]])

    spin = new_spin_polarization(numbers, dtype=positions.dtype)
    calc = Calculator(numbers, GFN1_XTB, interaction=[spin])
    result = calc.singlepoint(positions, spin=1)
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb import IndexHelper
from dxtb._src.typing import Any, Slicers, Tensor, override

from ..base import ChargeChannel, Interaction, InteractionCache
from .constants import ANGULAR_PAIR_INDEX

__all__ = [
    "LABEL_SPIN_POLARIZATION",
    "SpinPolarization",
    "SpinPolarizationCache",
]


LABEL_SPIN_POLARIZATION = "SpinPolarization"
"""Label for the 'SpinPolarization' interaction, coinciding with the class name."""


class SpinPolarizationCache(InteractionCache):
    """
    Restart data for the spin-polarization interaction.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    wll: Tensor
    """Atom-local shell-pair spin matrix (shape: ``(..., nsh, nsh)``)."""

    __slots__ = ["__store", "wll"]

    def __init__(
        self,
        wll: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=wll.device if device is None else device,
            dtype=wll.dtype if dtype is None else dtype,
        )
        self.wll = wll
        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        wll: Tensor
        """Atom-local shell-pair spin matrix."""

        def __init__(self, wll: Tensor) -> None:
            self.wll = wll

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(self.wll)

        slicer = slicers["shell"]
        self.wll = self.wll[[~conv, *slicer, *slicer]]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Cache cannot be restored; store is empty.")

        self.wll = self.__store.wll


class SpinPolarization(Interaction):
    """
    On-site, shell-resolved spin-polarization interaction.

    Note
    ----
    The interaction acts on the magnetization channel and adding it to a
    calculator automatically enables two spin channels (UHF).
    """

    charge_channel = ChargeChannel.MAGNETIZATION
    requires_uhf = True

    spin_constants: Tensor
    """
    Spin constants (ss, sp, pp, sd, pd, dd) indexed by atomic number
    (shape: ``(MAX_SPIN_ELEMENT + 1, 6)``).
    """

    wscale: Tensor
    """Scaling factor for all spin constants."""

    __slots__ = ["spin_constants", "wscale"]

    def __init__(
        self,
        spin_constants: Tensor,
        wscale: Tensor | float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=spin_constants.device if device is None else device,
            dtype=spin_constants.dtype if dtype is None else dtype,
        )
        self.spin_constants = spin_constants.to(**self.dd)
        self.wscale = torch.as_tensor(wscale, **self.dd)

    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> SpinPolarizationCache:
        """
        Create restart data for the spin-polarization interaction, i.e., the
        atom-local shell-pair matrix of spin constants.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor | None, optional
            Not required for the on-site interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        SpinPolarizationCache
            Restart data for the interaction.
        """
        if numbers is None:
            raise ValueError("Atomic numbers are required for the cache.")
        if ihelp is None:
            raise ValueError("IndexHelper is required for the cache.")

        cachevars = (
            numbers.detach().clone(),
            ihelp.angular.detach().clone(),
            self.spin_constants.detach().clone(),
            self.wscale.detach().clone(),
        )

        # A cached `W` that depends on differentiable parameters would be
        # part of a graph that may already be freed by a backward pass.
        differentiable = self.spin_constants.requires_grad or (
            self.wscale.requires_grad
        )
        if not differentiable and self.cache_is_latest(cachevars) is True:
            if not isinstance(self.cache, SpinPolarizationCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        # if the cache is built, store the cachvar for validation
        self._cachevars = cachevars

        # spin constants (ss, sp, pp, sd, pd, dd) for each shell
        shell_constants = ihelp.spread_atom_to_shell(
            self.spin_constants[numbers], dim=-2, extra=True
        )

        # select the constant for each angular momentum pair (padding shells
        # have negative angular momenta and are masked below)
        angular = ihelp.angular.clamp(min=0)
        index = torch.tensor(ANGULAR_PAIR_INDEX, device=self.device)[
            angular.unsqueeze(-1), angular.unsqueeze(-2)
        ]
        wll = torch.gather(shell_constants, dim=-1, index=index)

        # only on-site (same atom) blocks of real shells
        sh2at = ihelp.shells_to_atom
        mask = (sh2at.unsqueeze(-1) == sh2at.unsqueeze(-2)) & (
            (sh2at.unsqueeze(-1) >= 0) & (sh2at.unsqueeze(-2) >= 0)
        )
        wll = torch.where(mask, wll, torch.zeros_like(wll))

        self.cache = SpinPolarizationCache(wll * self.wscale)
        return self.cache

    @override
    def get_monopole_shell_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        **_: Any,
    ) -> Tensor:
        """
        Calculate the shell-resolved spin-polarization energy.

        Parameters
        ----------
        cache : SpinPolarizationCache
            Restart data for the interaction.
        qat : Tensor
            Shell-resolved magnetization charges (shape: ``(..., nsh)``).

        Returns
        -------
        Tensor
            Shell-resolved spin-polarization energy (shape: ``(..., nsh)``).
        """
        return 0.5 * qat * self.get_monopole_shell_potential(cache, qat)

    @override
    def get_monopole_shell_potential(
        self,
        cache: InteractionCache,
        qsh: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate the shell-resolved magnetization potential.

        Parameters
        ----------
        cache : SpinPolarizationCache
            Restart data for the interaction.
        qsh : Tensor
            Shell-resolved magnetization charges (shape: ``(..., nsh)``).
        qdp : Tensor | None, optional
            Not used by the monopolar interaction.
        qqp : Tensor | None, optional
            Not used by the monopolar interaction.

        Returns
        -------
        Tensor
            Shell-resolved magnetization potential (shape: ``(..., nsh)``).
        """
        assert isinstance(cache, SpinPolarizationCache)
        return einsum("...ij,...j->...i", cache.wll, qsh)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(wscale={self.wscale})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)
