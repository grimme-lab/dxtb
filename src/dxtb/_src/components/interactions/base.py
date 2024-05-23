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
# pylint: disable=assignment-from-none
"""
Provides base class for interactions in the extended tight-binding Hamiltonian.
The `Interaction` class is not purely abstract as its methods return zero.
"""
from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.typing import Any, Slicers, Tensor, TensorOrTensors

from ...components.base import Component, ComponentCache
from .container import Charges, Potential

__all__ = ["Interaction", "InteractionCache"]


class InteractionCache(ComponentCache):
    """
    Restart data for individual interactions, extended by subclasses as
    needed.
    """

    __slots__: list[str] = []

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        pass

    def restore(self) -> None:
        pass


class Interaction(Component):
    """
    Base class for defining interactions with the charge density.

    Every charge-dependent energy contribution should inherit from this class
    as it conveniently handles the different resolutions (atom, shell, orbital)
    required for all subclasses. Depending on the contribution the user must
    implement a ``get_<atom/shell>_<potential/energy/gradient>`` method. The
    methods that are not implemented automatically evaluate to zero. As they
    are always called in the process of collecting the different resolutions,
    you must NOT implement them.

    .. warning::

        Never overwrite the ``get_potential``, ``get_energy`` and
        ``get_gradient`` methods in the subclass. These methods collect the
        respective variable for their (internally-handled) evaluation.

    Additionally, a ``get_cache`` method is required that should precalculate
    and store all charge-independent variables to avoid repeated calculations
    during the SCF.

    Note
    ----
    The nuclear gradient of the Mulliken charges essentially yields a
    derivative of the overlap integral. This is handled elsewhere and must
    not be implemented by the ``get_<atom/shell>_gradient`` methods.

    Example
    -------
    The third-order electrostatics class:`dxtb.components.ES3` of GFN1-xTB
    contain only atom-resolved parameters. Hence, it only implements
    ``get_atom_energy`` and ``get_atom_potential`` (besides the required
    ``get_cache`` method).
    Since there is not positional dependence (except for the charges, which is
    handled elsewhere) in the third-order electrostatics, no gradient method
    must be implemented.
    """

    label: str
    """Label for the interaction."""

    __slots__ = ["label"]

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the interaction."""
        super().__init__(device, dtype)

    # pylint: disable=unused-argument
    def get_cache(
        self, *, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> InteractionCache:
        """
        Create restart data for individual interactions.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty :class:`.Interaction` by returning an empty
        class:`.InteractionCache`.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp: IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        InteractionCache
            Restart data for the interaction.
        """
        return InteractionCache()

    def get_potential(
        self,
        charges: Charges,
        cache: InteractionCache,
        ihelp: IndexHelper,
    ) -> Potential:
        """
        Compute the potential from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        charges : Charges
            Orbital-resolved partial charges.
        cache : InteractionCache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        # monopole potential: shell-resolved
        qsh = ihelp.reduce_orbital_to_shell(charges.mono)
        vsh = self.get_shell_potential(qsh, cache)

        # monopole potential: atom-resolved
        qat = ihelp.reduce_shell_to_atom(qsh)
        vat = self.get_atom_potential(qat, cache)

        # spread to orbital-resolution
        vsh += ihelp.spread_atom_to_shell(vat)
        vmono = ihelp.spread_shell_to_orbital(vsh)

        # multipole potentials
        vdipole = self.get_dipole_potential(charges, cache)
        vquad = self.get_quadrupole_potential(charges, cache)

        return Potential(vmono, dipole=vdipole, quad=vquad, label=self.label)

    def get_shell_potential(self, charges: Tensor, *_) -> Tensor:
        """
        Compute the potential from the charges, all quantities are shell-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.

        Returns
        -------
        Tensor
            Potential vector for each atom partial charge.
        """
        return torch.zeros_like(charges)

    def get_atom_potential(self, charges: Tensor, *_) -> Tensor:
        """
        Compute the potential from the charges, all quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.

        Returns
        -------
        Tensor
            Atom-resolved potential vector for each atom partial charge.
        """
        return torch.zeros_like(charges)

    def get_dipole_potential(self, *_) -> Tensor | None:
        """
        Compute the dipole potential. All quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning None.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.

        Returns
        -------
        Tensor | None
            Atom-resolved potential vector for each atom or None if not needed.
        """
        return None

    def get_quadrupole_potential(self, *_) -> Tensor | None:
        """
        Compute the quadrupole potential. All quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning None.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.

        Returns
        -------
        Tensor | Nene
            Atom-resolved potential vector for each atom or None if not needed.
        """
        return None

    def get_energy(
        self, charges: Charges, cache: InteractionCache, ihelp: IndexHelper
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        cache : InteractionCache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Atom-resolved energy vector.

        Note
        ----
        The subclasses of `Interaction` should implement the `get_<type>_energy`
        methods. If they are not implemented in the subclass, they will
        evaluate to zero.
        """
        if charges.mono is None:
            raise RuntimeError(
                "Charge collection is empty. At least monopolar partial "
                "charges are required."
            )

        qsh = ihelp.reduce_orbital_to_shell(charges.mono)
        esh = self.get_shell_energy(qsh, cache)

        qat = ihelp.reduce_shell_to_atom(qsh)
        eat = self.get_atom_energy(qat, cache)

        e = eat + ihelp.reduce_shell_to_atom(esh)

        if charges.dipole is not None:
            edp = self.get_dipole_energy(charges.dipole, cache)
            e += edp

        if charges.quad is not None:
            eqp = self.get_quadrupole_energy(charges.quad, cache)
            e += eqp

        return e

    def get_atom_energy(self, charges: Tensor, *_: Any) -> Tensor:
        """
        Compute the energy from the charges, all quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.

        Returns
        -------
        Tensor
            Energy vector for each atom partial charge.
        """
        return torch.zeros_like(charges)

    def get_shell_energy(self, charges: Tensor, *_: Any) -> Tensor:
        """
        Compute the energy from the charges, all quantities are shell-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.

        Returns
        -------
        Tensor
            Energy vector for each shell partial charge.
        """
        return torch.zeros_like(charges)

    def get_dipole_energy(self, charges: Tensor, *_: Any) -> Tensor:
        """
        Compute the energy from the atomic dipole moments, all quantities are
        atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Atomic dipole moments of all atoms.

        Returns
        -------
        Tensor
            Energy vector for each atomic dipole moment.
        """
        return torch.zeros_like(charges).sum(-1)

    def get_quadrupole_energy(self, charges: Tensor, *_: Any) -> Tensor:
        """
        Compute the energy from the atomic quadrupole moments, all quantities
        are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Atomic quadrupole moments of all atoms.

        Returns
        -------
        Tensor
            Energy vector for each atomic quadrupole moment.
        """
        return torch.zeros_like(charges).sum(-1)

    def get_gradient(
        self,
        charges: Charges,
        positions: Tensor,
        cache: InteractionCache,
        ihelp: IndexHelper,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Compute the nuclear gradient using orbital-resolved charges.

        Note
        ----
        This method calls both :meth:`.get_atom_gradient` and
        :meth:`.get_shell_gradient` and adds up both gradients. Hence, one of
        the contributions must be zero.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : InteractionCache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Nuclear gradient for each atom.
        """
        qao = charges.mono.detach()

        qsh = ihelp.reduce_orbital_to_shell(qao)
        gsh = self.get_shell_gradient(
            qsh, positions, cache, grad_outputs, retain_graph, create_graph
        )

        qat = ihelp.reduce_shell_to_atom(qsh)
        gat = self.get_atom_gradient(
            qat, positions, cache, grad_outputs, retain_graph, create_graph
        )

        return gsh + gat

    def get_shell_gradient(self, _: Any, positions: Tensor, *__: Any) -> Tensor:
        """
        Return zero gradient.

        This method should be implemented by the subclass.
        However, returning zeros here serves three purposes:

        - the interaction can (theoretically) be empty
        - the gradient of the interaction is indeed zero and thus requires no
          gradient implementation (one can, however, implement a method that
          returns zeros to make this more obvious)
        - the interaction always uses atom-resolved charges and shell-resolved
          charges are never required

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates (for shape of gradient).

        Returns
        -------
        Tensor
            Nuclear gradient for each atom.
        """
        return torch.zeros_like(positions)

    def get_atom_gradient(self, _: Any, positions: Tensor, *__: Any) -> Tensor:
        """
        Return zero gradient.

        This method should be implemented by the subclass.
        However, returning zeros here serves three purposes:

        - the interaction can (theoretically) be empty
        - the gradient of the interaction is indeed zero and thus requires no
          gradient implementation (one can, however, implement a method that
          returns zeros to make this more obvious)
        - the interaction always uses shell-resolved charges and atom-resolved
          charges are never required

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates (for shape of gradient).

        Returns
        -------
        Tensor
            Nuclear gradient for each atom.
        """
        return torch.zeros_like(positions)
