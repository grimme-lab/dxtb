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
# pylint: disable=unused-argument

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.typing import Any, Slicers, Tensor, TensorOrTensors, final

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
        """Remove converged data from the cache."""

    def restore(self) -> None:
        """Restore the cache from the last iteration."""


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

    # Already defined in `Component` parent class
    label: str
    """Label for the interaction."""

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the interaction."""
        super().__init__(device, dtype)

    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
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

    @final
    def get_potential(
        self,
        cache: InteractionCache,
        charges: Charges,
        ihelp: IndexHelper,
    ) -> Potential:
        """
        Compute the potential from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        charges : Charges
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        # monopole potential: shell-resolved
        qsh = ihelp.reduce_orbital_to_shell(charges.mono)
        vsh = self.get_monopole_shell_potential(cache, qsh)

        # monopole potential: atom-resolved
        qat = ihelp.reduce_shell_to_atom(qsh)
        vat = self.get_monopole_atom_potential(
            cache, qat, qdp=charges.dipole, qqp=charges.quad
        )

        # spread to orbital-resolution
        vsh += ihelp.spread_atom_to_shell(vat)
        vmono = ihelp.spread_shell_to_orbital(vsh)

        # multipole potentials
        vdipole = self.get_dipole_atom_potential(
            cache, qat, charges.dipole, charges.quad
        )
        vquad = self.get_quadrupole_atom_potential(
            cache, qat, charges.dipole, charges.quad
        )

        return Potential(vmono, dipole=vdipole, quad=vquad, label=self.label)

    def get_monopole_shell_potential(
        self,
        cache: ComponentCache,
        qsh: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the potential from the charges, all quantities are shell-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qsh : Tensor
            Shell-resolved partial charges.

        Returns
        -------
        Tensor
            Potential vector for each atom partial charge.
        """
        return torch.zeros_like(qsh)

    def get_monopole_atom_potential(
        self,
        cache: ComponentCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the potential from the charges, all quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved potential vector for each atom partial charge.
        """
        return torch.zeros_like(qat)

    def get_dipole_atom_potential(
        self,
        cache: ComponentCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor | None:
        """
        Compute the dipole potential. All quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning None.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor | None
            Atom-resolved potential vector for each atom or None if not needed.
        """
        return None

    def get_quadrupole_atom_potential(
        self,
        cache: ComponentCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor | None:
        """
        Compute the quadrupole potential. All quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning None.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor | Nene
            Atom-resolved potential vector for each atom or None if not needed.
        """
        return None

    ##########################################################################

    @final
    def get_energy(
        self, cache: InteractionCache, charges: Charges, ihelp: IndexHelper
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are
        orbital-resolved.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Atom-resolved energy vector.

        Note
        ----
        The subclasses of :class:`dxtb.components.base.Interaction` should
        implement the `get_<type>_energy` methods. If they are not implemented
        in the subclass, they will evaluate to zero.
        """
        if charges.mono is None:
            raise RuntimeError(
                "Charge collection is empty. At least monopolar partial "
                "charges are required."
            )

        qsh = ihelp.reduce_orbital_to_shell(charges.mono)
        esh = self.get_monopole_shell_energy(cache, qsh)

        qat = ihelp.reduce_shell_to_atom(qsh)
        eat = self.get_monopole_atom_energy(cache, qat)

        e = eat + ihelp.reduce_shell_to_atom(esh)

        if charges.dipole is not None:
            edp = self.get_dipole_atom_energy(
                cache, qat=qat, qdp=charges.dipole, qqp=charges.quad
            )
            e += edp

        if charges.quad is not None:
            eqp = self.get_quadrupole_atom_energy(
                cache, qat=qat, qdp=charges.dipole, qqp=charges.quad
            )
            e += eqp

        return e

    def get_monopole_atom_energy(
        self, cache: InteractionCache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).

        Returns
        -------
        Tensor
            Energy vector for each atom partial charge.
        """
        return torch.zeros_like(qat)

    def get_monopole_shell_energy(
        self, cache: InteractionCache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are shell-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        qat : Tensor
            Shell-resolved partial charges.

        Returns
        -------
        Tensor
            Energy vector for each shell partial charge.
        """
        return torch.zeros_like(qat)

    def get_dipole_atom_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the energy from the atomic dipole moments, all quantities are
        atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Energy vector for each atomic dipole moment.
        """
        return torch.zeros_like(qat)

    def get_quadrupole_atom_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the energy from the atomic quadrupole moments, all quantities
        are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Energy vector for each atomic quadrupole moment.
        """
        return torch.zeros_like(qat)

    ##########################################################################

    @final
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

    def get_atom_gradient(
        self,
        charges: Tensor,
        positions: Tensor,
        cache: InteractionCache,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
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

    def get_shell_gradient(
        self,
        charges: Tensor,
        positions: Tensor,
        cache: InteractionCache,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
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
