"""
Container for classical contributions.
"""

from __future__ import annotations

import torch
from tad_mctc.typing import Any, Literal, Tensor, overload, override

from dxtb.basis import IndexHelper
from dxtb.timing import timer

from ..list import ComponentList, _docstring_reset, _docstring_update
from .base import Classical
from .halogen import LABEL_HALOGEN, Halogen
from .repulsion import LABEL_REPULSION, Repulsion


class ClassicalList(ComponentList[Classical]):
    """
    List of classical contributions.
    """

    @override
    def get_cache(self, numbers: Tensor, ihelp: IndexHelper) -> ClassicalList.Cache:
        """
        Create restart data for individual classical contributions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        ihelp: IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        ClassicalList.Cache
            Restart data for the classicals.
        """
        cache = self.Cache()

        d = {}
        for classical in self.components:
            timer.start(classical.label, parent_uid="Classicals")
            d[classical.label] = classical.get_cache(numbers=numbers, ihelp=ihelp)
            timer.stop(classical.label)

        cache.update(**d)
        return cache

    @override
    def get_energy(
        self, positions: Tensor, cache: ComponentList.Cache
    ) -> dict[str, Tensor]:
        """
        Compute the energy for a list of classicals.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        cache : Cache
            Restart data for the classical contribution.

        Returns
        -------
        dict[str, Tensor]
            Energy vectors of all classical contributions.
        """
        if len(self.components) <= 0:
            return {"none": positions.new_zeros(positions.shape[:-1])}

        energies = {}
        for classical in self.components:
            timer.start(classical.label)
            energies[classical.label] = classical.get_energy(
                positions, cache[classical.label]
            )
            timer.stop(classical.label)

        return energies

    @override
    def get_gradient(
        self, energy: dict[str, Tensor], positions: Tensor
    ) -> dict[str, Tensor]:
        """
        Calculate gradient for a list of classicals.

        Parameters
        ----------
        energy : dict[str, Tensor]
            Energies of all classical contributions that will be differentiated.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).

        Returns
        -------
        dict[str, Tensor]
            Nuclear gradients of all classical contributions.
        """
        if len(self.components) <= 0:
            return {"none": torch.zeros_like(positions)}

        gradients = {}
        for classical in self.components:
            timer.start(f"{classical.label} Gradient")
            gradients[classical.label] = classical.get_gradient(
                energy[classical.label], positions
            )
            timer.stop(f"{classical.label} Gradient")

        return gradients

    ###########################################################################

    @overload
    def get_interaction(self, name: Literal["Halogen"]) -> Halogen: ...

    @overload
    def get_interaction(self, name: Literal["Repulsion"]) -> Repulsion: ...

    @override  # generic implementation for typing
    def get_interaction(self, name: str) -> Classical:
        return super().get_interaction(name)

    ###########################################################################

    @_docstring_reset
    def reset_halogen(self) -> Classical:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_HALOGEN)

    @_docstring_reset
    def reset_repulsion(self) -> Classical:
        """Reset tensor attributes to a detached clone of the current state."""
        return self.reset(LABEL_REPULSION)

    ###########################################################################

    @_docstring_update
    def update_halogen(self, **kwargs: Any) -> Classical:
        return self.update(LABEL_HALOGEN, **kwargs)

    @_docstring_update
    def update_repulsion(self, **kwargs: Any) -> Classical:
        """"""
        return self.update(LABEL_REPULSION, **kwargs)
