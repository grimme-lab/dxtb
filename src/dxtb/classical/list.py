"""
Container for classical contributions.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from ..basis import IndexHelper
from ..utils import Timers
from .base import Classical


class ClassicalList(Classical):
    """
    List of classical contributions.
    """

    class Cache(Classical.Cache, dict):
        """
        List of classical contribution caches.
        """

        __slots__ = ()

    def __init__(
        self, *classicals: Classical | None, timer: Timers | None = None
    ) -> None:
        """
        Instantiate the collection of classical contributions.

        Parameters
        ----------
        classicals : tuple[Classical | None, ...] | list[Classical | None]
            List or tuple of classical contribution classes.
        timer : Timers | None, optional
            Instance of a timer collection. Defaults to `None`, which
            creates a new timer instance.

        Note
        ----
        Duplicate classical contributions will be removed automatically.
        """
        super().__init__(torch.device("cpu"), torch.float)
        self.classicals = list(
            {classical for classical in classicals if classical is not None}
        )

        self.timer = Timers("classicals") if timer is None else timer

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
        for classical in self.classicals:
            self.timer.start(classical.label)
            d[classical.label] = classical.get_cache(numbers=numbers, ihelp=ihelp)
            self.timer.stop(classical.label)

        cache.update(**d)
        return cache

    def get_energy(self, positions: Tensor, cache: Cache) -> dict[str, Tensor]:
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
        if len(self.classicals) <= 0:
            return {"none": positions.new_zeros(positions.shape[:-1])}

        energies = {}
        for classical in self.classicals:
            self.timer.start(classical.label)
            energies[classical.label] = classical.get_energy(
                positions, cache[classical.label]
            )
            self.timer.stop(classical.label)

        return energies

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
        if len(self.classicals) <= 0:
            return {"none": torch.zeros_like(positions)}

        gradients = {}
        for classical in self.classicals:
            self.timer.start(f"{classical.label} Gradient")
            gradients[classical.label] = classical.get_gradient(
                energy[classical.label], positions
            )
            self.timer.stop(f"{classical.label} Gradient")

        return gradients
