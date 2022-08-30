# This file is part of xtbml.

"""
Definition of energy terms as abstract base class for classical interactions.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from ..basis import IndexHelper
from ..typing import Tensor


class Classical(ABC):
    """
    Abstract base class for calculation of classical contributions.
    """

    numbers: Tensor
    """The atomic numbers of the atoms in the system."""

    class Cache(ABC):
        pass

    @abstractmethod
    def get_cache(self, numbers: Tensor, ihelp: IndexHelper) -> "Cache":
        """
        Store variables for energy calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms.
        ihelp : IndexHelper
            Helper class for indexing.

        Returns
        -------
        Cache
            Cache class for storage of variables.
        """
        ...

    @abstractmethod
    def get_energy(self, positions: Tensor, cache: "Cache") -> Tensor:
        """
        Obtain energy of the contribution.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms.
        cache : Halogen.Cache
            Cache for the halogen bond parameters.

        Returns
        -------
        Tensor
             Atomwise energy contributions.
        """
        ...