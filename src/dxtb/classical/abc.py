# This file is part of xtbml.
"""
Definition of energy terms as abstract base class for classical interactions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .._types import Tensor
from ..basis import IndexHelper


class Classical(ABC):
    """
    Abstract base class for calculation of classical contributions.
    """

    numbers: Tensor
    """The atomic numbers of the atoms in the system."""

    class Cache(ABC):
        """
        Abstract base class for the Cache of the contribution.
        """

    @abstractmethod
    def get_cache(self, numbers: Tensor, ihelp: IndexHelper) -> Cache:
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

        Note
        ----
        The cache of a classical contribution does not require `positions` as
        it only becomes useful if `numbers` remain unchanged and `positions`
        vary, i.e., during geometry optimization.
        """

    @abstractmethod
    def get_energy(self, positions: Tensor, cache: Cache, **kwargs: str) -> Tensor:
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
