"""
Classical contributions (ABC)
=============================

This module contains the abstract base class for all classical (i.e., non-
selfconsistent or density-dependent) energy terms.

Every contribution contains a `Cache` that holds position-independent
variables. Therefore, the positions must always be supplied to the `get_energy`
(or `get_grad`) method.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .._types import Tensor, TensorLike
from ..basis import IndexHelper


class ClassicalABC(ABC):
    """
    Abstract base class for calculation of classical contributions.
    """

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


class Classical(ClassicalABC, TensorLike):
    """
    Base class for calculation of classical contributions.
    """

    def get_gradient(self, energy: Tensor, positions: Tensor) -> Tensor:
        """
        Calculates nuclear gradient of an classical energy contribution via
        PyTorch's autograd engine.

        Parameters
        ----------
        energy : Tensor
            Energy that will be differentiated.
        positions : Tensor
            Nuclear positions. Needs `requires_grad=True`.

        Returns
        -------
        Tensor
            Nuclear gradient of `energy`.

        Raises
        ------
        RuntimeError
            `positions` tensor does not have `requires_grad=True`.
        """
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        (gradient,) = torch.autograd.grad(
            energy, positions, grad_outputs=torch.ones_like(energy)
        )
        return gradient
