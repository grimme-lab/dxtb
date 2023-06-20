"""
ABC: SCF Mixer
==============

This module contains the abstract base class for all mixers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..._types import Any, Slicer, Tensor

default_opts = {"maxiter": 20, "damp": 0.3, "f_tol": 1e-5, "x_tol": 1e-5}


class Mixer(ABC):
    """
    Abstract base class for mixer.
    """

    label: str
    """Label for the Mixer."""

    iter_step: int
    """Number of mixing iterations taken."""

    options: dict[str, Any]
    """Options for the mixer (damping, tolerances, ...)."""

    _delta: Tensor | None
    """Difference between the current and previous systems."""

    _is_batch: bool
    """Whether the mixer operates in batch mode."""

    def __init__(
        self, options: dict[str, Any] | None = None, is_batch: bool = False
    ) -> None:
        self.label = self.__class__.__name__
        self.options = options if options is not None else default_opts
        self.iter_step = 0
        self._delta = None

        # inferring batch mode from shapes of tensor is unreliable, so we
        # explicitly set this information
        self._is_batch = is_batch

    def __repr__(self) -> str:
        """
        Returns representative string.
        """
        return f"{self.__class__.__name__}({self.iter_step})"

    @abstractmethod
    def iter(self, x_new: Tensor, x_old: Tensor | None = None) -> Tensor:
        """
        Performs the mixing operation & returns the newly mixed system.

        This should contain only the code required to carry out the mixing
        operation.

        Parameters
        ----------
        x_new : Tensor
            New system.
        x_old : Tensor | None, optional
            Old system. Default to `None`.

        Returns
        -------
        Tensor
            Newly mixed system(s).

        Note
        ----
        The ``x_old`` argument is normally identical to the ``x_mix``
        value returned from the previous iteration, which is stored by the
        class internally. As such, the ``x_old`` argument can be omitted
        from all but the first step if desired.
        """

    @abstractmethod
    def cull(self, conv: Tensor, slicers: Slicer = (...,)) -> None:
        """
        Purge selected systems from the mixer.

        This is useful when a subset of systems have converged during mixing.

        Parameters
        ----------
        conv : Tensor
            Tensor with booleans indicating which systems should be culled
            (True) and which should remain (False).
        slicers : Slicer, optional
            New anticipated size of future inputs excluding the batch
            dimension. This is used to allow superfluous padding values to
            be removed form subsequent inputs. Defaults to `(...,)`.
        """

    @property
    def delta(self) -> Tensor:
        """
        Difference between the current and previous systems.

        This may need to be locally overridden if `_delta` needs to be
        reshaped prior to it being returned.
        """
        if self._delta is None:
            raise RuntimeError("Mixer has no been started yet.")
        return self._delta

    @property
    def converged(self) -> Tensor:
        """
        Tensor of bools indicating convergence status of the system(s).

        A system is considered to have converged if the maximum absolute
        difference between the current and previous systems is less than
        the ``tolerance`` value.
        """
        # Check that mixing has been conducted
        if self.delta is None:
            raise RuntimeError("Nothing has been mixed")

        if self._is_batch is True:
            # norm goes over all dims except first (batch dimension)
            dims = tuple(range(-(self.delta.ndim - 1), 0))
            delta_norm = torch.norm(self.delta, dim=dims)
        else:
            delta_norm = torch.norm(self.delta)

        return delta_norm < self.options["x_tol"]

    def reset(self):
        """
        Resets the mixer to its initial state.

        Calling this function will reset the class & its internal attributes.
        However, any properties set during the initialisation process will be
        retained.
        """
        self.iter_step = 0
        self.x_old = None
        self._delta = None
