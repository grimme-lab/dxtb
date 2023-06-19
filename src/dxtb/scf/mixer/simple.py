"""
Simple Mixing
=============
"""
from __future__ import annotations

from ..._types import Any, Slicer, Tensor
from .base import Mixer

default_opts = {"maxiter": 100, "damp": 0.3}


class Simple(Mixer):
    """
    Perform simple or linear mixing for root finding. In simple mixing, a
    scalar Jacobian approximation is used, i.e., he damped differences are
    updated during each step.

    Inherits from the `Mixer` class.
    """

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        opts = dict(default_opts)
        if options is not None:
            opts.update(options)
        super().__init__(opts)

        # Holds the system from the previous iteration.
        self.x_old: Tensor | None = None

    def iter(self, x_new: Tensor, x_old: Tensor | None = None) -> Tensor:
        # Increment the step number variable
        self.iter_step += 1

        # Use the previous x_old value if none was specified
        x_old = self.x_old if x_old is None else x_old

        # Perform the mixing operation to create the new mixed x value
        x_mix = x_old + (x_new - x_old) * self.options["damp"]

        # Update the x_old attribute
        self.x_old = x_mix

        # Update the delta
        self._delta = x_mix - x_old

        # Return the newly mixed system
        return x_mix

    def cull(self, conv: Tensor, slicers: Slicer = (...,)) -> None:
        if self.x_old is None or self._delta is None:
            raise RuntimeError("Nothing has been mixed yet.")

        # Invert list for culling, gather & reassign `x_old` and `delta` so only
        # those marked False remain.
        notconv = ~conv
        self.x_old = self.x_old[[notconv, *slicers]]
        self._delta = self._delta[[notconv, *slicers]]
