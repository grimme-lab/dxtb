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
"""
Simple Mixing
=============
"""

from __future__ import annotations

from dxtb._src.typing import Any, Slicer, Tensor

from .base import Mixer

__all__ = ["Simple"]


default_opts = {"maxiter": 100, "damp": 0.3}


class Simple(Mixer):
    """
    Perform simple or linear mixing for root finding. In simple mixing, a
    scalar Jacobian approximation is used, i.e., he damped differences are
    updated during each step.

    Inherits from the `Mixer` class.
    """

    def __init__(
        self, options: dict[str, Any] | None = None, batch_mode: int = 0
    ) -> None:
        opts = dict(default_opts)
        if options is not None:
            opts.update(options)
        super().__init__(opts, batch_mode=batch_mode)

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

    def cull(self, conv: Tensor, slicers: Slicer = (...,), mpdim: int = 1) -> None:
        if self.x_old is None or self._delta is None:
            raise RuntimeError("Nothing has been mixed yet.")

        if slicers == (...,):
            tmp = self.x_old.shape[-1]
        else:
            # NOTE: Only works with vectors (not with Charge container!)
            if isinstance(slicers[0], type(...)):
                tmp = slicers[0]
            elif isinstance(slicers[0], slice):
                tmp = slicers[0].stop
                if isinstance(tmp, Tensor):
                    tmp = int(tmp)
            else:
                raise RuntimeError("Unknown slicer given.")

        # Invert list for culling, gather & reassign `x_old` and `delta` so only
        # those marked False remain.
        notconv = ~conv
        self.x_old = self.x_old[notconv, :mpdim, :tmp]
        self._delta = self._delta[notconv, :mpdim, :tmp]
