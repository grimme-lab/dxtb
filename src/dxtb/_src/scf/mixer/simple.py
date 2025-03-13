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

from .base import DEFAULT_OPTS, Mixer

__all__ = ["Simple"]


class Simple(Mixer):
    r"""
    Simple linear mixer mixing algorithm.

    Iteratively mixes pairs of systems via a simple linear combination:

    .. math::

        (1-f)x_n + fx_{n-1}

    Where :math:`x_n`/:math:`x_{n-1}` are the new/old systems respectively &
    :math:`f` is the mixing parameter, i.e. the fraction of :math:`x_n` that
    is to be retained. Given a suitable tolerance, and a small enough mixing
    parameter, the `Simple` mixer is guaranteed to converge, however, it also
    tends to be significantly slower than other, more advanced, methods.

    Note
    ----
    Mixer instances require the user to explicitly specify during
    initialisation whether it is a single system or a batch of systems
    that are to be mixed.

    Examples
    --------
    The attractive fixed point of the function:

    Integral feed updates
    >>> import torch
    >>>
    >>> def func(x: torch.Tensor) -> torch.Tensor:
    >>>     return torch.tensor(
    >>>         [0.5 * torch.sqrt(x[0] + x[1]), 1.5 * x[0] + 0.5 * x[1]]
    >>>     )

    can be identified using the ``.Simple`` mixer as follows:

    >>> from dxtb._src.scf.mixer import Simple
    >>>
    >>> x = torch.tensor([2., 2.])  # Initial guess
    >>> mixer = Simple()
    >>> for i in range(200):
    >>>     x = mixer.iter(func(x), x)
    >>> print(x)
    >>> # tensor([1., 3.])
    """

    def __init__(
        self, options: dict[str, Any] | None = None, batch_mode: int = 0
    ) -> None:
        opts = dict(DEFAULT_OPTS)
        if options is not None:
            opts.update(options)
        super().__init__(opts, batch_mode=batch_mode)

        # Holds the system from the previous iteration.
        self._x_old: Tensor | None = None

    def iter(self, x_new: Tensor, x_old: Tensor | None = None) -> Tensor:
        """
        Performs the simple mixing operation & returns the mixed system.

        Iteratively mix pairs of systems via a simple linear combination:

        .. math::

            (1-f)x_n + fx_{n-1}

        Parameters
        ----------
        x_new : Tensor
            New system(s) that is to be mixed.
        x_old : Tensor | None, optional
            Previous system(s) that is to be mixed. Only required for the
            first mix operation. Default to ``None``.

        Returns
        -------
        x_mix : Tensor
            Newly mixed system(s).

        Note
        ----
        The ``x_old`` argument is normally identical to the ``x_mix``
        value returned from the previous iteration, which is stored by the
        class internally. As such, the ``x_old`` argument can be omitted
        from all but the first step if desired.
        """
        # Increment the step number variable
        self.iter_step += 1

        # Use the previous x_old value if none was specified
        x_old = self._x_old if x_old is None else x_old

        # Safety check
        if self._batch_mode > 0:
            if x_old is not None:
                if x_new.shape[0] != x_old.shape[0]:
                    raise RuntimeError(
                        "Batch dimension of x_new and x_old do not match; "
                        "ensure calls are made to mixer.cull as needed."
                    )
                if x_new.shape[1:] != x_old.shape[1:]:
                    raise RuntimeError(
                        f"Non-batch dimension mismatch encountered - x_new "
                        f"{x_new.shape}, x_old {x_old.shape}."
                    )

        # Perform the mixing operation to create the new mixed x value
        # if self.options["damp"] == 1.0:
        # x_mix = x_new
        # else:
        x_mix = x_old + (x_new - x_old) * self.options["damp"]

        # Update the x_old attribute
        self.x_old = x_mix

        # Update the delta
        self._delta = x_new - x_old

        # Return the newly mixed system
        return x_mix

    def cull(
        self, conv: Tensor, slicers: Slicer = (...,), mpdim: int = 1
    ) -> None:
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
