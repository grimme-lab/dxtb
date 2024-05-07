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
Anderson Mixing
===============

This module contains the Andersion mixing algorithm.

The implementation is taken from TBMaLT (with minor modifications).
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb._src.typing import Any, Slicer, Tensor
from dxtb._src.utils import t2int

from .base import Mixer

__all__ = ["Anderson"]


default_opts = {
    "maxiter": 100,
    "damp": 0.5,
    "generations": 5,
    "diagonal_offset": 0.01,
    "damp_init": 0.01,
}


class Anderson(Mixer):
    """
    Accelerated Anderson mixing algorithm.

    Anderson mixing is a method for accelerating convergence. Instead of mixing
    the input and output vectors directly together (simple mixing), it uses the
    "optimal linear combination of the input and output vectors within the
    spaces spanned by the vectors of the M previous iterations". This way, "the
    memory of the whole iteration process is built in which helps finding the
    final solution quite fast".

    Note
    ----
    Note that simple mixing will be used for the first ``generations``
    number of steps

    The Anderson mixing functions primarily follow the equations set out
    by Eyert [Eyert]_. However, this code borrows heavily from the DFTB+
    implementation [DFTB]_. This deviates from the DFTB+ implementation
    in that it does not compute or use the theta zero values, as they
    cause stability issues in this implementation. For more information on
    Anderson mixing see See "Anderson Acceleration, Mixing and
    Extrapolation" [Anderson]_.

    Warning
    -------
    Setting ``generations`` too high can lead to a linearly dependent set
    of equations. However, this effect can be mitigated through the use of
    the ``diagonal_offset`` parameter.

    References
    ----------
    .. [Eyert] Eyert, V. (1996). A Comparative Study on Methods for
       Convergence Acceleration of Iterative Vector Sequences. Journal of
       Computational Physics, 124(2), 271–285.
    .. [DFTB] Hourahine, B., Aradi, B., Blum, V., Frauenheim, T. et al.,
       (2020). DFTB+, a software package for efficient approximate density
       functional theory based atomistic simulations. The Journal of
       Chemical Physics, 152(12), 124101.
    .. [Anderson] Anderson, D. M. (2018). Comments on “Anderson Acceleration,
       Mixing and Extrapolation.” Numerical Algorithms, 80(1), 135–234.
    """

    mix_param: float
    """
    Mixing parameter, ∈(0, 1), controls the extent of mixing. Larger values
    result in more aggressive mixing. Defaults to 0.5 according to [Eyert]_.
    """

    init_mix_param: float
    """Mixing parameter to use during the initial simple mixing steps (0.01)."""

    diagonal_offset: float
    """
    Offset added to the equation system's diagonal's to prevent a linear
    dependence during the mixing process. If set to ``None`` then rescaling will
    be disabled. [DEFAULT=0.01]
    """

    generations: int
    """
    Number of generations to use during mixing.
    Defaults to 5 as suggested by [Eyert]_.
    """

    def __init__(
        self, options: dict[str, Any] | None = None, batch_mode: int = 0
    ) -> None:
        opts = dict(default_opts)
        if options is not None:
            opts.update(options)
        super().__init__(opts, batch_mode=batch_mode)

        self.mix_param = self.options["damp"]
        self.generations = self.options["generations"]
        self.init_mix_param = self.options["damp_init"]
        self.diagonal_offset = self.options["diagonal_offset"]

        # Holds "x" history and "x" delta history
        self._x_hist: Tensor | None = None
        self._f: Tensor | None = None

        # Systems are flatted during input to permit shape agnostic
        # programming. The shape information is stored here.
        self._shape_in: list[int] | None = None
        self._shape_out: list[int] | None = None

    def _setup_hook(self, x_new: Tensor) -> None:
        """
        Perform post instantiation initialisation operation.

        This instantiates internal variables.

        Parameters
        ----------
        x_new : Tensor
            New system(s) that is to be mixed.
        """
        # Tensors are converted to _shape_in when passed in and back to their
        # original shape _shape_out when returned to the user.
        self._shape_out = list(x_new.shape)

        if self._batch_mode == 0:
            self._shape_in = list(torch.flatten(x_new).shape)
        else:
            self._shape_in = list(x_new.reshape(x_new.shape[0], -1).shape)

        # Instantiate the x history (x_hist) and the delta history 'd_hist'
        size = (self.generations + 1, *self._shape_in)
        self._x_hist = x_new.new_zeros(size)
        self._f = x_new.new_zeros(size)

    @property
    def delta(self) -> Tensor:
        """
        Difference between the current and previous systems.

        Note
        ----
        The output delta is reshaped.
        """
        if self._delta is None or self._shape_out is None:
            raise RuntimeError("Mixer has no been started yet.")

        return self._delta.reshape(self._shape_out)

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
            Old system. Default to ``None``.

        Returns
        -------
        Tensor
            Newly mixed system(s).

        Note
        ----
        Simple mixing will be used for the first n steps, where n is the
        number of previous steps to be use in the mixing process.
        """
        # At some point a check should be put in place to give a more useful
        # error when the user padding values from new inputs without providing
        # a new_size argument during calls to the cull method.

        # Call setup hook if this is the 1st cycle.
        if self.iter_step == 0:
            # check if x_old (=guess) is given in first iteration
            if x_old is None:
                raise RuntimeError(
                    "In the first iteration, the `x_old` argument cannot be "
                    "``None`` as it is the starting guess."
                )

            self._setup_hook(x_new)

        # The setup hook should have initialized the history
        if (
            self._shape_out is None
            or self._shape_in is None
            or self._x_hist is None
            or self._f is None
        ):
            raise RuntimeError(
                "The `setup_hook` of the mixer failed to initalize all "
                "required variables."
            )

        if self.iter_step == 0:
            self._x_hist[0] = x_old.reshape(self._shape_in)  # type: ignore

        self.iter_step += 1  # Increment step_number

        # Following Eyert's notation, "f" refers to the delta:
        #   F = x_new - x_old
        # However, for clarity "x_hist" is used in place of Eyert's "x".

        # Inputs must be reshaped to ensure they a vector (or batch thereof)
        x_new = x_new.reshape(self._shape_in)

        # If x_old specified; overwrite last entry in self._x_hist.
        x_old = self._x_hist[0] if x_old is None else x_old.reshape(self._shape_in)

        # Calculate x_new - x_old delta & assign to the delta history _f
        self._f[0] = x_new - x_old

        # If a sufficient history has been built up then use Anderson mixing
        if self.iter_step > self.generations:
            # Setup and solve the linear equation system, as described in
            # equation 4.3 (Eyert), to get the coefficients "thetas":
            #   a(i,j) =  <F(l) - F(l-i)|F(l) - F(l-j)>
            #   b(i)   =  <F(l) - F(l-i)|F(l)>
            # here dF = <F(l) - F(l-i)|
            df = self._f[0] - self._f[1:]
            a = einsum("i...v,j...v->...ij", df, df)
            b = einsum("h...v,...v->...h", df, self._f[0])

            # Rescale diagonals to prevent linear dependence on the residual
            # vectors by adding 1 + offset^2 to the diagonals of "a", see
            # equation 8.2 (Eyert)
            if self.diagonal_offset is not None:
                eye = torch.eye(a.shape[-1], device=x_new.device, dtype=x_new.dtype)
                one = torch.tensor(1.0, device=x_new.device, dtype=x_new.dtype)
                a *= torch.where(eye != 0, eye + self.diagonal_offset**2, one)

            # Solve for the coefficients. As torch.solve cannot solve for 1D
            # tensors a blank dimension must be added

            thetas = torch.squeeze(torch.linalg.solve(a, torch.unsqueeze(b, -1)))

            # Construct the 2'nd terms of eq 4.1 & 4.2 (Eyert). These are
            # the "averaged" histories of x and F respectively:
            #   x_bar = sum(j=1 -> m) ϑ_j(l) * (|x(l-j)> - |x(l)>)
            #   f_bar = sum(j=1 -> m) ϑ_j(l) * (|F(l-j)> - |F(l)>)
            # These are not the x_bar & F_var values of eq. 4.1 & 4.2 (Eyert)
            # yet as they are still missing the 1st terms.
            x_bar = einsum(
                "...h,h...v->...v", thetas, (self._x_hist[1:] - self._x_hist[0])
            )
            f_bar = einsum("...h,h...v->...v", thetas, -df)

            # The first terms of equations 4.1 & 4.2 (Eyert):
            #   4.1: |x(l)> and & 4.2: |F(l)>
            # Have been replaced by:
            #   ϑ_0(l) * |x(j)> and ϑ_0(l) * |x(j)>
            # respectively, where "ϑ_0(l)" is the coefficient for the current
            # step and is defined as (Anderson):
            #   ϑ_0(l) = 1 - sum(j=1 -> m) ϑ_j(l)
            # Code deviates from DFTB+ here to prevent "stability issues"
            # theta_0 = 1 - torch.sum(thetas)
            # x_bar += theta_0 * self._x_hist[0]  # <- DFTB+
            # f_bar += theta_0 * self._f[0]  # <- DFTB+
            x_bar += self._x_hist[0]
            f_bar += self._f[0]

            # Calculate the new mixed dQ following equation 4.4 (Eyert):
            #   |x(l+1)> = |x_bar(l)> + beta(l)|f_bar(l)>
            # where "beta" is the mixing parameter
            x_mix = x_bar + (self.mix_param * f_bar)

        # If there is insufficient history for Anderson; use simple mixing
        else:
            x_mix = self._x_hist[0] + (self._f[0] * self.init_mix_param)

        # Shift f & x_hist over; a roll follow by a reassignment is
        # necessary to avoid an inplace error. (gradients remain intact)
        self._f = torch.roll(self._f, 1, 0)
        self._x_hist = torch.roll(self._x_hist, 1, 0)

        # Assign the mixed x to the x_hist history array. The last x_mix value
        # is saved on the assumption that it will be used in the next step.
        self._x_hist[0] = x_mix

        # Save the last difference to _delta
        self._delta = self._f[1]

        # Reshape the mixed system back into the expected shape and return it
        return x_mix.reshape(self._shape_out)

    def cull(self, conv: Tensor, slicers: Slicer = (...,), mpdim: int = 1) -> None:
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
        if (
            self._shape_out is None
            or self._shape_in is None
            or self._delta is None
            or self._x_hist is None
            or self._f is None
        ):
            raise RuntimeError("Mixer has not been started yet.")

        if slicers == (...,):
            shape = self._shape_out[1:]
        else:
            # NOTE: Maybe refactor the whole slicer approach...
            if isinstance(slicers[0], type(...)):
                tmp = slicers[0]
            elif isinstance(slicers[0], slice):
                tmp = slicers[0].stop
                if isinstance(tmp, Tensor):
                    tmp = t2int(tmp)
            else:
                raise RuntimeError("Unknown slicer given.")

            shape = [mpdim, tmp]

        # Length of flattened arrays after factoring in the new
        l = t2int(torch.prod(torch.tensor(shape, device=self._x_hist.device)))

        # Invert the cull_list, gather & reassign self._delta self._x_hist &
        # self._f so only those marked False remain.
        notconv = ~conv

        def _cull(tensor: Tensor) -> Tensor:
            # Perform culling on flattened tensor
            culled = tensor[..., notconv, :]
            shp = culled.shape[:-1]

            # Reshape tensor (unflatten)
            assert self._shape_out is not None
            reshaped = culled.view(*shp, *self._shape_out[1:])

            # Select elements and reshape back to flattened view
            return reshaped[..., :mpdim, : (l // mpdim)].contiguous().view(*shp, -1)

        self._delta = _cull(self._delta)
        self._f = _cull(self._f)
        self._x_hist = _cull(self._x_hist)

        # Adjust the the shapes accordingly
        self._shape_in[0] -= list(conv).count(
            torch.tensor(True, device=self._x_hist.device)
        )
        self._shape_in[-1] = l
        self._shape_out = [self._shape_in[0], *shape]

    def reset(self):
        """Reset mixer to its initial state."""
        self.iter_step = 0
        self._x_hist = self._f = self._delta = None
        self._shape_in = self._shape_out = None
