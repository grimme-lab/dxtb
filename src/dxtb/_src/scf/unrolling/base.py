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
SCF Unrolling: Base
===================

Base class for all SCF implementations that unroll the SCF iterations in the
backward pass, i.e., the implicit function theorem is not used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tad_mctc import storch

from dxtb import OutputHandler
from dxtb._src.constants import labels
from dxtb._src.timing.decorator import timer_decorator
from dxtb._src.typing import Any, Tensor

from ..base import BaseSCF
from ..mixer import Anderson, Mixer, Simple

if TYPE_CHECKING:
    from dxtb._src.components.interactions import InteractionList
del TYPE_CHECKING

__all__ = ["BaseTSCF"]


class BaseTSCF(BaseSCF):
    """
    Base class for a standard self-consistent field iterator.

    This base class implements the `get_overlap` and the `diagonalize` methods
    using plain tensors. The diagonalization routine is taken from TBMaLT
    (hence the T in the class name).

    This base class only lacks the `scf` method, which implements mixing and
    convergence.
    """

    mixer: Mixer
    """Mixer for the SCF iterations."""

    def __init__(
        self,
        interactions: InteractionList,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(interactions, *args, **kwargs)

        # initialize the correct mixer with tolerances etc.
        if isinstance(self.config.mixer, Mixer):
            # TODO: We wont ever land here, int is enforced in the config
            self.mixer = self.config.mixer
        else:
            batched = self.config.batch_mode
            if self.config.mixer == labels.MIXER_LINEAR:
                self.mixer = Simple(self.fwd_options, batch_mode=batched)
            elif self.config.mixer == labels.MIXER_ANDERSON:
                self.mixer = Anderson(self.fwd_options, batch_mode=batched)
            elif self.config.mixer == labels.MIXER_BROYDEN:

                # Broyden is not implemented for SCF with full gradient, but
                # is the default setting. Without changing the setting, the
                # code immediately raises an error, which is inconvenient.
                if self.config.scf_mode not in (
                    labels.SCF_MODE_IMPLICIT,
                    labels.SCF_MODE_IMPLICIT_NON_PURE,
                ):
                    msg = (
                        "Broyden mixer is not implemented for SCF with full "
                        "gradient tracking."
                    )

                    if self.config.strict is True:
                        raise NotImplementedError(msg)

                    OutputHandler.warn(msg + " Using Anderson mixer instead.")
                    self.mixer = Anderson(self.fwd_options, batch_mode=batched)
            else:
                raise ValueError(f"Unknown mixer '{self.config.mixer}'.")

    def get_overlap(self) -> Tensor:
        """
        Get the overlap matrix.

        Returns
        -------
        Tensor
            Overlap matrix.
        """

        smat = self._data.ints.overlap

        zeros = torch.eq(smat, 0)
        mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

        return smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)

    @timer_decorator("Diagonalize", "SCF")
    def diagonalize(self, hamiltonian: Tensor) -> tuple[Tensor, Tensor]:
        """
        Diagonalize the Hamiltonian.

        The overlap matrix is retrieved within this method using the
        `get_overlap` method.

        Parameters
        ----------
        hamiltonian : Tensor
            Current Hamiltonian matrix.

        Returns
        -------
        evals : Tensor
            Eigenvalues of the Hamiltonian.
        evecs : Tensor
            Eigenvectors of the Hamiltonian.
        """
        o = self.get_overlap()

        # We only need to use a broadening method if gradients are required.
        if hamiltonian.requires_grad is False and o.requires_grad is False:
            broadening_method = None
        else:
            broadening_method = "lorn"

        return storch.eighb(
            a=hamiltonian,
            b=o,
            is_posdef=True,
            factor=torch.finfo(self.dtype).eps ** 0.5,
            broadening_method=broadening_method,
        )
