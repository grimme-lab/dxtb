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
SCF Implicit: Standard Variant
==============================

Standard implementation of SCF iterations utilizing the implicit function
theorem for the backward (see :mod:`.fixed_point`).
"""

from __future__ import annotations

import torch

from dxtb._src.components.interactions import Charges, Potential
from dxtb._src.constants import labels
from dxtb._src.typing import Tensor

from ..mixer import Simple
from ..pure.iterations import iterate_charges
from .base import BaseXSCF
from .fixed_point import equilibrium

__all__ = ["SelfConsistentFieldImplicit"]


class SelfConsistentFieldImplicit(BaseXSCF):
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    The default class makes use of the implicit function theorem. Hence, the
    derivatives of the iterative procedure are only calculated from the
    equilibrium solution, i.e., the gradient must not be tracked through all
    iterations.

    The forward iterations use the root solvers of the vendored `xitorch
    <https://xitorch.readthedocs.io>`__. The backward pass is implemented in
    :mod:`.fixed_point` and gives exact first and second derivatives.
    """

    def scf(
        self, guess: Tensor, return_charges: bool = True
    ) -> Charges | Potential | Tensor:
        # TODO: Pass mixer options in `method` arg.
        # Currently ignored. Always "broyden1".

        # The gradient cannot be more accurate than the converged SCF, hence
        # the adjoint tolerance follows the SCF tolerance (unless given), but
        # must stay above the round-off of the precision in use
        eps = torch.finfo(guess.dtype).eps
        bck_options = {
            "atol": max(1e-10, 1e-2 * self.config.f_atol, 100 * eps),
            **self.bck_options,
        }

        # The stateless map neither reads nor writes `self`, so that the
        # function stored in the autograd graph cannot form a reference cycle.
        # Hence, `self._data` is only updated below, from the solution.
        q_converged, niter = equilibrium(
            fcn=self.stateless_map(),
            y0=guess,
            bck_options=bck_options,
            batched=self.config.batch_mode > 0,
            **self.fwd_options,
        )
        self._data.iter += niter

        # Converged state (density, eigenpairs, ...) from the solution. In
        # Fock mode, the converged quantity is the Hamiltonian itself.
        if self.config.scp_mode == labels.SCP_MODE_FOCK:
            self._data.hamiltonian = q_converged

        # To reconnect the H0 energy with the computational graph, we
        # compute one extra SCF cycle with strong damping (no SCF iteration).
        # Note that this is not required for SCF with full gradient tracking.
        # (see https://github.com/grimme-lab/dxtb/issues/124)
        if self.config.scp_mode == labels.SCP_MODE_CHARGE:
            mixer = Simple({**self.fwd_options, "damp": 1e-5})
            q_new = iterate_charges(
                q_converged,
                self._data,
                self.config,
                self.interactions,
                self.diagonalize,
            )
            q_converged = mixer.iter(q_new, q_converged)

        if return_charges is True:
            return self.converged_to_charges(q_converged)
        return q_converged
