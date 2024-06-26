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
SCF Unrolling: Single-Shot Variant
==================================

Standard SCF implementation with single-shot gradient calculation.

The single-shot gradient tracking was a first idea to reduce time and memory
consumption in the SCF. Here, the SCF is performed outside of the purview
of the computational graph. Only after convergence, an additional SCF step
with enabled gradient tracking is performed to reconnect to the autograd
engine. However, this approach is not correct as a derivative w.r.t. the
input features is missing. Apparently, the deviations are small if the
input features do not change much.
"""

from __future__ import annotations

import torch

from dxtb import OutputHandler
from dxtb._src.components.interactions import Charges
from dxtb._src.typing import Tensor, override

from ..base import SCFResult
from .default import SelfConsistentFieldFull

__all__ = ["SelfConsistentFieldSingleShot"]


class SelfConsistentFieldSingleShot(SelfConsistentFieldFull):
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    .. warning:

        Do not use in production. The gradients of the single-shot method
        are not exact (derivative w.r.t. the input features is missing).
    """

    @override
    def __call__(self, charges: Charges | Tensor | None = None) -> SCFResult:
        """
        Run the self-consistent iterations until a stationary solution is reached

        Parameters
        ----------
        charges : Tensor, optional
            Initial orbital charges vector.

        Returns
        -------
        Tensor
            Converged orbital charges vector.
        """

        guess = super()._guess(charges)

        # calculate charges in SCF without gradient tracking
        with torch.no_grad():
            OutputHandler.write_stdout(
                f"\n{'iter':<5} {'Energy':<24} {'Delta E':<16}"
                f"{'Delta Pnorm':<15} {'Delta q':<15}",
                v=3,
            )
            OutputHandler.write_stdout(77 * "-", v=3)

            # Normally, the SCF always returns charges, but we need to pass the
            # converged quantity to `self._fcn`, and hence, need the property
            # selected for convergence directly.
            scp_nograd = self.scf(guess, return_charges=False)
            if not isinstance(scp_nograd, Tensor):
                scp_nograd = scp_nograd.as_tensor()

            OutputHandler.write_stdout(77 * "-", v=3)

        # SCF step with gradient using converged result as "perfect" guess.
        # This is not exact, as the implicit derivative w.r.t. the variational
        # parameters is missing. The gradient only comes in through the
        # Hamiltonian (integrals) in ``potential_to_hamiltonian``.
        scp_grad = self._fcn(scp_nograd)
        OutputHandler.write_stdout(77 * "-", v=3)

        def _norm_nograd(q1, q2):
            with torch.no_grad():
                return torch.linalg.vector_norm(q1 - q2)

        # Check consistency between SCF solution and single step.
        # Especially for elements and their ions, the SCF may oscillate and the
        # single step for the gradient may differ from the converged solution.
        if (_norm_nograd(scp_grad, scp_nograd) > self.config.f_atol).any():
            OutputHandler.warn(
                "The single SCF step differs from the converged solution. "
                "Trying again with mixing!"
            )

            scp_grad = self.mixer.iter(scp_grad, scp_nograd)

            if (_norm_nograd(scp_grad, scp_nograd) > self.config.f_atol).any():
                OutputHandler.warn(
                    "The single SCF step differs from the converged solution. "
                    "Re-calculating with full gradient tracking!"
                )

                self.mixer.reset()
                q = self.scf(scp_nograd)
                OutputHandler.write_stdout(77 * "-", v=3)
            else:
                q = self.converged_to_charges(scp_grad)
        else:
            q = self.converged_to_charges(scp_grad)

        OutputHandler.write_stdout("", v=3)

        q.nullify_padding()
        energy = self.get_energy(q)
        fenergy = self.get_electronic_free_energy()

        return {
            "charges": q,
            "coefficients": self._data.evecs,
            "density": self._data.density,
            "emo": self._data.evals,
            "energy": energy,
            "fenergy": fenergy,
            "hamiltonian": self._data.hamiltonian,
            "occupation": self._data.occupation,
            "potential": self.charges_to_potential(q),
            "iterations": self._data.iter,
        }
