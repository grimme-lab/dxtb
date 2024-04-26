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

import warnings

import torch

from dxtb.components.interactions import Charges
from dxtb.constants import labels
from dxtb.typing import Tensor

from ..base import SCFResult
from ..mixer import Anderson, Mixer, Simple
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

        if charges is None:
            charges = Charges(mono=torch.zeros_like(self._data.occupation))
        if isinstance(charges, Tensor):
            charges = Charges(mono=charges)

        # TODO: This piece of code is used like twenty times (refactor?)
        if self.config.scp_mode == labels.SCP_MODE_CHARGE:
            guess = charges.as_tensor()
        elif self.config.scp_mode == labels.SCP_MODE_POTENTIAL:
            potential = self.charges_to_potential(charges)
            guess = potential.as_tensor()
        elif self.config.scp_mode == labels.SCP_MODE_FOCK:
            potential = self.charges_to_potential(charges)
            guess = self.potential_to_hamiltonian(potential)
        else:
            raise ValueError(
                f"Unknown convergence target (SCP mode) '{self.config.scp_mode}'."
            )

        # calculate charges in SCF without gradient tracking
        with torch.no_grad():
            scp_conv = self.scf(guess).as_tensor()

        # initialize the correct mixer with tolerances etc.
        mixer = self.config.mixer  # type: ignore
        if isinstance(mixer, str):
            mixers = {"anderson": Anderson, "simple": Simple}
            if mixer.casefold() not in mixers:
                raise ValueError(f"Unknown mixer '{mixer}'.")

            # select and init mixer
            mixer: Mixer = mixers[mixer.casefold()](
                self.fwd_options, is_batch=self.batched
            )

        # SCF step with gradient using converged result as "perfect" guess
        scp_new = self._fcn(scp_conv)
        scp = mixer.iter(scp_new, scp_conv)
        scp = self.converged_to_charges(scp)

        # Check consistency between SCF solution and single step.
        # Especially for elements and their ions, the SCF may oscillate and the
        # single step for the gradient may differ from the converged solution.
        if (
            torch.linalg.vector_norm(scp_conv - scp)
            > torch.finfo(self.dtype).eps ** 0.5 * 10
        ).any():
            warnings.warn(
                "The single SCF step differs from the converged solution. "
                "Re-calculating with full gradient tracking!"
            )
            charges = self.scf(scp_conv)

        charges.nullify_padding()
        energy = self.get_energy(charges)
        fenergy = self.get_electronic_free_energy()

        return {
            "charges": charges,
            "coefficients": self._data.evecs,
            "density": self._data.density,
            "emo": self._data.evals,
            "energy": energy,
            "fenergy": fenergy,
            "hamiltonian": self._data.hamiltonian,
            "occupation": self._data.occupation,
            "potential": self.charges_to_potential(charges),
            "iterations": self._data.iter,
        }
