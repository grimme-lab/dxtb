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
theorem via `xitorch` for the backward.
"""

from __future__ import annotations

from dxtb._src.components.interactions import Charges, Potential
from dxtb._src.constants import labels
from dxtb._src.typing import Tensor

from ..mixer import Simple
from .base import BaseXSCF

__all__ = ["SelfConsistentFieldImplicit"]


class SelfConsistentFieldImplicit(BaseXSCF):
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    The default class makes use of the implicit function theorem. Hence, the
    derivatives of the iterative procedure are only calculated from the
    equilibrium solution, i.e., the gradient must not be tracked through all
    iterations.

    The implementation is based on `xitorch <https://xitorch.readthedocs.io>`__,
    which appears to be abandoned and unmaintained at the time of
    writing, but still provides a reasonably good implementation of the
    iterative solver required for the self-consistent field iterations.
    """

    def scf(
        self, guess: Tensor, return_charges: bool = True
    ) -> Charges | Potential | Tensor:
        # pylint: disable=import-outside-toplevel
        from dxtb._src.exlibs import xitorch as xt

        fcn = self._fcn

        # TODO: Pass mixer options in `method` arg.
        # Currently ignored. Always "broyden1".
        q_converged = xt.optimize.equilibrium(
            fcn=fcn,
            y0=guess,
            bck_options={**self.bck_options},
            **self.fwd_options,
        )

        # To reconnect the H0 energy with the computational graph, we
        # compute one extra SCF cycle with strong damping.
        # Note that this is not required for SCF with full gradient tracking.
        # (see https://github.com/grimme-lab/dxtb/issues/124)
        if self.config.scp_mode == labels.SCP_MODE_CHARGE:
            mixer = Simple({**self.fwd_options, "damp": 1e-4})
            q_new = fcn(q_converged)
            q_converged = mixer.iter(q_new, q_converged)

        if return_charges is True:
            return self.converged_to_charges(q_converged)
        return q_converged
