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
SCF Implicit: Base
==================

Base class for all SCF implementations that make use of the implicit function
theorem in the backward pass.
"""

from __future__ import annotations

import copy

import torch

from dxtb._src.exlibs import xitorch as xt
from dxtb._src.timing.decorator import timer_decorator
from dxtb._src.typing import Callable, Tensor

from ..base import BaseSCF
from ..pure.iterations import iter_options

__all__ = ["BaseXSCF"]


class BaseXSCF(BaseSCF):
    """
    Base class for the implicitly differentiated self-consistent field
    iterator.

    This base class implements the `get_overlap` and the `diagonalize` methods
    that use `xitorch`'s `LinearOperator`s and its symmetric eigensolver, and
    the stateless fixed-point map.

    This class only lacks the `scf` method, which implements mixing and
    convergence.
    """

    def get_overlap(self) -> xt.LinearOperator:
        """
        Get the overlap matrix.

        Returns
        -------
        LinearOperator
            Overlap matrix.
        """

        smat = self._data.ints.overlap

        zeros = torch.eq(smat, 0)
        mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

        return xt.LinearOperator.m(
            smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)
        )

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
        h_op = xt.LinearOperator.m(hamiltonian)
        o_op = self.get_overlap()

        return xt.linalg.lsymeig(A=h_op, M=o_op, **self.eigen_options)

    def stateless_map(self) -> Callable[[Tensor], Tensor]:
        """
        Fixed-point map ``x -> g(x)`` that neither reads nor writes ``self``.

        All tensors that can carry gradients (integrals, interaction caches)
        are reached through a frozen shallow copy of the SCF data, and every
        call works on its own scratch copy of that snapshot. The returned
        function therefore holds no reference to this object or to anything
        that is later attached to the output of the SCF (e.g., ``self._data``
        after ``scf``), which would form a reference cycle through the autograd
        graph.

        Returns
        -------
        Callable[[Tensor], Tensor]
            The map for the current convergence target (SCP mode).
        """
        template = copy.copy(self._data)
        cfg = copy.copy(self.config)
        cfg.eigen_options = self.eigen_options
        interactions = self.interactions
        fcn = iter_options[self.config.scp_mode]

        def g(x: Tensor) -> Tensor:
            return fcn(x, copy.copy(template), cfg, interactions)

        return g
