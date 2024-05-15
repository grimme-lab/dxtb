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

import torch

from dxtb._src.exlibs import xitorch as xt
from dxtb._src.timing.decorator import timer_decorator
from dxtb._src.typing import Tensor

from ..base import BaseSCF

__all__ = ["BaseXSCF"]


class BaseXSCF(BaseSCF, xt.EditableModule):
    """
    Base class for the `xitorch`-based self-consistent field iterator.

    This base class implements the `get_overlap` and the `diagonalize` methods
    that use `LinearOperator`s. Additionally, `getparamnames` is implemented,
    which is mandatory for all descendents of `xitorch`s base class called
    `EditableModule`.

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

    def getparamnames(
        self, methodname: str, prefix: str = ""
    ) -> list[str]:  # pragma: no cover
        if methodname == "scf":
            a = self.getparamnames("iterate_potential")
            b = self.getparamnames("charges_to_potential")
            c = self.getparamnames("potential_to_charges")
            return a + b + c

        if methodname == "get_energy":
            return [prefix + "_data.energy"]

        if methodname == "iterate_charges":
            a = self.getparamnames("charges_to_potential", prefix=prefix)
            b = self.getparamnames("potential_to_charges", prefix=prefix)
            return a + b

        if methodname == "iterate_potential":
            a = self.getparamnames("potential_to_charges", prefix=prefix)
            b = self.getparamnames("charges_to_potential", prefix=prefix)
            return a + b

        if methodname == "iterate_fockian":
            a = self.getparamnames("hamiltonian_to_density", prefix=prefix)
            b = self.getparamnames("density_to_charges", prefix=prefix)
            c = self.getparamnames("charges_to_potential", prefix=prefix)
            d = self.getparamnames("potential_to_hamiltonian", prefix=prefix)
            return a + b + c + d

        if methodname == "charges_to_potential":
            return []

        if methodname == "potential_to_charges":
            a = self.getparamnames("potential_to_density", prefix=prefix)
            b = self.getparamnames("density_to_charges", prefix=prefix)
            return a + b

        if methodname == "potential_to_density":
            a = self.getparamnames("potential_to_hamiltonian", prefix=prefix)
            b = self.getparamnames("hamiltonian_to_density", prefix=prefix)
            return a + b

        if methodname == "density_to_charges":
            return [
                prefix + "_data.ints.hcore",
                prefix + "_data.ints.overlap",
                prefix + "_data.n0",
            ]

        if methodname == "potential_to_hamiltonian":
            return [
                prefix + "_data.ints.hcore",
                prefix + "_data.ints.overlap",
            ]

        if methodname == "hamiltonian_to_density":
            a = [prefix + "_data.occupation"]
            b = self.getparamnames("diagonalize", prefix=prefix)
            c = self.getparamnames("get_overlap", prefix=prefix)
            return a + b + c

        if methodname == "get_overlap":
            return [prefix + "_data.ints.overlap"]

        if methodname == "diagonalize":
            return []

        raise KeyError(f"Method '{methodname}' has no paramnames set")
