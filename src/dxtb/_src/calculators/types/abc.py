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
Calculators: ABC
================

Abstract helper classes for reduction of boilerplace method definitions (i.e.,
duplication) in calculators.
"""
from abc import ABC, abstractmethod

from tad_mctc.molecule import Mol

from dxtb._src.typing import Tensor

__all__ = ["BaseCalculator"]


class PropertyNotImplementedError(NotImplementedError):
    """Raised if a calculator does not implement the requested property."""


class GetPropertiesMixin(ABC):
    """
    Mixin class which provides :meth:`get_energy`, :meth:`get_forces` and so on.

    Inheriting classes must implement :meth:`get_property`.
    """

    implemented_properties: list[str]
    """Names of implemented methods of the Calculator."""

    def get_implemented_properties(self) -> list[str]:
        return self.implemented_properties

    @abstractmethod
    def get_property(self, name: str, mol: Mol | None = None) -> Tensor:
        """
        Get the named property.

        Parameters
        ----------
        name : str
            Name of the property to get.
        mol : Mol, optional
            Molecule to get the property of.
        """

    def get_energy(self, mol: Mol | None = None) -> Tensor:
        return self.get_property("energy", mol)

    def get_potential_energy(self, mol: Mol | None = None) -> Tensor:
        return self.get_property("energy", mol)

    def get_forces(self, mol: Mol | None = None) -> Tensor:
        return self.get_property("forces", mol)

    def get_dipole_moment(self, mol: Mol | None = None) -> Tensor:
        return self.get_property("dipole", mol)

    def get_charges(self, mol: Mol | None = None) -> Tensor:
        return self.get_property("charges", mol)


class BaseCalculator(GetPropertiesMixin, ABC):
    """
    Base calculator for the extended tight-binding (xTB) models.

    This calculator provides analytical, autograd, and numerical versions of all
    properties.
    """

    def __init__(self, mol: Mol) -> None:
        """
        Initialize the calculator.

        Parameters
        ----------
        mol : Mol
            Molecule to calculate properties for.
        """
        self.results = {}

    @abstractmethod
    def calculate(self, mol: Mol, properties: list[str]) -> dict:
        """
        Calculate the requested properties of a molecule.

        Parameters
        ----------
        mol : Mol
            Molecule to calculate properties for.
        properties : list[str]
            List of properties to calculate.

        Returns
        -------
        dict
            Dictionary of calculated properties.
        """

    def get_property(
        self, name: str, mol: Mol | None = None, allow_calculation=True
    ) -> Tensor | None:
        """
        Get the named property.

        Parameters
        ----------
        name : str
            Name of the property to get.
        mol : Mol, optional
            Molecule to get the property of.
        allow_calculation : bool, optional
            If the property is not present, allow its calculation.
        """

        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(
                f"{name} property not implemented. Use one of: "
                f"{self.implemented_properties}."
            )

        if mol is None:
            mol = self.mol

        if name not in self.results:
            if not allow_calculation:
                return None

            if self.use_cache:
                self.mol = mol.copy()

            self.calculate(mol, [name])

        # For some reason the calculator was not able to do what we want...
        if name not in self.results:
            raise PropertyNotImplementedError(
                f"{name} not present in this calculation."
            )

        result = self.results[name]
        if isinstance(result, Tensor):
            result = result.clone()
        return result
