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

from dxtb._src.constants import defaults
from dxtb._src.typing import Any, Literal, Tensor

__all__ = ["GetPropertiesMixin", "PropertyNotImplementedError"]


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
    def get_property(
        self,
        name: str,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        """
        Get the named property.

        Parameters
        ----------
        name : str
            Name of the property to get.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to 0.

        Returns
        -------
        Tensor
            The requested property.
        """

    # energy

    def get_energy(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        return self.get_property("energy", positions, chrg=chrg, spin=spin, **kwargs)

    def get_potential_energy(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        return self.get_property("energy", positions, chrg=chrg, spin=spin, **kwargs)

    # nuclear derivatives

    def get_forces(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        grad_mode: Literal["autograd", "backward", "functorch", "row"] = "autograd",
        **kwargs: Any
    ) -> Tensor:
        return self.get_property(
            "forces", positions, chrg=chrg, spin=spin, grad_mode=grad_mode, **kwargs
        )

    def get_hessian(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        return self.get_property("hessian", positions, chrg=chrg, spin=spin, **kwargs)

    def get_vibration(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        return self.get_property("vibration", positions, chrg=chrg, spin=spin, **kwargs)

    def get_normal_modes(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        return self.get_property(
            "normal_modes", positions, chrg=chrg, spin=spin, **kwargs
        )

    def get_frequencies(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        return self.get_property(
            "frequencies", positions, chrg=chrg, spin=spin, **kwargs
        )

    # field derivatives

    def get_dipole(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        return self.get_property("dipole", positions, chrg=chrg, spin=spin, **kwargs)

    def get_dipole_moment(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        return self.get_property("dipole", positions, chrg=chrg, spin=spin, **kwargs)

    # SCF properties

    def get_charges(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any
    ) -> Tensor:
        return self.get_property("charges", positions, chrg=chrg, spin=spin, **kwargs)
